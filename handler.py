"""RunPod Serverless worker: Real-ESRGAN video upscaling (e.g. 480p -> 720p).

Input (event["input"]):
  videos:            list[str] | list[{"url": str, "key": str?}]   required
  model:             "realesr-general-x4v3" | "RealESRGAN_x2plus" | "RealESRGAN_x4plus"
                     (default realesr-general-x4v3; x2plus is best when outscale~1.5)
  outscale:          float, final scale vs input (default 1.5)
  prefix:            R2 key prefix for outputs (default "")
  public_base:       public URL base for returned links (default env R2_PUBLIC_BASE_URL)
  bucket:            override bucket (default env R2_BUCKET)
  fp32:              bool, use fp32 (default false = fp16)
  channels_last:     bool, NHWC memory format (default true)
  frame_batch:       int, frames per GPU forward (default env ESR_FRAME_BATCH or 8)
  concurrency:       int, videos processed in parallel (default env ESR_CONCURRENCY or 1)
  preset:            x264 preset when not using NVENC (default env X264_PRESET or "veryfast")
  vcodec:            "auto" | "libx264" | "nvenc"  (default "auto")
  compile:           bool, torch.compile the model (default false; one-off warm-up cost)

Env: R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, R2_PUBLIC_BASE_URL
"""
import os
import json
import queue
import threading
import time
import subprocess
import tempfile
import traceback
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

MODELS = {
    "realesr-general-x4v3": "/models/realesr-general-x4v3.pth",
    "RealESRGAN_x2plus": "/models/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus": "/models/RealESRGAN_x4plus.pth",
}

_MODELS = {}
_NVENC = None
_NVENC_ERROR = None


def cuda_info():
    avail = torch.cuda.is_available()
    return {
        "device": "cuda" if avail else "cpu",
        "cuda_available": avail,
        "gpu": torch.cuda.get_device_name(0) if avail else None,
        "capability": torch.cuda.get_device_capability(0) if avail else None,
        "arch_list": torch.cuda.get_arch_list() if avail else [],
    }


def nvenc_available():
    """Probe once whether this build/driver exposes h264_nvenc."""
    global _NVENC, _NVENC_ERROR
    if _NVENC is None:
        try:
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
                 "-c:v", "h264_nvenc", "-f", "null", "-"],
                capture_output=True, timeout=40,
            )
            _NVENC = (r.returncode == 0)
            _NVENC_ERROR = None if _NVENC else r.stderr.decode(errors="replace")[-800:]
        except Exception as exc:
            _NVENC = False
            _NVENC_ERROR = f"{type(exc).__name__}: {exc}"[:800]
    return _NVENC


def codec_info():
    """A real encode probe, not merely a compiled-in encoder listing."""
    available = nvenc_available()
    return {
        "nvenc_available": available,
        "nvenc_error": _NVENC_ERROR,
        "driver_capabilities": os.environ.get("NVIDIA_DRIVER_CAPABILITIES"),
    }


def get_model(name, half=True, channels_last=True, compile_model=False):
    """Load a .pth via spandrel (no basicsr/realesrgan needed).

    Returns (model, scale). Cached per config.
    """
    key = (name, half, channels_last, compile_model)
    if key in _MODELS:
        return _MODELS[key]
    from spandrel import ModelLoader

    info = cuda_info()
    if info["device"] != "cuda":
        raise RuntimeError(f"CUDA not available in worker: {info}")
    desc = ModelLoader().load_from_file(MODELS[name])
    scale = float(desc.scale)
    model = desc.cuda()
    model = model.half() if half else model.float()
    model = model.eval()
    if channels_last:
        # spandrel's descriptor overrides .to() and rejects memory_format;
        # apply it on the wrapped nn.Module instead.
        try:
            getattr(model, "model", model).to(memory_format=torch.channels_last)
        except Exception:
            pass
    if compile_model:
        try:
            model = torch.compile(model)
        except Exception:
            pass
    _MODELS[key] = (model, scale)
    return model, scale


def ffprobe(path):
    """Single ffprobe call: width, height, fps, duration, frames, has_audio."""
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path,
    ])
    d = json.loads(out)
    st = next(s for s in d["streams"] if s["codec_type"] == "video")
    num, den = st["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    duration = float(d["format"]["duration"])
    nb = st.get("nb_frames")
    frames = int(nb) if nb and nb not in ("N/A", "0") else round(duration * fps)
    has_audio = any(s["codec_type"] == "audio" for s in d["streams"])
    return st["width"], st["height"], fps, duration, frames, has_audio


def count_frames(path):
    """Exact decoded video frame count (catches silent truncation)."""
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "json", path,
    ])
    return int(json.loads(out)["streams"][0]["nb_read_frames"])


def enhance_batch(frames, model, outscale, scale, half=True, channels_last=True):
    """Run the ESR model on a batch of BGR uint8 frames, downscale on GPU. Returns BGR uint8."""
    device = next(getattr(model, "model", model).parameters()).device
    arr = np.stack(frames)  # N,H,W,3 BGR (contiguous)
    t = torch.from_numpy(arr).to(device, non_blocking=True).permute(0, 3, 1, 2).flip(1)  # RGB
    t = t.contiguous(memory_format=torch.channels_last) if channels_last else t.contiguous()
    t = (t.half() if half else t.float()).div_(255.0)
    with torch.inference_mode():
        out = model(t)
        out = out.float().clamp(0, 1)
        if abs(outscale - float(scale)) > 1e-6:
            h, w = frames[0].shape[:2]
            out = F.interpolate(out, size=(int(h * outscale), int(w * outscale)),
                                mode="bicubic", align_corners=False).clamp(0, 1)
        arr_out = (out * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return [np.ascontiguousarray(o[:, :, ::-1]) for o in arr_out]


def upscale_video(src, dst, model, outscale, scale, vcodec="auto", preset=None, batch=8,
                  half=True, channels_last=True):
    """Decode -> (prefetch) -> GPU enhance -> encode, muxing the original audio."""
    w, h, fps, duration, expected_frames, has_audio = ffprobe(src)
    expected_frames = count_frames(src)
    out_w = int(w * outscale); out_w -= out_w % 2
    out_h = int(h * outscale); out_h -= out_h % 2

    reader = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        stdout=subprocess.PIPE,
    )

    use_nvenc = (vcodec == "nvenc") or (vcodec == "auto" and nvenc_available())
    if use_nvenc:
        enc = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19", "-pix_fmt", "yuv420p"]
        codec = "h264_nvenc"
    else:
        enc = ["-c:v", "libx264", "-preset", preset or os.environ.get("X264_PRESET", "veryfast"),
               "-crf", "17", "-pix_fmt", "yuv420p"]
        codec = "libx264"

    wcmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", "pipe:0"]
    if has_audio:
        # NOTE: no -shortest here; it truncates trailing video frames when audio is copied.
        wcmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    else:
        wcmd += ["-map", "0:v:0"]
    wcmd += enc + ["-movflags", "+faststart", dst]
    writer = subprocess.Popen(wcmd, stdin=subprocess.PIPE)

    frame_bytes = w * h * 3
    frame_batch = max(1, int(batch))

    # Decode runs in a producer thread so the GPU never waits on the ffmpeg pipe.
    q = queue.Queue(maxsize=max(2, frame_batch * 2))
    producer_error = []
    cancelled = threading.Event()

    def enqueue(item):
        while not cancelled.is_set():
            try:
                q.put(item, timeout=0.2)
                return
            except queue.Full:
                pass

    def produce():
        try:
            while not cancelled.is_set():
                buf = reader.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    if buf:
                        raise RuntimeError("decoder emitted an incomplete frame")
                    break
                enqueue(np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy())
        except Exception as e:  # noqa: BLE001
            producer_error.append(e)
        finally:
            enqueue(None)

    th = threading.Thread(target=produce, daemon=True)
    th.start()

    frames = 0
    enhance_time = 0.0
    pending = []

    def flush():
        nonlocal frames, enhance_time
        if not pending:
            return
        t0 = time.time()
        outs = enhance_batch(pending, model, outscale, scale, half=half,
                             channels_last=channels_last)
        enhance_time += time.time() - t0
        for o in outs:
            writer.stdin.write(o[:out_h, :out_w].tobytes())
            frames += 1
        pending.clear()

    try:
        while True:
            item = q.get()
            if item is None:
                break
            pending.append(item)
            if len(pending) >= frame_batch:
                flush()
        flush()
        reader.wait(timeout=10)
    finally:
        cancelled.set()
        # On GPU/encoder failure the producer can be blocked on the decoder pipe.
        if reader.poll() is None:
            reader.terminate()
        th.join(timeout=5)
        try:
            writer.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            writer.wait(timeout=60)
        except subprocess.TimeoutExpired:
            writer.kill()
            writer.wait()
            raise RuntimeError("encoder did not exit within 60 seconds")
        reader.stdout.close()
        try:
            reader.wait(timeout=10)
        except subprocess.TimeoutExpired:
            reader.kill()
            reader.wait()
    if writer.returncode:
        raise RuntimeError(f"encoder exited with status {writer.returncode}")
    if reader.returncode:
        raise RuntimeError(f"decoder exited with status {reader.returncode}")
    if producer_error:
        raise producer_error[0]

    # Hard guard: never ship a silently truncated video (the v9 -shortest regression).
    actual = count_frames(dst)
    if actual != expected_frames:
        raise RuntimeError(
            f"frame mismatch: source={expected_frames} output={actual} "
            f"({expected_frames - actual} lost)")

    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", dst,
                    "-f", "null", "-"], check=True, timeout=300)

    return {"width": out_w, "height": out_h, "fps": fps, "duration": duration,
            "frames": frames, "verified_frames": actual,
            "vcodec": codec, "frame_batch": frame_batch,
            "enhance_time": round(enhance_time, 2),
            "enhance_fps": round(frames / enhance_time, 2) if enhance_time > 0 else None}


def download(url, dst):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RunPod-ESR/1.0)"}
    last = None
    for _ in range(4):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as r, open(dst, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(dst) > 0:
                return
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2)
    raise last if last else RuntimeError("download failed")


def s3_client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def handler(event):
    inp = event.get("input") or {}
    if inp.get("operation") == "diagnostics":
        return {"runtime": cuda_info(), "codecs": codec_info(),
                "revision": os.environ.get("IMAGE_REVISION", "unknown")}
    videos = inp.get("videos") or []
    if not videos:
        return {"error": "no videos provided"}

    model_name = inp.get("model", "realesr-general-x4v3")
    if model_name not in MODELS:
        return {"error": f"unknown model {model_name}; choose {list(MODELS)}"}
    outscale = float(inp.get("outscale", 1.5))
    half = not inp.get("fp32", False)
    channels_last = bool(inp.get("channels_last", True))
    compile_model = bool(inp.get("compile", False))
    vcodec = inp.get("vcodec", "auto")
    preset = inp.get("preset")
    prefix = inp.get("prefix") or ""
    bucket = inp.get("bucket") or os.environ["R2_BUCKET"]
    public_base = (inp.get("public_base") or os.environ.get("R2_PUBLIC_BASE_URL", "")).rstrip("/")
    frame_batch = int(inp.get("frame_batch", os.environ.get("ESR_FRAME_BATCH", 8)))
    concurrency = max(1, int(inp.get("concurrency", os.environ.get("ESR_CONCURRENCY", 1))))

    from concurrent.futures import ThreadPoolExecutor

    from boto3.s3.transfer import TransferConfig
    s3 = s3_client()
    tc = TransferConfig(max_concurrency=16)

    info = cuda_info()
    t_load = time.time()
    model, model_scale = get_model(model_name, half=half, channels_last=channels_last,
                                   compile_model=compile_model)
    load_time = round(time.time() - t_load, 2)

    def process_one(item):
        url = item["url"] if isinstance(item, dict) else item
        key = item.get("key") if isinstance(item, dict) else None
        base = os.path.splitext(os.path.basename(url.split("?")[0]))[0]
        if not key:
            key = f"{prefix}{base}_720p.mp4"
        entry = {"input": url, "key": key}
        t0 = time.time()
        try:
            with tempfile.TemporaryDirectory() as td:
                src = os.path.join(td, "in.mp4")
                dst = os.path.join(td, "out.mp4")
                td0 = time.time()
                download(url, src)
                dl_time = round(time.time() - td0, 2)
                meta = upscale_video(src, dst, model, outscale, model_scale, vcodec=vcodec,
                                     preset=preset, batch=frame_batch, half=half,
                                     channels_last=channels_last)
                s3.upload_file(dst, bucket, key, Config=tc,
                               ExtraArgs={"ContentType": "video/mp4"})
            entry.update(meta)
            entry["download_time"] = dl_time
            entry["url"] = f"{public_base}/{key}"
            entry["elapsed"] = round(time.time() - t0, 2)
        except Exception as e:  # noqa: BLE001
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = traceback.format_exc()[-800:]
        return entry

    if concurrency == 1:
        results = [process_one(it) for it in videos]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            results = list(ex.map(process_one, videos))

    return {
        "count": len(results),
        "ok": sum(1 for r in results if "error" not in r),
        "model": model_name,
        "outscale": outscale,
        "concurrency": concurrency,
        "frame_batch": frame_batch,
        "channels_last": channels_last,
        "runtime": info,
        "codecs": codec_info(),
        "model_load_time": load_time,
        "results": results,
    }


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
