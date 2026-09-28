"""RunPod Serverless worker: Real-ESRGAN video upscaling (480p -> 720p).

Input (event["input"]):
  videos:            list[str] | list[{"url": str, "key": str?}]   required
  model:             "realesr-general-x4v3" | "RealESRGAN_x4plus"  (default realesr-general-x4v3)
  outscale:          float, final scale vs input (default 1.5)
  prefix:            R2 key prefix for outputs (default "")
  public_base:       public URL base for returned links (default env R2_PUBLIC_BASE_URL)
  bucket:            override bucket (default env R2_BUCKET)
  tile:              int, 0 = no tiling (default 0)
  fp32:              bool, use fp32 (default false = fp16)
  preset:            x264 preset when not using NVENC (default env X264_PRESET or "veryfast")
  vcodec:            "auto" | "libx264" | "nvenc"  (default "auto")

Env: R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, R2_PUBLIC_BASE_URL
"""
import os
import json
import time
import subprocess
import tempfile
import traceback
import urllib.request

import numpy as np
import torch

import cv2

torch.backends.cudnn.benchmark = True

MODELS = {
    "realesr-general-x4v3": {
        "path": "/models/realesr-general-x4v3.pth",
        "netscale": 4,
        "arch": ("srvgg", 32),
    },
    "RealESRGAN_x4plus": {
        "path": "/models/RealESRGAN_x4plus.pth",
        "netscale": 4,
        "arch": ("rrdb", 23),
    },
}

_UPSAMPLERS = {}
_NVENC = None


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
    global _NVENC
    if _NVENC is None:
        try:
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
                 "-c:v", "h264_nvenc", "-f", "null", "-"],
                capture_output=True, timeout=40,
            )
            _NVENC = (r.returncode == 0)
        except Exception:
            _NVENC = False
    return _NVENC


def _build_model(name):
    from basicsr.archs.srvgg_arch import SRVGGNetCompact
    from basicsr.archs.rrdbnet_arch import RRDBNet

    cfg = MODELS[name]
    kind, n = cfg["arch"]
    if kind == "srvgg":
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=n,
                               upscale=4, act_type="prelu")
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=n,
                   num_grow_ch=32, scale=4)


def get_upsampler(name, tile=0, half=True):
    key = (name, tile, half)
    if key in _UPSAMPLERS:
        return _UPSAMPLERS[key]
    from realesrgan import RealESRGANer

    cfg = MODELS[name]
    info = cuda_info()
    if info["device"] != "cuda":
        raise RuntimeError(f"CUDA not available in worker: {info}")
    up = RealESRGANer(
        scale=cfg["netscale"],
        model_path=cfg["path"],
        model=_build_model(name),
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=half,
        device="cuda",
    )
    _UPSAMPLERS[key] = up
    return up


def ffprobe(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-show_entries", "format=duration", "-of", "json", path,
    ])
    d = json.loads(out)
    st = d["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    duration = float(d["format"]["duration"])
    has_audio = bool(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True,
    ).stdout.strip())
    return st["width"], st["height"], fps, duration, has_audio


def enhance_batch(frames, upsampler, outscale):
    """Run the ESR model on a batch of BGR uint8 frames, doing the downscale on GPU."""
    import torch.nn.functional as F

    arr = np.stack(frames)  # N,H,W,3 BGR uint8
    t = torch.from_numpy(arr).to(upsampler.device, non_blocking=True)
    t = t.permute(0, 3, 1, 2).flip(1).contiguous()  # N,3,H,W (RGB)
    t = (t.half() if upsampler.half else t.float()).div_(255.0)
    with torch.no_grad():
        out = upsampler.model(t)
    out = out.float().clamp_(0, 1)
    if outscale != float(upsampler.scale):
        h, w = frames[0].shape[:2]
        out = F.interpolate(out, size=(int(h * outscale), int(w * outscale)),
                            mode="bicubic", align_corners=False).clamp_(0, 1)
    out = (out * 255.0).round_().to(torch.uint8).permute(0, 2, 3, 1)  # N,H,W,3 RGB
    arr_out = out.cpu().numpy()
    return [np.ascontiguousarray(o[:, :, ::-1]) for o in arr_out]  # RGB -> BGR


def upscale_video(src, dst, upsampler, outscale, vcodec="auto", preset=None, batch=4):
    w, h, fps, duration, has_audio = ffprobe(src)
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
        wcmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy", "-shortest"]
    else:
        wcmd += ["-map", "0:v:0"]
    wcmd += enc + [dst]
    writer = subprocess.Popen(wcmd, stdin=subprocess.PIPE)

    frames = 0
    enhance_time = 0.0
    frame_bytes = w * h * 3
    frame_batch = max(1, int(batch))
    pending = []

    def flush(pending):
        nonlocal frames, enhance_time
        if not pending:
            return
        t0 = time.time()
        outs = enhance_batch(pending, upsampler, outscale)
        enhance_time += time.time() - t0
        for o in outs:
            writer.stdin.write(o[:out_h, :out_w].tobytes())
            frames += 1

    try:
        while True:
            buf = reader.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            pending.append(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
            if len(pending) >= frame_batch:
                flush(pending)
                pending = []
        flush(pending)
    finally:
        try:
            writer.stdin.close()
        except Exception:
            pass
        writer.wait()
        reader.stdout.close()
        reader.wait()

    return {"width": out_w, "height": out_h, "fps": fps, "duration": duration,
            "frames": frames, "vcodec": codec, "frame_batch": frame_batch,
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
    videos = inp.get("videos") or []
    if not videos:
        return {"error": "no videos provided"}

    model = inp.get("model", "realesr-general-x4v3")
    if model not in MODELS:
        return {"error": f"unknown model {model}; choose {list(MODELS)}"}
    outscale = float(inp.get("outscale", 1.5))
    tile = int(inp.get("tile", 0))
    vcodec = inp.get("vcodec", "auto")
    preset = inp.get("preset")
    prefix = inp.get("prefix") or ""
    bucket = inp.get("bucket") or os.environ["R2_BUCKET"]
    public_base = (inp.get("public_base") or os.environ.get("R2_PUBLIC_BASE_URL", "")).rstrip("/")

    from boto3.s3.transfer import TransferConfig
    s3 = s3_client()
    tc = TransferConfig(max_concurrency=16)

    info = cuda_info()
    t_load = time.time()
    up = get_upsampler(model, tile=tile, half=not inp.get("fp32", False))
    load_time = round(time.time() - t_load, 2)

    from concurrent.futures import ThreadPoolExecutor

    info = cuda_info()
    t_load = time.time()
    up = get_upsampler(model, tile=tile, half=not inp.get("fp32", False))
    load_time = round(time.time() - t_load, 2)
    frame_batch = inp.get("frame_batch", 4)
    concurrency = max(1, int(inp.get("concurrency", 1)))

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
                meta = upscale_video(src, dst, up, outscale, vcodec=vcodec, preset=preset,
                                     batch=frame_batch)
                s3.upload_file(dst, bucket, key, Config=tc,
                               ExtraArgs={"ContentType": "video/mp4"})
            entry.update(meta)
            entry["download_time"] = dl_time
            entry["url"] = f"{public_base}/{key}"
            entry["elapsed"] = round(time.time() - t0, 2)
        except Exception as e:
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
        "model": model,
        "outscale": outscale,
        "concurrency": concurrency,
        "frame_batch": frame_batch,
        "runtime": info,
        "model_load_time": load_time,
        "results": results,
    }


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
