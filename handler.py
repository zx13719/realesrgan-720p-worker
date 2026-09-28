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

Env (endpoint env):
  R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, R2_PUBLIC_BASE_URL

Output:
  {"count": n, "results": [{"input","key","url","width","height","fps","duration","frames"}]}
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    up = RealESRGANer(
        scale=cfg["netscale"],
        model_path=cfg["path"],
        model=_build_model(name),
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=(half and device == "cuda"),
        device=device,
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


def upscale_video(src, dst, upsampler, outscale):
    w, h, fps, duration, has_audio = ffprobe(src)
    out_w = int(w * outscale); out_w -= out_w % 2
    out_h = int(h * outscale); out_h -= out_h % 2

    reader = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        stdout=subprocess.PIPE,
    )
    wcmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", "pipe:0"]
    if has_audio:
        wcmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy", "-shortest"]
    else:
        wcmd += ["-map", "0:v:0"]
    wcmd += ["-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", dst]
    writer = subprocess.Popen(wcmd, stdin=subprocess.PIPE)

    frames = 0
    frame_bytes = w * h * 3
    try:
        while True:
            buf = reader.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            out, _ = upsampler.enhance(frame, outscale=outscale)
            out = out[:out_h, :out_w]
            writer.stdin.write(out.astype(np.uint8).tobytes())
            frames += 1
    finally:
        try:
            writer.stdin.close()
        except Exception:
            pass
        writer.wait()
        reader.stdout.close()
        reader.wait()

    return {"width": out_w, "height": out_h, "fps": fps,
            "duration": duration, "frames": frames}


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
    prefix = inp.get("prefix") or ""
    bucket = inp.get("bucket") or os.environ["R2_BUCKET"]
    public_base = (inp.get("public_base") or os.environ.get("R2_PUBLIC_BASE_URL", "")).rstrip("/")

    from boto3.s3.transfer import TransferConfig
    s3 = s3_client()
    tc = TransferConfig(max_concurrency=16)
    up = get_upsampler(model, tile=tile, half=not inp.get("fp32", False))

    results = []
    for item in videos:
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
                urllib.request.urlretrieve(url, src)
                meta = upscale_video(src, dst, up, outscale)
                s3.upload_file(dst, bucket, key, Config=tc,
                               ExtraArgs={"ContentType": "video/mp4"})
            entry.update(meta)
            entry["url"] = f"{public_base}/{key}"
            entry["elapsed"] = round(time.time() - t0, 2)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = traceback.format_exc()[-800:]
        results.append(entry)

    return {
        "count": len(results),
        "ok": sum(1 for r in results if "error" not in r),
        "model": model,
        "outscale": outscale,
        "results": results,
    }


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
