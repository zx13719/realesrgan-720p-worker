# RunPod Serverless · Real-ESRGAN 720p video upscaler

Upscales short portrait videos (e.g. 480×854) to 720×1280 with Real-ESRGAN,
preserves the original audio, and writes the result to Cloudflare R2.

## Endpoint env
```
R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET=aimcn
R2_PUBLIC_BASE_URL=https://pub-....r2.dev
```

## Invoke

POST `https://api.runpod.ai/v2/<ENDPOINT_ID>/run` with `Authorization: Bearer <RUNPOD_API_KEY>`

```json
{
  "input": {
    "videos": [
      {"url": "https://.../clip-480p.mp4"},
      {"url": "https://.../other-480p.mp4", "key": "assets/production/x/other_720p.mp4"}
    ],
    "model": "RealESRGAN_x2plus",
    "outscale": 1.5,
    "frame_batch": 8,
    "concurrency": 1,
    "prefix": "assets/production/x-720p/",
    "public_base": "https://pub-....r2.dev"
  }
}
```

### Model choice

- `RealESRGAN_x2plus` — best for `outscale ≈ 1.5` (480p→720p). The model runs at 2×,
  so ~4× fewer output pixels than a 4× model; use this unless you need 4× detail.
- `realesr-general-x4v3` — compact SRVGG, 4× net; good default for heavier upscales.
- `RealESRGAN_x4plus` — heaviest, 4× RRDB.

### Tuning

- `frame_batch` (default 8): frames per GPU forward. Raise until VRAM-bound.
- `concurrency` (default 1): videos processed in parallel on one worker; helps hide
  ffmpeg decode/encode and R2 I/O. 2–3 is usually comfortable on a 24 GB card.
- `channels_last` (default true): NHWC path for fp16 convs.
- `compile` (default false): `torch.compile` the model; adds one-off warm-up cost.
- `vcodec`: `auto` uses `h264_nvenc` when available, else `libx264`.

### Env defaults

`ESR_FRAME_BATCH`, `ESR_CONCURRENCY`, `X264_PRESET` set the defaults so a workload
image can tune throughput without changing the request payload.

Response:
```json
{"count":2,"ok":2,"results":[{"input":"...","key":"...","url":"...","width":720,"height":1280,"fps":30,"duration":13.1,"frames":392,"enhance_fps":11.2,"elapsed":21.4}]}
```
