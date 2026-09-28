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
    "model": "realesr-general-x4v3",
    "outscale": 1.5,
    "prefix": "assets/production/x-720p/",
    "public_base": "https://pub-....r2.dev"
  }
}
```

Response:
```json
{"count":2,"ok":2,"results":[{"input":"...","key":"...","url":"...","width":720,"height":1280,"fps":30,"duration":13.1,"frames":392,"elapsed":21.4}]}
```
