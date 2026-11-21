# Data

| Path | In repo | Content |
|---|---|---|
| `calibration/videos.json` | yes | ground truth of the 298 calibration videos: camera, duration, moving vehicles per class |
| `calibration/frames.jsonl` | yes | 35,481 snapshots on a 5 s grid: `{video, camera, T, pts, image}` (`image` = `<video>/<frame>.jpg`) |
| `calibration/camera_folds.json` | yes | fixed camera → fold (0/1/2) assignment used for cross-validation |
| `calibration/images/` | no | the 512×288 snapshots (made by `build_calibration_set.py` from the source videos) |
| `detections/calibration_yolo11x_1280.jsonl.gz` | yes | YOLO11x @1280 boxes of every calibration snapshot |
| `detections/city_cam_yolo11x_1280.jsonl.gz` | yes | YOLO11x @1280 boxes of every image of the 4 real cameras (3 sessions) |
| `detections/city_cam_frames.jsonl` | yes | frame index of the real cameras: `{session, camera, t (receive time, +07:00), image}` |
| `city_cam/` | no | raw city-camera sessions (images + metadata), see below |

Detection lines are `{"image": key, "dets": [[cls, x_center, y_bottom, height, width, conf], ...]}` in 1920×1080
pixel space, kept down to conf 0.08. With these caches, calibration, inference, demand settings and scenarios
can all be reproduced without images or a GPU.

## Calibration videos (not redistributed)

The 298 videos (lot N, 98 HCMC traffic cameras, ~10 min each) come from the AI Challenge HCMC dataset, and their
tracks from our tracking run. `build_calibration_set.py` documents the expected input format. The derived ground
truth counts and detections are included so that `calibrate.py` runs as is.

## City-camera sessions (not redistributed)

Collected from the HCMC traffic portal (giaothong.hochiminhcity.gov.vn) by polling every 5 s or 15 s and keeping
only images that differ from the previous one. Expected layout under `data/city_cam/`:

```
data/city_cam/
├── 2026-10-29_0850-1250/     29 Oct 2026 08:50–12:50, gaps ~10–15 s
├── 2026-11-07_1306-1706/     07 Nov 2026 13:06–17:06, gaps ~15 s
└── 2026-11-12_0530-0700/     12 Nov 2026 05:30–07:00, gaps ~10–15 s
    └── Cam_<name>/
        ├── metadata.csv      filename, received_at_vn, http_date, request_url, sha256, ...
        └── frame_NNNNNN_YYYYMMDD_HHMMSS_mmm.jpg   512×288
```

Cameras: `Cam_Ly_Thuong_Kiet_Bac_Hai`, `Cam_Thanh_Thai_To_Hien_Thanh`, `Cam_To_Hien_Thanh_1`,
`Cam_To_Hien_Thanh_2` (used), and `Cam_To_Hien_Thanh_Su_Van_Hanh` (collected, not used).
Exposure times are unknown; the receive time is used. A few gaps of 2–2.5 min exist; pairs more than 35 s apart
are treated as missing data.
