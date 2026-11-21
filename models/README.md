# Models

`yolo11x.pt`: Ultralytics YOLO11x, COCO-pretrained, official release (stored with Git LFS, 115 MB).
sha256 `7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24`. Not fine-tuned.

Used by `flow_estimation/detect.py` with `imgsz=1280` on 512×288 city-camera images. If the file is missing
(no Git LFS), `detect.py` loads `yolo11x.pt` by name and Ultralytics downloads the same weights.

The fitted calibration (thresholds and conversion factors β) that goes with this detector is
`flow_estimation/calibration/yolo11x_1280.json`.
