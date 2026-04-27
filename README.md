# Object Tracking / Detection

Object detection using **YOLOv3** (Darknet weights) through **OpenCV’s DNN** module, with optional **cvlib** helpers for quick detection and person counting. The main workflow lives in `ObjectDetectionFinal.ipynb`.

## What this project does

- **Loads YOLOv3** trained on the **COCO** dataset (80 classes) using `cv2.dnn.readNetFromDarknet`.
- **Runs detection on video**: reads frames, resizes, runs forward pass, applies **non-maximum suppression (NMS)**, draws bounding boxes and labels with **Matplotlib**.
- **Runs detection on still images** (BGR → RGB, same pipeline).
- **Counts people** in an image using **cvlib**’s `detect_common_objects` and `draw_bbox`.

## Repository layout

| Item | Description |
|------|-------------|
| `ObjectDetectionFinal.ipynb` | Notebook: setup, video loop, image demo, person count |

## Requirements

Install in your environment (Python 3 recommended):

```text
opencv-python
numpy
matplotlib
cvlib
```

On **Google Colab**, the notebook uses `wget` to fetch model files and assumes paths under `/content`. For **local Jupyter**, install the packages above and adjust paths (see below).

## Model files

The notebook downloads:

- `coco.names` — COCO class names  
- `yolov3.weights` — YOLOv3 weights (large file)  
- `yolov3.cfg` — network configuration  

In Colab these are expected under `/content`. Locally, place them in a folder of your choice and point `labelsPath`, `weightsPath`, `configPath`, and `readNetFromDarknet` to that folder.

## Running the notebook

1. Open `ObjectDetectionFinal.ipynb` in **Jupyter** or **Google Colab**.
2. Run the first cell to download weights/config and build the `detector` net.
3. **Video cell**: set `cv2.VideoCapture` to your video path. The sample uses Google Drive (`/content/drive/MyDrive/...`); change this to your file path.
4. **Image cells**: set `cv2.imread` to your image path instead of Drive paths if you are not on Colab.

### Path checklist for local use

- Replace `/content/...` with paths on your machine.
- Replace `from google.colab.patches import cv2_imshow` if you use plain OpenCV display (`cv2.imshow`) or keep Matplotlib only.
- Ensure the YOLO input size (e.g. `416×416` in `blobFromImage`) matches what you expect for speed vs. accuracy.

## Parameters (tuning)

- **Confidence threshold**: `0.5` in the detection loop (raise to reduce false positives).
- **NMS**: `cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)` — score threshold `0.5`, NMS threshold `0.4`.

## Notes

- **YOLOv3** is older than YOLOv5/v8 but is a common teaching setup with OpenCV DNN and no extra frameworks.
- **Video + `plt.imshow` in a loop** can be slow; for production you would typically write frames to a file or use a proper video display pipeline.
- The first cell imports **cvlib** but the core DNN path uses only OpenCV + NumPy + Matplotlib; cvlib is used in the person-counting cell.

## License

Model weights and `coco.names` / cfg originate from the Darknet/YOLO ecosystem; respect their licenses when redistributing weights or derived work.
