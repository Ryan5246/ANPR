# 🚗 Automatic Number Plate Recognition (ANPR) & OCR System

An end-to-end Automatic Number Plate Recognition (ANPR) system built using **YOLO** for high-accuracy license plate detection and **PaddleOCR** for text extraction. It features an interactive OpenCV video interface with timeline scrubbing, play/pause trackbars, and optimized frame-skipping logic to maximize real-time processing performance.

## ✨ Features
* **Real-Time License Plate Detection:** High-speed object detection powered by Ultralytics YOLO (`imgsz=416`, `conf=0.65`).
* **Text Extraction (OCR):** Uses **PaddleOCR (`TextRecognition`)** on cropped detection boxes to extract plate numbers automatically.
* **CPU Performance Optimization:** Adjustable frame interval (`OCR_INTERVAL = 6`) prevents CPU bottlenecks during video playback.
* **Interactive OpenCV GUI:** Custom window controls featuring dynamic timeline scrubbing and a live Play/Pause toggle.
* **Custom Dataset Integration:** Trained on a custom dataset managed and formatted using **Roboflow**.

## 🛠️ Tech Stack & Dependencies
* **Object Detection:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* **Optical Character Recognition:** [PaddleOCR / PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR)
* **Computer Vision & GUI:** OpenCV (`opencv-python`)
* **Dataset Management:** [Roboflow](https://roboflow.com/)
* **Numerical Operations:** NumPy

## 📂 Dataset & Model Weights
* **Dataset Source:** Curated and preprocessed via [Roboflow](https://roboflow.com/).
* **Trained Weights:** Fine-tuned detection weights located at `runs/detect/train-3/weights/best.pt`.

## 🚀 Quickstart & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Ryan5246/ANPR.git](https://github.com/Ryan5246/ANPR.git)
   cd ANPR
Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
   ```
   pip install ultralytics opencv-python paddleocr paddlepaddle numpy roboflow
Run the detection pipeline:
   ```
   python main.py

🎮 UI Controls
Play / Pause Trackbar: Toggle between 1 (Play) and 0 (Pause).


Timeline Scrubbing: Slide the Timeline bar to jump to any frame in the video.


Quit: Press d while active on the video window to terminate the program.


Created by Ryan Priyank
