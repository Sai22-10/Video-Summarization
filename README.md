# Video Summarization Project

This project implements a video summarization application using OpenCV and YOLO (You Only Look Once) for object detection. The application processes video input, detects objects in real-time, and generates a summarized output video based on the detected objects.

## Project Structure

```
video-summarization
├── src
│   ├── main.py               # Entry point for the video summarization application
│   ├── yolo
│   │   ├── yolov3.cfg        # Configuration file for YOLOv3 model
│   │   ├── yolov3.weights     # Pre-trained weights for YOLOv3 model
│   │   └── coco.names        # Class names for COCO dataset
│   └── utils
│       └── video_utils.py    # Utility functions for video processing
├── requirements.txt          # List of dependencies
└── README.md                 # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd video-summarization
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`. After activating your environment, run:
   ```
   pip install -r requirements.txt
   ```

3. **Download YOLOv3 weights and configuration**:
   Ensure that the `yolov3.weights`, `yolov3.cfg`, and `coco.names` files are placed in the `src/yolo` directory.

## Usage

To run the video summarization application, execute the following command in your terminal:

```
python src/main.py --input <path_to_input_video> --output <path_to_output_video>
```

Replace `<path_to_input_video>` with the path to the video file you want to summarize and `<path_to_output_video>` with the desired output file path.

## Overview

The application utilizes the YOLOv3 model for real-time object detection. It processes each frame of the input video, detects objects, and summarizes the video based on the presence of significant objects. The output is a new video file that highlights the detected objects, providing a concise summary of the original footage.

## License

This project is licensed under the MIT License - see the LICENSE file for details.