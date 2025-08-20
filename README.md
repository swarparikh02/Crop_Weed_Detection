# CropWeedDetector: YOLOv3-Based Crop and Weed Detection

## Overview
This project implements a custom object detection model using YOLOv3 (via Darknet) for identifying crops and weeds in agricultural images. Developed in Google Colab with GPU acceleration, it clones the Darknet repository, prepares a dataset from Google Drive, trains the model on labeled images, and tests it for real-time detection. The model distinguishes between "crop" and "weed" classes, aiding precision agriculture by enabling automated weed identification.

Key aspects from the notebook:
- Dataset preparation: Assumes annotated images in YOLO format (e.g., `/mydrive/Agriculture/train/` with .jpg and .txt labels).
- Training: Uses pretrained Darknet53 weights for transfer learning.
- Testing: Threshold-based detection (e.g., 0.3) on sample images.
- Visualization: Displays predictions with bounding boxes.

This project demonstrates computer vision skills for real-world applications like smart farming.

## Features
- **Dataset Handling**: Generates train/test splits from image paths in Google Drive.
- **Model Training**: Custom YOLOv3 configuration for 2 classes (crop, weed); trains with batch=32, subdivisions=16.
- **GPU Acceleration**: Enables CUDA and OpenCV in Darknet Makefile for faster training.
- **Testing**: Runs inference on test images with adjustable confidence threshold; visualizes results using Matplotlib.
- **Helper Functions**: Includes image display (`imShow`), file upload/download for Colab integration.
- Supports prevention of runtime disconnection via JavaScript snippet.

## Architecture
- **Framework**: Darknet (YOLOv3 implementation in C, wrapped in Python via Colab).
- **Data Flow**: Mount Google Drive → Prepare obj.data, obj.names, train/test.txt → Train on custom .cfg → Test with weights.
- **Classes**: 2 (crop, weed); filters adjusted to (2+5)*3=21 per layer.
- **Environment**: Google Colab with GPU (Tesla K80 or similar), Python 3.

## Setup Instructions
1. **Prerequisites**: Google Colab account, GPU runtime enabled. Dataset in Google Drive (e.g., `/mydrive/Agriculture/` with train/test folders containing .jpg images and .txt labels).
2. **Run in Colab**:
   - Open `crop_weed_detection.ipynb` in Colab.
   - Mount Google Drive and ensure dataset paths match (e.g., update `/mydrive/Agriculture/` if needed).
   - Execute cells sequentially to clone Darknet, build with GPU/OpenCV, prepare data, and train.
3. **Dataset Preparation**:
   - Place images and labels in `/mydrive/Agriculture/train/` and `/mydrive/Agriculture/test/`.
   - Run cells to generate `train.txt` and `test.txt`.
4. **Training**:
   - Download pretrained weights: `!wget http://pjreddie.com/media/files/darknet53.conv.74`.
   - Train: `!./darknet detector train data/obj.data cfg/crop_weed.cfg darknet53.conv.74 -dont_show`.
5. **Prevent Disconnect**: Paste the provided JavaScript into Colab console for long training sessions.

## Usage
- **Training**: Run the training cell; monitor progress and chart.png for loss curves.
- **Testing**: Switch to test mode in .cfg (batch=1, subdivisions=1), then: `!./darknet detector test data/obj.data cfg/crop_weed.cfg /mydrive/Agriculture/backup/yolov3_custom_final.weights /mydrive/Agriculture/test/weed_1.jpeg -thresh 0.3`.
- **Visualization**: Use `imShow('predictions.jpg')` to display detected bounding boxes.
- Example Output: Bounding boxes with labels (crop/weed) and confidence scores.

## Technologies
- **Framework**: Darknet (YOLOv3)
- **Language**: Python 3 (Colab), C (Darknet backend)
- **Libraries**: OpenCV (cv2 for image processing), Matplotlib (for visualization), Google Colab utilities (drive mount, file upload/download)
- **Hardware**: GPU (CUDA-enabled for training acceleration)
- **Other**: Bash commands for repo cloning, Makefile edits, and model execution

## Limitations
- Assumes YOLO-format annotations (not included; prepare your own dataset).
- Colab-specific (GPU runtime required; may timeout on long trains without JS snippet).
- Tested on agricultural images; fine-tune for other domains.

## License
MIT License. For educational purposes; dataset not included due to size/privacy.
