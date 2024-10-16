
# Object Detection System

This object detection application utilizes the YOLO (You Only Look Once) algorithm to identify and classify objects in real-time from a webcam feed. By leveraging OpenCV and a text-to-speech engine, it detects objects, draws bounding boxes around them, and announces their labels. The app first checks for available cameras, processes video frames, and applies non-maximum suppression to enhance detection accuracy, providing an interactive way to recognize objects visually and audibly.
## Run Locally

Clone the project

```bash
  git clone https://github.com/chamoddissanayake/Object-Detection-System.git
```

Install dependencies

```bash
  pip install -r requirements.txt
```

If your device has more than single camera provide the relevant camera id in here
```bash
cap = cv2.VideoCapture(0)
```

Start the Application

```bash
  python app.py
```


## Tech Stack

**Programming Language:** Python

**Libraries/Frameworks:**

* OpenCV: For image processing and computer vision tasks.
* imutils: For convenient image processing functions.
* NumPy: For numerical operations and handling arrays.
* Pyttsx3: For text-to-speech conversion.

**Machine Learning Model:** 
* YOLO (You Only Look Once) for real-time object detection.

**Model Files:**

* yolov3.weights: Pre-trained weights for the YOLO model.
* yolov3.cfg: Configuration file for the YOLO model.
* coco.names: Class names for the objects detected by YOLO.