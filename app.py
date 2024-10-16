import cv2
import imutils
import numpy as np
import pyttsx3


def tts():
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate + 0)
    engine.say(label)
    engine.runAndWait()


# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = ([layer_names[i - 1] for i in net.getUnconnectedOutLayers()])
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# ***********************************************************************************
# Function to list available cameras
def available_cameras():
    available = []
    for i in range(10):  # Check for first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()  # Release the camera once it's confirmed available
    return available

# Check for available cameras
cameras = available_cameras()
if cameras:
    print(f"Available cameras: {cameras}")
    camera_index = cameras[0]  # Select the first available camera
else:
    print("No cameras available.")
    exit()
#____________________________________________________________________________________
# Loading image
cap = cv2.VideoCapture(0)

while True:

    i = 0
    ret, img = cap.read()
    img = imutils.resize(img, width=600)
    # cv2.imshow("resized output",img)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if (classes[class_ids[i]] == classes[class_ids[i]]):
                i += 1
                print("Number of objects:", i)
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                print(label)
                # tts()

    # show the output frame
    cv2.imshow("output video", img)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()
