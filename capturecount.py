
# Working Object Tracker V1.2, adapted from https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
# This script will detect objects in a video stream from the camera and save the object counts to a CSV file (object_counts.csv)
# The script will run until interrupted with Ctrl+C
# The script will save the counts to a CSV file when interrupted
# The script will also save the counts to a CSV file when the camera is released (when the script is stopped)
# The code is working fine headless on the Raspberry Pi 5.

import cv2
import pandas as pd
from datetime import datetime
import numpy as np

# Initialize camera
# Adding a flag for headless mode (set to ture when running in headless mode over SSH)
headless_mode= True
camera = cv2.VideoCapture(0)  # Adjust '0' if you have more than one camera
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# Load pre-trained model and configuration
model = './yolo/yolov3.weights' # Change this to your model's path
config = './yolo/yolov3.cfg'    # Change this to your config file's path
net = cv2.dnn.readNetFromDarknet(config, model)

# Load class names
classes = []
with open("./yolo/coco.names", "r") as f: # Change to the path of your coco.names file
    classes = [line.strip() for line in f.readlines()]

# Function to get output layers
def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]


# Function to draw bounding box
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = np.random.uniform(0, 255, size=(3,))
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Initialize a DataFrame to store counts
data_frame = pd.DataFrame(columns=['Timestamp', 'Type', 'Count'])

# Main loop
try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        Width = frame.shape[1]
        Height = frame.shape[0]
        scale = 0.00392

        # Create a blob and pass it through the model
        blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        # Process the outputs
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            # No need for i = i[0] as indices is already a flat array
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_bounding_box(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
            #data_frame = data_frame.append({'Timestamp': datetime.now(), 'Type': classes[class_ids[i]], 'Count': 1}, ignore_index=True)
            data_frame = pd.concat([data_frame, pd.DataFrame([{'Timestamp': datetime.now(), 'Type': classes[class_ids[i]], 'Count': 1}])], ignore_index=True)
        # Display the frame
        #cv2.imshow("Object Detection", frame)

        # Break loop with 'q' key
        #if cv2.waitKey(1) & 0xFF == ord('q'):
         #break

except KeyboardInterrupt:
    print("Script Interrupted, saving data and shutting down.")

finally:
    data_frame.to_csv('object_counts.csv' , index=False)
    camera.release()

# Save counts to CSV
data_frame.to_csv('object_counts.csv', index=False)
