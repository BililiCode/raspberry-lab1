# V6 - Working like V5 but, it saves the complete frame when an object is detected instead of saving the cropped image of the detected object.

import cv2
import pandas as pd
import numpy as np
import subprocess
import os
from datetime import datetime

# Load pre-trained model and configuration
model = './yolo/yolov3.weights'  # Update this path to your model's path
config = './yolo/yolov3.cfg'     # Update this path to your config file's path
net = cv2.dnn.readNetFromDarknet(config, model)

# Load class names
classes = []
with open("./yolo/coco.names", "r") as f:  # Update to the path of your coco.names file
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

# Function to calculate centroid of a bounding box
def calculate_centroid(x, y, w, h):
    return (int(x + w/2), int(y + h/2))

# Function to capture image using libcamera-still
def capture_image(image_path):
    subprocess.run(["libcamera-still", "-o", image_path, "--width", "1920", "--height", "1080", "-n", "-t", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# Initialize a DataFrame to store counts
data_frame = pd.DataFrame(columns=['Timestamp', 'Type', 'Count'])
object_counts = {cls: 0 for cls in classes}  # Object count per category
centroid_tracking = {}  # Tracks centroids of detected objects

# Ensure output directory exists
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Main loop
image_path = "temp.jpg"
object_id = 0  # Unique identifier for each object

try:
    while True:
        capture_image(image_path)
        frame = cv2.imread(image_path)
        if frame is None:
            continue

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
        centroids = []
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
                    centroids.append(calculate_centroid(x, y, w, h))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)  # Ensure the values are integers

            centroid = centroids[i]
            class_id = class_ids[i]

            # Check if this object was already detected
            if class_id not in centroid_tracking or not any(np.linalg.norm(np.array(centroid) - np.array(old_centroid)) < 50 for old_centroid in centroid_tracking[class_id]):
                object_counts[classes[class_id]] += 1
                centroid_tracking.setdefault(class_id, []).append(centroid)
                
                # Update data_frame using pandas.concat
                new_row = pd.DataFrame([{'Timestamp': datetime.now(), 'Type': classes[class_id], 'Count': 1}])
                data_frame = pd.concat([data_frame, new_row], ignore_index=True)

                # Save the entire frame when an object is detected
                frame_filename = os.path.join(output_dir, f"frame_{object_id}_{classes[class_id]}.jpg")
                cv2.imwrite(frame_filename, frame)
                object_id += 1

            draw_bounding_box(frame, class_id, confidences[i], round(x), round(y), round(x+w), round(y+h))

        # Display the frame
        cv2.imshow("Object Detection", frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    if os.path.exists(image_path):
        os.remove(image_path)

# Print and save the total count of objects detected per category
total_counts = pd.DataFrame(object_counts.items(), columns=['Type', 'Total Count'])
print(total_counts)
total_counts.to_csv("total_object_counts.csv", index=False)

# Save detailed counts to CSV
data_frame.to_csv("object_counts.csv", index=False)

# Write the total count to a text log
with open("total_counts_log.txt", "w") as log_file:
    log_file.write(str(total_counts))
