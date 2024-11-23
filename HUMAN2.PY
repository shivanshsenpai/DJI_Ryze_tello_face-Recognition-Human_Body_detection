import cv2
import numpy as np
import face_recognition as fr
from djitellopy import Tello

# Load reference image for face recognition
reference_image_path = r"W:\DroneProject\FaceRecog\uploads\Shivi.jpg"
reference_image = fr.load_image_file(reference_image_path)
reference_face_encoding = fr.face_encodings(reference_image)[0]

# YOLO model paths
yolo_cfg = "models/yolov4.cfg"
yolo_weights = "models/yolov4.weights"
coco_names = "models/coco.names"

# Load YOLO model
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open(coco_names, "r") as f:
    classes = f.read().strip().split("\n")

human_index = classes.index("person")


# Function for YOLO-based human detection
def detect_humans(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences = [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == human_index:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    humans = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            confidence = confidences[i] * 100
            humans.append((x, y, w, h, confidence))
            # Draw bounding box and confidence
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box
            cv2.putText(frame, f"Person {confidence:.1f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame, humans


# Function for face detection and identification
def detect_faces(frame, humans):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = fr.face_locations(rgb_frame)

    for face_loc in face_locations:
        top, right, bottom, left = face_loc
        face_encoding = fr.face_encodings(rgb_frame, [face_loc])[0]
        match = fr.compare_faces([reference_face_encoding], face_encoding, tolerance=0.6)

        # Use green for matched faces, red otherwise
        color = (0, 255, 0) if match[0] else (0, 0, 255)
        label = "Match" if match[0] else "Unknown"
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


# Function to draw background and title
def draw_styled_overlay(frame):
    overlay = frame.copy()
    alpha = 0.5  # Transparency factor
    # Create a semi-transparent overlay
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
    # Blend overlay with the original frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # Add a title
    cv2.putText(frame, "Human & Face Detection", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


# Main function
def main():
    drone = Tello()
    drone.connect()
    drone.streamon()

    try:
        while True:
            frame = drone.get_frame_read().frame
            frame = cv2.resize(frame, (960, 720))  # Resize for consistency

            # Add styled overlay
            frame_with_overlay = draw_styled_overlay(frame)

            # Detect humans in the frame
            frame_with_humans, humans = detect_humans(frame_with_overlay)

            # Detect and identify faces in the frame
            frame_with_faces = detect_faces(frame_with_humans, humans)

            # Show the final output
            cv2.imshow("Detection Output", frame_with_faces)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Cleanup
        drone.streamoff()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
