# DJI_Ryze_tello_face-Recognition-Human_Body_detection
to be able to run this code you need to use YOLO(You only look once),library to be able to do that you need to go to following link on github and the download YOLO-cfg file and YOLO-weights file, please dont worry if in upcoming future this github link does not work, then dont worry just look for some other links you will find it.

**Link for CFG and Weights file - https://github.com/WongKinYiu/PyTorch_YOLOv4

Your program structure should be as follows :

DroneProject/
├── human.py         # The Python script
├── models/
│   ├── yolov4.cfg   # YOLO configuration
│   ├── yolov4.weights # YOLO weights
│   └── coco.names   # COCO labels file
