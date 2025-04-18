# The model in trained on Kaggle
# If you want to train the model on your own device, you can use the code below.
# Note that it is required to have data.yaml file in the data folder.

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8 model
model.train(
    data='data/data.yaml',  
    epochs=50,  
    imgsz=640,  
    batch=16,  
    device=0,  
    save_period=5,  
    project='models',
    name='traffic_sign_model',  
)
