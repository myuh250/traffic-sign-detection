import cv2
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

class TrafficSignCNN(nn.Module):
    def __init__(self):
        super(TrafficSignCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 52)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(-1, 256 * 8 * 8) 
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Detection:
    last_log_time = 0
    log_interval = 5

    # Load class names
    @staticmethod
    def load_class_names(file_path="data/classes_en.txt"):
        """
        Loads class names from a text file where each line corresponds to a class.
        
        Args:
        - file_path (str): Path to the class names file.

        Returns:
        - class_names (list): List of class names.
        """
        try:
            with open(file_path, 'r') as file:
                class_names = [line.strip() for line in file.readlines()]
            return class_names
        except Exception as e:
            print(f"Error reading class names from {file_path}: {e}")
            return []

    class_names = load_class_names.__func__()
    num_classes = len(class_names)

    @staticmethod
    def get_class_name(classNo):
        """
        Return the class name based on the class ID.
        """
        if 0 <= classNo < len(Detection.class_names):
            return Detection.class_names[classNo]
        else:
            return "Unknown"
        
    # Load CNN model
    cnn_model_path = "models/cnn_traffic_sign_model.pth"
    cnn_model = TrafficSignCNN()
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=torch.device('cpu')))
    cnn_model.eval()

    cnn_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    @staticmethod
    def detect_signs(frame, yolo_model):
        """
        Detect traffic signs using YOLO and CNN for classification.
        Returns detected objects info without modifying the frame.
        
        Args:
            frame (np.ndarray): Input image frame
            yolo_model: YOLO model for initial detection
            
        Returns:
            list: List of dictionaries containing detection data (box coordinates, label, confidence)
        """
        try:
            now = time.time()
            should_log = now - Detection.last_log_time >= Detection.log_interval
            results = yolo_model(frame, verbose=False)[0]
            detections = []

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = frame[y1:y2, x1:x2]

                if cropped.shape[0] < 10 or cropped.shape[1] < 10:
                    continue

                img_tensor = Detection.cnn_transform(cropped).unsqueeze(0)

                with torch.no_grad():
                    output = Detection.cnn_model(img_tensor)
                    probs = F.softmax(output, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                    label = Detection.get_class_name(predicted.item())
                    conf = confidence.item()

                detections.append({
                    'box': (x1, y1, x2, y2),
                    'label': label,
                    'confidence': conf
                })

                if should_log:
                    print(f"[CNN-on-YOLO] Detected: {label} ({conf:.2f}) at ({x1}, {y1})")

            if should_log:
                Detection.last_log_time = now

            return detections

        except Exception as e:
            print("Error in detect_signs:", e)
            return []
    
    @staticmethod
    def draw_detections(frame, detections):
        """
        Draw bounding boxes and labels on the frame based on detection results.
        
        Args:
            frame (np.ndarray): Input image frame
            detections (list): List of detection dictionaries from detect_signs
            
        Returns:
            np.ndarray: Frame with drawn bounding boxes and labels
        """
        try:
            result_frame = frame.copy()
            
            for detection in detections:
                x1, y1, x2, y2 = detection['box']
                label = detection['label']
                conf = detection['confidence']
                
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(result_frame, f'{label} ({conf:.2f})',
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                
            return result_frame
            
        except Exception as e:
            print("Error in draw_detections:", e)
            return frame
        
#  ====================================================================================================================
# # TO TEST YOLOv8 ONLY  
# class Detection:
#     last_log_time = 0
#     log_interval = 5  

#     @staticmethod
#     def detect_signs(frame, model):
#         """
#         Detect traffic signs using only YOLO.
#         Returns detected objects info without modifying the frame.
        
#         Args:
#             frame (np.ndarray): Input image frame
#             model: YOLO model for detection
            
#         Returns:
#             list: List of dictionaries containing detection data (box coordinates, label, confidence)
#         """
#         try:
#             now = time.time()
#             should_log = now - Detection.last_log_time >= Detection.log_interval
#             results = model(frame, verbose=False)[0]
#             detections = []

#             for box in results.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cls_id = int(box.cls[0])
#                 confidence = float(box.conf[0])
#                 label = Detection.get_class_name(cls_id)

#                 detections.append({
#                     'box': (x1, y1, x2, y2),
#                     'label': label,
#                     'confidence': confidence
#                 })

#                 if should_log:
#                     print(f"[YOLO] Detected: {label} ({confidence:.2f}) at ({x1}, {y1}), ({x2}, {y2})")

#             if should_log:
#                 Detection.last_log_time = now

#             return detections

#         except Exception as e:
#             print("Error in detect_signs:", e)
#             return []
    
#     @staticmethod
#     def draw_detections(frame, detections):
#         """
#         Draw bounding boxes and labels on the frame based on detection results.
        
#         Args:
#             frame (np.ndarray): Input image frame
#             detections (list): List of detection dictionaries from detect_signs
            
#         Returns:
#             np.ndarray: Frame with drawn bounding boxes and labels
#         """
#         try:
#             result_frame = frame.copy()
            
#             for detection in detections:
#                 x1, y1, x2, y2 = detection['box']
#                 label = detection['label']
#                 conf = detection['confidence']
                
#                 cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(result_frame, f'{label} {conf:.2f}', 
#                             (x1, y1 - 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 
#                             0.8, (0, 255, 0), 2)
                
#             return result_frame
            
#         except Exception as e:
#             print("Error in draw_detections:", e)
#             return frame

#     @staticmethod
#     def load_class_names(file_path="data/classes_en.txt"):
#         """
#         Loads class names from a text file where each line corresponds to a class.
        
#         Args:
#         - file_path (str): Path to the class names file.

#         Returns:
#         - class_names (list): List of class names.
#         """
#         try:
#             with open(file_path, 'r') as file:
#                 class_names = [line.strip() for line in file.readlines()]
#             return class_names
        
#         except Exception as e:
#             print(f"Error reading class names from {file_path}: {e}")
#             return []

#     class_names = load_class_names.__func__()

#     @staticmethod
#     def get_class_name(classNo):
#         """
#         Return the class name based on the class ID.
#         """
#         if 0 <= classNo < len(Detection.class_names):
#             return Detection.class_names[classNo]
#         else:
#             return "Unknown"