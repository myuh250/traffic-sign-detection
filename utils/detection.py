import cv2
import numpy as np

class Detection:
    def detect_traffic_sign(frame, model):
        """
        Detect traffic signs using YOLO and draw bounding boxes on the frame.
        Args:
        - frame: The processed image/frame from the camera or file.
        - model: The YOLO model for detection.
        Returns:
        - frame_with_boxes: The input frame with bounding boxes drawn.
        """
        try:
            results = model(frame)[0]

            # Loop through the results and draw bounding boxes
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                cls_id = int(box.cls[0])  
                confidence = float(box.conf[0])  
                label = Detection.get_class_name(cls_id) 

                # Draw the bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', 
                            (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 255, 0), 2)
            return frame  
        
        except Exception as e:
            print("Error in detection:", e)
            return frame

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

    class_names = load_class_names()

    def get_class_name(classNo):
        """
        Return the class name based on the class ID.
        """
        if 0 <= classNo < len(Detection.class_names):
            return Detection.class_names[classNo]
        else:
            return "Unknown"
