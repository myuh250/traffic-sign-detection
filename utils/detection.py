import cv2
import numpy as np
import time

class Detection:
    last_log_time = 0
    log_interval = 5  

    @staticmethod
    def detect_traffic_sign(frame, model):
        """
        Detect traffic signs using YOLO and draw bounding boxes on the frame.
        Logs only every 5 seconds.
        """
        try:
            now = time.time()
            should_log = now - Detection.last_log_time >= Detection.log_interval

            results = model(frame, verbose=False)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = Detection.get_class_name(cls_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', 
                            (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 255, 0), 2)

                if should_log:
                    print(f"[YOLO] Detected: {label} ({confidence:.2f}) at ({x1}, {y1}), ({x2}, {y2})")

            if should_log:
                Detection.last_log_time = now

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
