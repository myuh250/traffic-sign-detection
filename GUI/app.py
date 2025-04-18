import tkinter as tk
import cv2
import numpy as np
import time
from tkinter import filedialog
from PIL import Image, ImageTk
from threading import Thread
from ultralytics import YOLO
from utils.preprocessing import Preprocess
from utils.detection import Detection

def log_image_stats(image, tag=""):
    """
    Logs key image statistics for debugging and quality checks.
    
    Args:
        image (np.ndarray): BGR image (OpenCV format).
        tag (str): Optional tag to label the logging context.
    """
    # Convert to grayscale for Laplacian variance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
    # Convert to YCrCb for brightness and contrast check
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0]
    mean_brightness = np.mean(y_channel)
    std_contrast = np.std(y_channel)

    print(f"--- Image Statistics {tag} ---")
    print(f"Laplacian Variance (Sharpness): {laplacian_var:.2f}")
    print(f"Mean Brightness (Y): {mean_brightness:.2f}")
    print(f"Std Contrast (Y): {std_contrast:.2f}")
    print("-------------------------------")
    
class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Recognition")
        self.root.geometry("1200x700")

        self.root.grid_columnconfigure(0, weight=2)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Left frame for image/camera
        self.left_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.canvas = tk.Canvas(self.left_frame, bg="white")
        self.canvas.pack(expand=True, fill="both")

        self.button_frame = tk.Frame(self.left_frame)
        self.button_frame.pack(pady=10)

        self.upload_button = tk.Button(self.button_frame, text="Upload Image", command=self.load_image)
        self.upload_button.grid(row=0, column=0, padx=5)

        self.camera_button = tk.Button(self.button_frame, text="Use Camera", command=self.start_camera)
        self.camera_button.grid(row=0, column=1, padx=5)
        
        self.camera_button = tk.Button(self.button_frame, text="Stop Camera", command=self.stop_camera)
        self.camera_button.grid(row=0, column=2, padx=5)

        # Right frame for info
        self.right_frame = tk.Frame(self.root, bg="#e0e0e0")
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        self.info_label = tk.Label(self.right_frame, text="Traffic Sign Info:\n(placeholder)", font=("Arial", 14), justify="left")
        self.info_label.pack(padx=20, pady=20)

        self.cap = None
        self.running = False
        
        self.yolo_model = YOLO('models/yolov8m_tsd_best.pt')

    def load_image(self):
        """
        Load an image from the file system and display it on the canvas.
        Process the image and detect traffic signs.
        """
        self.stop_camera()
        self.root.after(100)
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            image = Image.open(file_path)
            image = image.resize((800, 600))

            # Convert PIL image to NumPy array for OpenCV
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Log stats before processing
            log_image_stats(image_np, tag="(Before Preprocessing)")

            # Preprocess the image
            processed_image = Preprocess.pre_process(image_np)
            
            # Log stats after processing
            log_image_stats(processed_image, tag="(After Preprocessing)")

            # Detect traffic signs
            processed_image_with_boxes = Detection.detect_traffic_sign(processed_image, self.yolo_model)

            # Convert back to PIL Image for Tkinter
            processed_image_with_boxes = Image.fromarray(cv2.cvtColor(processed_image_with_boxes, cv2.COLOR_BGR2RGB))
            self.img = ImageTk.PhotoImage(processed_image_with_boxes)
            self.canvas.create_image(0, 0, anchor="nw", image=self.img)

    def start_camera(self):
        """
        This function starts the camera and displays the video feed on the canvas.
        """
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            Thread(target=self.camera_loop, daemon=True).start()

    def camera_loop(self):
        """
        This function continuously captures frames from the camera and updates the canvas.
        Detect traffic signs in each frame.
        """
        last_log_time = 0  
        log_interval = 5

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()

                # --- Display raw frame only ---
                display_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                display_frame = cv2.resize(display_frame, (800, 600))
                img = Image.fromarray(display_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.root.after(0, self.update_canvas, imgtk)

                # --- Background processing for detection ---
                now = time.time()
                if now - last_log_time >= log_interval:
                    # Logging input stats
                    log_image_stats(self.current_frame, tag="(Camera Frame - Before Preprocessing)")
                    processed_image = Preprocess.pre_process(self.current_frame)
                    log_image_stats(processed_image, tag="(Camera Frame - After Preprocessing)")

                    # Run detection (in background)
                    _ = Detection.detect_traffic_sign(processed_image, self.yolo_model)

                    last_log_time = now
                else:
                    processed_image = Preprocess.pre_process(self.current_frame)
                    _ = Detection.detect_traffic_sign(processed_image, self.yolo_model)

                cv2.waitKey(15)
                
    def update_canvas(self, imgtk):
        """
        This function updates the canvas with the new frame from the camera.
        """
        self.canvas.delete("all")
        self.canvas.imgtk = imgtk  
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
        
    def get_current_frame(self):
        """
        This function returns the current frame from the camera.
        args:
            None
        return:
            current_frame: the current frame from the camera
        """
        return self.current_frame if hasattr(self, 'current_frame') else None
                
    def stop_camera(self):
        """
            This function stops the camera and releases the resources.
        """
        self.running = False
        if self.cap:
            self.root.after(100)
            self.cap.release()
            self.cap = None
            cv2.destroyAllWindows()
            self.canvas.delete("all")
