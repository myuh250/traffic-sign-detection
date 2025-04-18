import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from threading import Thread

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

    def load_image(self):
        """
        This function allows the user to upload an image file and display it on the canvas.
        """
        self.stop_camera()
        self.root.after(100)  # đợi 100ms để đảm bảo camera được release
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            image = Image.open(file_path)
            image = image.resize((800, 600))
            self.img = ImageTk.PhotoImage(image)
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
        """
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (800, 600))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.root.after(0, self.update_canvas, imgtk)
                
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
