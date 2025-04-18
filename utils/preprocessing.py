import cv2
import numpy as np
import time
from PIL import Image

class Preprocess:
    def gaussian_blur(image):
        """
        args:
            image (np.ndarray): The input image to be blurred.
        return:
            image (np.ndarray): The blurred image.  
        """
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    def bilateral_filter(image):
        """
        args:
            image (np.ndarray): The input image to be filtered.
        return:
            image (np.ndarray): The filtered image.
        """
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    def sharpen(image):
        """
        args:
            image (np.ndarray): The input image to be sharpened.
        return:
            image (np.ndarray): The sharpened image.
        """
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def histogram_equalization(image):
        """
        args:
            image (np.ndarray): The input image to be equalized.
        return:
            image (np.ndarray): The equalized image.
        """
        if len(image.shape) == 3:  # Check if the image is colored (3 channels)
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            channels = list(cv2.split(ycrcb))
            channels[0] = cv2.equalizeHist(channels[0])
            ycrcb = cv2.merge(channels)
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            return cv2.equalizeHist(image)
        
    def clahe(image):
        """
        args:
            image (np.ndarray): The input image to be equalized using CLAHE.
        return:
            image (np.ndarray): The equalized image.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged_lab = cv2.merge((cl, a, b))
        image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        return image
    
    def is_blur(image, threshold=200):
        """
        args:
            image (np.ndarray): The input image to be checked for blurriness.
            threshold (int): The threshold for blurriness.
        return:
            bool: True if the image is blurry, False otherwise.
        """
        variance = cv2.Laplacian(image, cv2.CV_64F).var()
        return variance < threshold
    
    def is_brightness(image, thresholdlow=90, thresholdhigh=220):
        """
        args:
            image (np.ndarray): The input image to be checked for brightness.
            thresholdlow (int): The lower threshold for brightness.
            thresholdhigh (int): The upper threshold for brightness.
        return:
            bool: True if the image is too bright or too dark, False otherwise.
        """
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]
        mean_brightness = np.mean(y_channel)
        return mean_brightness < thresholdlow or mean_brightness > thresholdhigh
    
    def is_contrast(image, thresholdlow=40, thresholdhigh=80):
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]
        std_contrast = np.std(y_channel)
        return std_contrast < thresholdlow or std_contrast > thresholdhigh
           
    def pre_process(image):
        """
        Args:
            image (np.ndarray): Input BGR image (from OpenCV).
        Returns:
            processed_image (np.ndarray): Processed image converted to PIL format for GUI display.
        """
        if Preprocess.is_blur(image):
            image = Preprocess.bilateral_filter(image)  
            
        image = Preprocess.sharpen(image)
        
        if Preprocess.is_brightness(image) or Preprocess.is_contrast(image):
            image = Preprocess.clahe(image)

        return image

