# Traffic Sign Detection

This project is a Python-based application that performs traffic sign detection and classification. It uses machine learning techniques to recognize and classify traffic signs from images or video streams. The application is built using Tkinter for the user interface and employs image processing and pattern recognition methods to detect traffic signs.

## Features

- **Image Upload**: Allows users to upload images or capture real-time video from a webcam for traffic sign detection and classification.
- **Traffic Sign Detection**: Uses the YOLOv8-seg model for detecting and segmenting traffic signs in images.
- **Traffic Sign Classification**: Utilizes Convolutional Neural Networks (CNNs) to classify detected traffic signs into various categories (e.g., stop signs, speed limit signs, etc.).
- **Description Generation**: Generates textual descriptions of detected signs using a Local Language Model (LLM), providing contextual information for each traffic sign.
- **User Interface**: The app provides an easy-to-use GUI built with Tkinter for user interaction.

## Technologies Used

- **Python**: The primary programming language.
- **Tkinter**: For building the graphical user interface (GUI).
- **YOLOv8-seg**: For real-time traffic sign detection and semantic segmentation.
- **Convolutional Neural Networks (CNNs)**: For classifying detected traffic signs into predefined categories.
- **OpenCV**: For image and video processing tasks (e.g., resizing, converting color, etc.).
- **TensorFlow/PyTorch**: For implementing machine learning models, including CNN and YOLOv8-seg.
- **Local Language Model (LLM)**: For generating textual descriptions of traffic signs.

## Requirements

- The set of packages required to run the application is listed in `requirements.txt`. You can install them using pip.
- We are currently using Python 3.12.6

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/myuh250/traffic-sign-detection.git
   cd traffic-sign-detection
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```

3. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application
    ```bash
    python main.py
    ```

## Results

The project has successfully implemented a traffic sign detection and classification system using state-of-the-art deep learning models. The YOLOv8-seg model demonstrated high accuracy in detecting and segmenting traffic signs from both images and video streams. 

## Future Directions

- Expand the dataset to include more diverse and rare traffic sign types.
- Further improve detection and classification accuracy, especially in challenging conditions (e.g., occlusion, low light).
- Integrate additional languages for description generation.
- Optimize the application for deployment on embedded systems or mobile devices.
- Explore advanced model architectures and transfer learning techniques.

## Summary

This project demonstrates a comprehensive approach to traffic sign detection and classification, combining modern computer vision techniques with a practical user interface. The current implementation provides a strong foundation for further research and real-world deployment.
