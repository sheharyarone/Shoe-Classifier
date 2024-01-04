# Shoe Classifier with ResNet and YOLOv7

## Overview

This project implements a shoe classifier that combines the power of YOLOv7 for object detection and ResNet for categorizing the identified objects. YOLOv7 is utilized to extract shoe objects within an image, and ResNet is employed to identify the specific category or class of each detected shoe. The frontend is built using Flask to provide a user-friendly interface for interacting with the classifier.

## How It Works

1. **Object Detection with YOLOv7:**
   - YOLO (You Only Look Once) is a real-time object detection system that divides an image into a grid and predicts bounding boxes and class probabilities for each grid cell.
   - YOLOv7, being the latest version, is employed to accurately locate and extract shoe objects in images.

2. **Shoe Classification with ResNet:**
   - ResNet (Residual Network) is a deep learning architecture known for its ability to train very deep neural networks effectively.
   - The extracted shoe objects from YOLOv7 are then fed into a ResNet model, which classifies each shoe into predefined categories.

3. **Flask Frontend:**
   - The frontend of the application is built using Flask, a web framework for Python.
   - Flask provides a user-friendly interface to interact with the shoe classifier, allowing users to upload images and receive classification results.

4. **Integration:**
   - The output from YOLOv7, containing bounding box coordinates and class probabilities for each shoe, is seamlessly integrated with the ResNet classifier.
   - The final result is a comprehensive shoe classification system with a Flask frontend that simplifies user interaction.

## Dependencies

- Python 3.x
- PyTorch
- YOLOv7
- ResNet implementation (such as torchvision.models)
- Flask

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/sheharyarone/Shoe-Classifier.git
    cd Shoe-Classifier
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Pre-trained Weights

Download the pre-trained weights for YOLOv7 and ResNet from the following Google Drive link:

[Pre-trained Weights](https://drive.google.com/drive/u/0/folders/1bFDgVzprK9BgCfGmcTzbHYBdDC_bBXxc)

Make sure to place the downloaded weights in the appropriate directories before running the classification script.

## Usage

1. Download the pre-trained weights and place them in Shoe-Classifier directory
2. Run the Flask application:

    ```bash
    python app.py
    ```

3. Open your browser and navigate to `http://localhost:5000` to use the shoe classifier through the Flask frontend.

## Contributing

Feel free to contribute to the development of this project. Open issues, submit pull requests, or suggest improvements.

## License

This project is licensed under the [MIT License](LICENSE).
