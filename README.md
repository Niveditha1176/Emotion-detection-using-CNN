# Real-Time Facial Emotion Recognition using CNN

## Project Overview

This project implements a **Real-Time Facial Emotion Recognition
System** using a **Convolutional Neural Network (CNN)**. The system
detects human faces from a webcam feed and predicts emotions such as
*Happy, Sad, Fear, Angry, Neutral, Disgust,* and *Surprise*.

The project demonstrates concepts from **Design and Analysis of
Algorithms (DAA)** , **Signals processing** and **Machine Learning**, focusing on optimization,
computational efficiency, and deep learning-based image classification.

------------------------------------------------------------------------

##  Objective

-   To design an efficient algorithm capable of recognizing emotions
    from facial expressions.
-   To apply CNN-based feature extraction and classification techniques.
-   To perform real-time emotion detection using a webcam.
-   To analyze performance using accuracy and loss metrics.

------------------------------------------------------------------------

## Methodology

1.  **Data Preprocessing**
    -   Convert images to grayscale.
    -   Resize images to 48×48 pixels.
    -   Normalize pixel values.
2.  **Model Design**
    -   Convolutional Neural Network (CNN)
    -   Batch Normalization and Dropout layers
    -   Softmax classification layer.
3.  **Training**
    -   Optimizer: Adam
    -   Loss Function: Categorical Crossentropy
    -   Early Stopping & Learning Rate Scheduling used for optimization.
4.  **Real-Time Detection**
    -   Face detection using Haar Cascade.
    -   Emotion prediction using trained CNN model.

------------------------------------------------------------------------

## Technologies Used

-   Python
-   TensorFlow / Keras
-   OpenCV
-   NumPy
-   Matplotlib

------------------------------------------------------------------------

## Model Performance

-   Training Accuracy: \~83%
-   Validation Accuracy: \~66%
-   Optimized using adaptive learning rate and early stopping.

------------------------------------------------------------------------

## Applications

-   Human--Computer Interaction
-   Mental health monitoring
-   Smart classroom engagement systems
-   Driver fatigue detection
-   AI assistants and robotics

------------------------------------------------------------------------

## 📁 Project Structure

    EmotionDetection/
    │
    ├── training_notebook.ipynb
    ├── realtimedetection.py
    ├── final_emotion_model.keras
    ├── images/
    └── README.md

------------------------------------------------------------------------

## How to Run

1.  Install dependencies:

```{=html}
pip install --r requirements.py 
```
    pip install -tensorflow -opencv-python -numpy -matplotlib -sckit learn

2.  Run real-time detection:

```
    python realtimedetection.py
```
3.  Press **ESC** to exit webcam.

------------------------------------------------------------------------

## Algorithm Perspective 

The system applies: - Convolution operations for feature extraction. -
Iterative optimization using gradient descent. - Adaptive stopping
criteria for efficient convergence.

Time complexity primarily depends on convolution operations:

    O(L × n²)

where *L* = number of layers and *n* = image dimension.

------------------------------------------------------------------------

## Results

The proposed CNN model successfully classifies facial emotions in real
time with stable convergence and good generalization performance.

------------------------------------------------------------------------
## 📈 Graphs

Model Accuracy:

<img width="582" height="455" alt="image" src="https://github.com/user-attachments/assets/0d2b1e7b-023d-44dc-ba2c-f1f4382a2360" />

Model Loss Vs Epochs:

<img width="695" height="470" alt="image" src="https://github.com/user-attachments/assets/f668f6a6-9505-42f1-b3db-07fd3763a563" />

------------------------------------------------------------------------

## Author

**Niveditha Venkatesh**\

------------------------------------------------------------------------

## License

MIT Licence 
