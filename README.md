🚦 Traffic Sign Classifier with Convolutional Neural Networks

Classify German traffic signs using a deep learning model built with TensorFlow and Keras.

📋 Table of Contents

	•	About the Project
	•	Features
	•	Dataset
	•	Getting Started
	•	Prerequisites
	•	Installation
	•	Usage
	•	Model Architecture
	•	Results
	•	Contributing
	•	License
	•	Acknowledgments

📝 About the Project

This project implements a Convolutional Neural Network (CNN) to classify traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB). The model is trained to recognize 43 different categories of traffic signs with high accuracy.

✨ Features

	•	Data Loading and Preprocessing: Efficiently loads and preprocesses images.
	•	Data Augmentation: Enhances dataset diversity with transformations like rotation, shifting, and zooming.
	•	Custom CNN Model: Builds a deep neural network using TensorFlow and Keras.
	•	Training and Evaluation: Provides tools for training the model and evaluating performance.
	•	Model Saving: Saves the trained model for future use.

📚 Dataset

The GTSRB dataset contains over 50,000 images of 43 different traffic sign classes. Each class is represented by a unique identifier (from 0 to 42).

Dataset Structure

gtsrb/
├── 0/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── 1/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
└── ...

Note: Ensure that the dataset is organized into subdirectories for each category, with images inside each subdirectory.

🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

✅ Prerequisites

	•	Python 3.x
	•	Packages:
	•	NumPy
	•	OpenCV (cv2)
	•	TensorFlow
	•	scikit-learn

📥 Installation

	1.	Clone the Repository

git clone https://github.com/yourusername/traffic-sign-classifier.git
cd traffic-sign-classifier


	2.	Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


	3.	Install Dependencies

pip install -r requirements.txt

If requirements.txt is not available, manually install the packages:

pip install numpy opencv-python tensorflow scikit-learn



🧑‍💻 Usage

Run the script with the path to the data directory. Optionally, specify a filename to save the trained model.

python traffic.py data_directory [model.h5]

	•	data_directory: Path to the dataset directory.
	•	model.h5: (Optional) Filename to save the trained model.

Example

python traffic.py gtsrb traffic_model.h5

Command-Line Arguments

	•	Data Directory: The directory containing the traffic sign images organized into subdirectories by class label.
	•	Model Filename: (Optional) The filename to save the trained model.

🏗️ Model Architecture

The CNN model is designed to effectively capture the features of traffic sign images.

Layers Overview

	1.	Convolutional Layer: 32 filters, 3x3 kernel, ReLU activation.
	2.	Batch Normalization
	3.	Max-Pooling: 2x2 pool size.
	4.	Convolutional Layer: 64 filters, 3x3 kernel, ReLU activation.
	5.	Batch Normalization
	6.	Max-Pooling: 2x2 pool size.
	7.	Convolutional Layer: 128 filters, 3x3 kernel, ReLU activation.
	8.	Batch Normalization
	9.	Max-Pooling: 2x2 pool size.
	10.	Flatten Layer
	11.	Dense Layer: 512 units, ReLU activation.
	12.	Dropout: 50%
	13.	Output Layer: Softmax activation for 43 categories.

Model Summary

_________________________________________________________________
Layer (Type)                 Output Shape              Param #
=================================================================
Conv2D (32 filters, 3x3)     (None, 28, 28, 32)        896
BatchNormalization           (None, 28, 28, 32)        128
MaxPooling2D                 (None, 14, 14, 32)        0
Conv2D (64 filters, 3x3)     (None, 12, 12, 64)        18496
BatchNormalization           (None, 12, 12, 64)        256
MaxPooling2D                 (None, 6, 6, 64)          0
Conv2D (128 filters, 3x3)    (None, 4, 4, 128)         73856
BatchNormalization           (None, 4, 4, 128)         512
MaxPooling2D                 (None, 2, 2, 128)         0
Flatten                      (None, 512)               0
Dense (512 units)            (None, 512)               262656
Dropout (50%)                (None, 512)               0
Output Dense (43 units)      (None, 43)                22059
=================================================================
Total params: 378,859
Trainable params: 378,283
Non-trainable params: 576
_________________________________________________________________

📈 Results

After training for 30 epochs with data augmentation, the model achieves:

	•	Training Accuracy: ~98%
	•	Validation Accuracy: ~96%

Note: Results may vary based on the dataset and training conditions.

Training Graphs

🤝 Contributing

Contributions are welcome! Please follow these steps:

	1.	Fork the Project
	2.	Create your Feature Branch (git checkout -b feature/AmazingFeature)
	3.	Commit your Changes (git commit -m 'Add some AmazingFeature')
	4.	Push to the Branch (git push origin feature/AmazingFeature)
	5.	Open a Pull Request

📄 License

Distributed under the MIT License. See LICENSE for more information.

🙏 Acknowledgments

	•	German Traffic Sign Recognition Benchmark
	•	TensorFlow
	•	Keras Documentation
	•	OpenCV
	•	scikit-learn
