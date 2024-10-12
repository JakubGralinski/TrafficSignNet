Traffic Sign Classifier

This project implements a convolutional neural network (CNN) to classify traffic sign images using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The model is trained to recognize 43 different categories of traffic signs.

Features

	•	Data Loading and Preprocessing: Loads images from the GTSRB dataset and resizes them to a uniform size.
	•	Data Augmentation: Applies transformations like rotation, shifting, and zooming to enhance the dataset.
	•	Convolutional Neural Network: Implements a CNN with multiple convolutional and pooling layers.
	•	Training and Evaluation: Trains the model and evaluates its performance on a test set.
	•	Model Saving: Optionally saves the trained model to a file for future use.

Table of Contents

	•	Prerequisites
	•	Dataset
	•	Installation
	•	Usage
	•	Model Architecture
	•	Results
	•	License
	•	Acknowledgments

Prerequisites

	•	Python 3.x
	•	NumPy
	•	OpenCV (cv2)
	•	TensorFlow
	•	scikit-learn

Dataset

Download the GTSRB dataset from the official website or Kaggle. Ensure the dataset is organized into subdirectories for each category (0 to 42), with images inside each subdirectory.

Expected Directory Structure

gtsrb/
├── 0/
│   ├── img1.png
│   ├── img2.png
│   └── ...
├── 1/
│   ├── img1.png
│   ├── img2.png
│   └── ...
└── ...

Installation

	1.	Clone the Repository

git clone https://github.com/yourusername/traffic-sign-classifier.git
cd traffic-sign-classifier


	2.	Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


	3.	Install Dependencies

pip install -r requirements.txt

If a requirements.txt file is not provided, install the packages manually:

pip install numpy opencv-python tensorflow scikit-learn



Usage

Run the script with the path to the data directory. Optionally, specify a filename to save the trained model.

python traffic.py data_directory [model.h5]

	•	data_directory: Path to the GTSRB dataset directory.
	•	model.h5: (Optional) Filename to save the trained model.

Example

python traffic.py gtsrb traffic_model.h5

Model Architecture

The CNN model consists of:

	•	Convolutional Layers: Extract features using 32, 64, and 128 filters with 3x3 kernels.
	•	Batch Normalization: Normalize outputs from convolutional layers.
	•	Max-Pooling Layers: Reduce spatial dimensions using 2x2 pooling.
	•	Flatten Layer: Converts 2D feature maps to a 1D feature vector.
	•	Dense Layers: Fully connected layers with ReLU activation and dropout for regularization.
	•	Output Layer: Uses softmax activation to classify into 43 categories.

Layer Details

Layer (Type)                   Output Shape              Param #
=================================================================
Conv2D (32 filters, 3x3)       (None, 28, 28, 32)        896
BatchNormalization             (None, 28, 28, 32)        128
MaxPooling2D                   (None, 14, 14, 32)        0
Conv2D (64 filters, 3x3)       (None, 12, 12, 64)        18496
BatchNormalization             (None, 12, 12, 64)        256
MaxPooling2D                   (None, 6, 6, 64)          0
Conv2D (128 filters, 3x3)      (None, 4, 4, 128)         73856
BatchNormalization             (None, 4, 4, 128)         512
MaxPooling2D                   (None, 2, 2, 128)         0
Flatten                        (None, 512)               0
Dense (512 units)              (None, 512)               262656
Dropout (50%)                  (None, 512)               0
Output Dense (43 units)        (None, 43)                22059
=================================================================
Total params: 378,859

Results

After training for 30 epochs with data augmentation, the model achieves:

	•	Training Accuracy: Approximately 98%
	•	Validation Accuracy: Approximately 96%

Note: Actual results may vary based on the dataset and training conditions.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments

	•	German Traffic Sign Recognition Benchmark
	•	TensorFlow Documentation
