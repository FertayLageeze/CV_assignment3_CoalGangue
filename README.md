Methods
The project employs a deep learning approach using a CNN for the segmentation task. The model architecture consists of convolutional layers, pooling layers, and upsampling layers to achieve pixel-wise classification. The training process involves optimizing a cross-entropy loss function using the Adam optimizer. The model's performance is evaluated using metrics such as training loss, accuracy, processing time, and the coal-to-gangue ratio.

Directory Structure
The project directory is structured as follows:

bash
Copy code
project-root/
├── raw_data/               # Directory containing raw input images
├── groundtruth/            # Directory containing ground truth segmentation images
├── results/                # Directory where results (comparative images and predictions) are saved
├── model.py                # Script defining the CNN model
├── train.py                # Script for training the model
├── predict.py              # Script for generating predictions and visualizations
├── utils.py                # Script containing utility functions for prediction and visualization
├── README.md               # This README file
Setup and Requirements
Python Environment: Ensure you have Python 3.x installed.
Dependencies: Install the required Python packages using pip:
bash
Copy code
pip install torch torchvision matplotlib numpy pillow
Usage
Training the Model
To train the model, run the train.py script. This script will load the training data, initialize the model, and train it for a specified number of epochs. The training loss and accuracy will be saved and plotted.

bash
Copy code
python train.py
Generating Predictions and Visualizations
After training the model, use the predict.py script to generate predictions on the test dataset and visualize the results. This script will create comparative images showing the raw input, predicted segmentation, and ground truth, as well as images with processing time and coal ratio annotations.

bash
Copy code
python predict.py
Scripts Overview
model.py: Defines the SimpleCNN class, which implements the CNN architecture used for segmentation.
train.py: Handles the training process, including data loading, model training, and loss/accuracy plotting.
predict.py: Generates predictions on the test dataset and visualizes the results with annotations.
utils.py: Contains utility functions for prediction and visualization, including the predict_and_visualize function.
