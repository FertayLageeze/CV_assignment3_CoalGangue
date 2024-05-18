# Automated Coal and Gangue Segmentation Using Convolutional Neural Networks

## Background
In the mining industry, separating coal from gangue is crucial for enhancing coal quality and reducing transportation costs. Traditional methods for this separation are often labor-intensive, time-consuming, and prone to errors. This project leverages deep learning, specifically convolutional neural networks (CNNs), to automate and improve the accuracy of coal and gangue segmentation in images. By training a CNN on a dataset of coal and gangue images, the model can learn to accurately identify and classify regions of coal and gangue, providing a more efficient and reliable solution.

## Methods
The project employs a deep learning approach using a CNN for the segmentation task. The model architecture consists of convolutional layers, pooling layers, and upsampling layers to achieve pixel-wise classification. The training process involves optimizing a cross-entropy loss function using the Adam optimizer. The model's performance is evaluated using metrics such as training loss, accuracy, processing time, and the coal-to-gangue ratio.

## Directory Structure
The project directory is structured as follows:

project-root/

├── raw_data/ # Directory containing raw input images

├── groundtruth/ # Directory containing ground truth segmentation images

├── results/ # Directory where results (comparative images and predictions) are saved

├── model.py # Script defining the CNN model

├── train.py # Script for training the model

├── predict.py # Script for generating predictions and visualizations

├── utils.py # Script containing utility functions for prediction and visualization

├── README.md # This README file



## Setup and Requirements
1. **Python Environment**: Ensure you have Python 3.x installed.
2. **Dependencies**: Install the required Python packages using pip:
    ```bash
    pip install torch torchvision matplotlib numpy pillow
    ```

## Usage
### Training the Model
To train the model, run the `train.py` script. This script will load the training data, initialize the model, and train it for a specified number of epochs. The training loss and accuracy will be saved and plotted.

  ```bash
  python train.py
  ```
### Generating Predictions and Visualizations
After training the model, use the predict.py script to generate predictions on the test dataset and visualize the results. This script will create comparative images showing the raw input, predicted segmentation, and ground truth, as well as images with processing time and coal ratio annotations.

```bash
python predict.py
```
## Usage
### Scripts Overview
model.py: Defines the SimpleCNN class, which implements the CNN architecture used for segmentation.

train.py: Handles the training process, including data loading, model training, and loss/accuracy plotting.

predict.py: Generates predictions on the test dataset and visualizes the results with annotations.

utils.py: Contains utility functions for prediction and visualization, including the predict_and_visualize function.


```vbnet
This format is ready for you to copy and paste into your GitHub repository's README.md file.
```

