# ASL Alphabet Recognition
This project focuses on recognizing American Sign Language (ASL) alphabets using Convolutional Neural Network(CNNs) and ResNet-50. The goal is to train a model that can accurately classify different alphabets from 'A' to 'Z' in ASL.

Link to dataset - https://www.kaggle.com/datasets/ayuraj/american-sign-language-dataset

## Files Included
- `asl`: Directory containing the ASL alphabet images categorized by their respective alphabets.
- `Resnet.ipynb`: Jupyter Notebook for ResNet model training and testing
- `CNN.ipynb`: Jupyter Notebook for CNN model training and testing
- `image1.jpeg`: Sample image for prediction testing.
- `cnn_model_history.json`: saved history file for CNN model
- `resnet_model_history.json`: Saved history file for ResNet50


## Setup and Installation
1. Install the required libraries using `pip install -r requirements.txt`.
2. Ensure the ASL dataset is structured properly in the `asl` directory.

## Preprocessing
### CNN:

    •    Resize the images to a uniform size using cv2.resize or appropriate functions.
    •   Convert the images to grayscale using cv2.cvtColor.
    •   Normalize the pixel values to a range between 0 and 1 by dividing by 255.0 for better convergence during training.
    •    Assign numerical labels to each class (ASL alphabet).
    •    Convert the categorical labels into one-hot encoded vectors using to_categorical from Keras.
    •    Divide the dataset into training and testing sets using train_test_split from sklearn.model_selection.
### ResNet50:
    •    Resize the images to a consistent size required by ResNet50
    •    ResNet50 expects images in RGB format hence, they are not converted into grayscale
    •    Perform label encoding and convert categorical labels into one-hot encoded vectors 
    •    Split the dataset into training and testing sets using train_test_split as done for CNN.
## Model Training
- The CNN model is trained using grayscale ASL images and augmented using Keras' `ImageDataGenerator`.
- The ResNet50-based model is trained on augmented RGB ASL images.

### Usage
- Use `predict_label()` function  to predict the label of an uploaded ASL image.
- Test the models using `image1.jpeg` or upload your ASL alphabet images for prediction.
## Results
- The trained CNN achieved an accuracy of approximately 99% on the test set.
- The ResNet50-based model achieved an accuracy of approximately 89% on the test set.
- Sample predictions can be tested using the provided image file.

