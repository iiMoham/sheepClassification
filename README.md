# Sheep Classification Challenge

This repository contains a solution for a sheep classification challenge. The goal of this project is to build a deep learning model that accurately classifies images of sheep into different categories. This solution uses a powerful pre-trained model and various techniques like data augmentation, cross-validation, and ensembling to achieve a high F1 score.

## Dataset

The data for this challenge is provided by a Kaggle competition. You can find more details and download the data from the following link:

[Sheep Classification Challenge 2025 Data](https://www.kaggle.com/competitions/sheep-classification-challenge-2025/data)

The dataset is composed of a training set of labeled images and a test set of unlabeled images.

## Getting Started

### Prerequisites

To run this project, you need to have Python 3 and the following libraries installed:

```bash
pip install torch torchvision timm scikit-learn pandas numpy albumentations>=1.1.0 matplotlib
```

### Data Setup

After downloading the data from Kaggle, you should organize it as follows:

```
/content/drive/MyDrive/data/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── test/
│   ├── image_test1.jpg
│   ├── image_test2.jpg
│   └── ...
└── train_labels.csv
```

### Running the Notebook

The main code is in the `sheepClassification (1).ipynb` notebook. You can run it cell by cell in a Jupyter environment or Google Colab. Make sure to adjust the file paths if your data is stored in a different location.

## Model and Training

This solution employs a state-of-the-art approach to image classification:

  * **Model:** We use the `efficientnet_b5.sw_in12k` model, pre-trained on a large dataset, and fine-tune it for this specific task.
  * **Data Augmentation:** To improve the model's robustness, we apply various data augmentation techniques using the `albumentations` library. These include random resized crops, horizontal flips, color jitters, and coarse dropouts.
  * **Mixup:** We also use the Mixup data augmentation technique, which helps the model to generalize better.
  * **Cross-Validation:** The model is trained using a 5-fold stratified cross-validation strategy. This helps to ensure that the model's performance is consistent across different subsets of the data.
  * **Training:** For each fold, the model is trained for 15 epochs using the AdamW optimizer and the OneCycleLR learning rate scheduler.

## Results

The model's performance is evaluated using the weighted F1 score. The Out-of-Fold (OOF) F1 score for this solution is **0.9433**.

## Future Work

Here are some ideas for further improving the model's performance:

  * **Experiment with other pre-trained models:** Try other state-of-the-art models like Vision Transformers or newer versions of EfficientNet.
  * **Hyperparameter tuning:** Optimize the learning rate, batch size, and other hyperparameters.
  * **Advanced data augmentation:** Explore more advanced data augmentation techniques.
  * **Ensemble more models:** Combine the predictions of different models to get a more robust final prediction.
