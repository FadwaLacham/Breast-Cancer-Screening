# Breast Cancer Detection - Project S5

This project aims to develop a binary classification system to detect breast cancer using deep learning models. We tested and compared three pre-trained neural network architectures: **VGG16**, **VGG19**, and **DenseNet201**.

## Table of Contents
1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Implemented Models](#implemented-models)
4. [Model Evaluation](#model-evaluation)
5. [Conclusion and Results](#conclusion-and-results)
6. [Interface Development](#interface-development)
7. [Code Link](#code-link)


---

## Project Description
Early detection of breast cancer is critical to reducing mortality rates. This project leverages computer vision models to analyze mammogram images and predict whether a sample is cancerous or not.

## Dataset
The dataset used in this project is the **Curated Breast Imaging Subset of DDSM (CBIS-DDSM)**, which contains mammography images labeled for training and evaluation.  
- **Source:** [CBIS-DDSM](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
- The images were preprocessed by resizing them to `224x224` pixels and normalizing pixel values to a range between 0 and 1.

## Implemented Models
We utilized three pre-trained convolutional neural network architectures:
- **VGG16**
- **VGG19**
- **DenseNet201**

Each model was fine-tuned by adding custom classification layers:
- **Dropout layers** to prevent overfitting.
- A `Flatten` layer to transform extracted features into 1D vectors.
- **Dense layers** with 1024 neurons each, followed by `BatchNormalization` and ReLU activation.
- A final `Dense(1)` layer with Sigmoid activation to produce probabilities between 0 and 1.

The pre-trained model weights were loaded without fully connected layers (`include_top=False`).

---

## Model Evaluation
The performance of each model was evaluated using various metrics, including precision, recall, F1-score, and overall accuracy. Below are the results:

| **Model**       | **Precision (Class 0)** | **Recall (Class 0)** | **F1-score (Class 0)** | **Precision (Class 1)** | **Recall (Class 1)** | **F1-score (Class 1)** | **Accuracy** | **Macro Avg** | **Weighted Avg** |
|------------------|--------------------------|------------------------|--------------------------|--------------------------|------------------------|--------------------------|--------------|---------------|------------------|
| **VGG16**        | 0.69                    | 1.00                  | 0.81                   | 1.00                    | 0.52                  | 0.69                   | 0.77         | 0.84          | 0.84             |
| **VGG19**        | 0.61                    | 1.00                  | 0.76                   | 1.00                    | 0.33                  | 0.50                   | 0.67         | 0.81          | 0.80             |
| **DenseNet201**   | 0.77                    | 0.91                  | 0.83                   | 0.88                    | 0.71                  | 0.79                   | 0.81         | 0.83          | 0.82             |

DenseNet201 achieved the best overall performance and was selected as the final model.

---

## Conclusion and Results
The **DenseNet201** model demonstrated superior performance in terms of weighted precision and recall. This model has been integrated into an interactive user interface for real-time classification.

---

## Interface Development
An interactive user interface was developed to allow users to upload mammogram images and receive predictions in real-time. The interface was implemented using **Streamlit** for a simple and accessible user experience.

---

## Code Link
For more details, please refer to the full code available on Kaggle:  
[Kaggle Notebook - Project S5](https://www.kaggle.com/code/fadwalacham/projet-s5)

