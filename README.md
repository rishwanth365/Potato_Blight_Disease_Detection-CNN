# Potato Blight Disease Detection using CNN

This project is a deep learning model designed to detect potato leaf diseases, specifically Late Blight, Early Blight, and Healthy leaf samples. Built using Convolutional Neural Networks (CNN), the model has been deployed across web, cloud, and mobile platforms.

## Architecture
![image](https://github.com/user-attachments/assets/1efc0643-5026-485b-845e-a60c103fabc0)
![image](https://github.com/user-attachments/assets/fe272b86-794f-45df-89bb-5b9245a16e29)



## Table of Contents
- [Dataset](#dataset)
- [Model Building](#model-building)
- [TensorFlow Serving](#tensorflow-serving)
- [Model Optimization](#model-optimization)
- [Frontend & Deployment](#frontend--deployment)
- [Mobile App Deployment](#mobile-app-deployment)
- [Challenges](#challenges)
- [References](#references)

---

### Dataset
- **Source**: [Kaggle - Plant Village Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
- **Classes Used**: Late Blight, Early Blight, Healthy
- **Directory**: `training/training.ipynb` contains the model-building steps.

---

### Model Building
- **Framework**: TensorFlow
- **Architecture**: Convolutional Neural Network (CNN)
- **Optimizer**: `adam`
- **Loss Function**: `SparseCategoricalCrossentropy`
- **Accuracy**: 99%
- **Saved Models**:
  - `potatoes.keras`
  - `potatoes.h5`
- **Location**: `saved_models/`

---

### TensorFlow Serving
- **Serving Tool**: TensorFlow Serving
  - **Directory**: Models saved in `saved_models/versions/`
- **API**: FastAPI used for local deployment using Docker
- **Directory**: API code in `api/`

---

### Model Optimization
- **Conversion**: Converted the model to TensorFlow Lite (TFLite) and quantized for optimization.
- **Saved Models**: TFLite models are in `tf-lite-models/`
- **Optimization Code**: Check `training/` directory for related `.ipynb` files.

---

### Frontend & Deployment
- **Frontend**: HTML, CSS and Java Script
- **Backend (Local)**: React JS and React Native
- **Cloud Deployment**: Google Cloud Platform (GCP)
  - **Cloud Function**: Model deployed via Cloud Run
  - **Endpoint**: GCP Predict Function URL: https://us-central1-potato-blight-disease-detect.cloudfunctions.net/predict
  - **Test**: Use POST method in Postman to receive the class label and confidence score.

---

### Mobile App Deployment
- **Platform**: Android & iOS
- **Framework**: React Native
- **Model**: Deployed TFLite model in GCP for mobile compatibility
- **Endpoint**: GCP predict-lite Function URL for mobile:
- **Testing**: Verified on Android and iOS emulators
- **Directory**: Mobile app code in `mobile-app/`

---

### Challenges
Encountered multiple challenges related to model optimization and deployment, providing a valuable learning experience in the process.

---

### References
- **Learning Resource**: [Codebasics YouTube Channel - Deep Learning Projects](https://www.youtube.com/playlist?list=PLeo1K3hjS3utJFNGyBpIvjWgSDY0eOE8S)

---

This project showcases a comprehensive approach to deploying deep learning models on various platforms.
