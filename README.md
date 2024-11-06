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

---

### Training the Model
- **Framework**: TensorFlow
- **Model Used**: Convolutional Neural Network (CNN) Deep Learning Model
- **Optimizer**: `adam`
- **Loss Function**: `SparseCategoricalCrossentropy`
- **Accuracy**: 99%
- **Saved Models**:
  - `potatoes.keras`
  - `potatoes.h5`
- **Location**: `saved_models/`
- **Install Python packages**: Recommended Anaconda3

```
pip3 install -r training/requirements.txt
pip3 install -r api/requirements.txt
```
- **Run Jupyter Notebook**
- **Directory**: `training/training.ipynb` contains the model-building steps.
---

### TensorFlow Serving
- **Serving Tool**: TensorFlow Serving
- **Directory**: Models saved in `saved_models/versions/`
- **API**: FastAPI used for local deployment using Docker
- **Directory**: API code in `api/`
- **Install Tensorflow Serving** ([Setup instructions](https://www.tensorflow.org/tfx/serving/setup))
- **Running the API**

##### Using FastAPI

1. Get inside `api` folder
2. To run the FastAPI Server using uvicorn go to `api/` and run main.py file
3. Your API is now running at `http://localhost:8000`
4. Test it in postman software at `http://localhost:8000/predict`
![image](https://github.com/user-attachments/assets/ddc61a54-f24d-40a7-a165-d19da006e714)


##### Using FastAPI & TF Serve

1. Get inside `api` folder

2. Run the TF Serve (Update config file path below) in Gitbash

```bash
docker run -t --rm -p 8501:8501 -v D:/code files/potato-blight-disease-detection:potato-blight-disease-detection tensorflow/serving --rest_api_port=8501 --model_config_file=/potato-blight-disease-detection/models.config
```

4. Run the FastAPI TF-Server using uvicorn directly run main-tf-serving.py
   
6. Your API is now running at `http://localhost:8000` by utilizing latest version model

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
- **Test**: Send POST request in Postman software to receive the class label and confidence score.
- **Running the Frontend**
```bash
cd frontend
```
```bash
npm run start
```
![image](https://github.com/user-attachments/assets/2f48b9b0-8780-42f0-b5e7-bb3a5a42a1d3)

---

### Mobile App Deployment
- **Platform**: Android & iOS
- **Framework**: React Native
- **Model**: Deployed TFLite model in GCP for mobile compatibility
- **Directory**: Mobile app code in `mobile-app/`
---

### Deploying the TF Model(.keras) on GCP

1. Create a [GCP account](https://console.cloud.google.com/freetrial/signup/tos?_ga=2.25841725.1677013893.1627213171-706917375.1627193643&_gac=1.124122488.1627227734.Cj0KCQjwl_SHBhCQARIsAFIFRVVUZFV7wUg-DVxSlsnlIwSGWxib-owC-s9k6rjWVaF4y7kp1aUv5eQaAj2kEALw_wcB).
2. Create a [Project on GCP](https://cloud.google.com/appengine/docs/standard/nodejs/building-app/creating-project) (Keep note of the project id).
3. Create a [GCP bucket](https://console.cloud.google.com/storage/browser/).
4. Upload the potatoes.keras model in the bucket in the path `models/potatos.keras`.
5. Install Google Cloud SDK ([Setup instructions](https://cloud.google.com/sdk/docs/quickstarts)).
6. Authenticate with Google Cloud SDK.

```bash
gcloud auth login
```

7. Run the deployment script.

```bash
cd gcp
gcloud functions deploy predict --runtime python312 --trigger-http --memory 1024 --project= your project_id
```

8. Your model is now deployed.
9. Use Postman to test the GCF using the GCP predict Function URL: https://us-central1-potato-blight-disease-detect.cloudfunctions.net/predict

---


### Deploying the TF Lite (.tflite) on GCP

1. Upload the 3.tflite model in the bucket in the path `models/3.tflite` from `tf-lite-models` directory.
2. Authenticate with Google Cloud SDK.

```bash
gcloud auth login
```

3. Run the deployment script.

```bash
cd gcp
gcloud functions deploy predict_lite --runtime python312 --trigger-http --memory 1024 --project= your project_id
```

4. Your model is now deployed.
5. Use Postman to test the GCF using the GCP predict-lite Function URL: https://us-central1-potato-blight-disease-detect.cloudfunctions.net/predict-lite

---

### Challenges
Encountered multiple challenges related to model optimization and deployment, providing a valuable learning experience in the process.

---

### References
- **Learning Resource**: [Codebasics YouTube Channel - Deep Learning Projects](https://www.youtube.com/playlist?list=PLeo1K3hjS3utJFNGyBpIvjWgSDY0eOE8S)

---

This project showcases a comprehensive approach to deploying deep learning models on various platforms.
