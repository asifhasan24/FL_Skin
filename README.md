# Explainable Federated Learning with Homomorphic Encryption for Skin Cancer Diagnosis

This repository contains a Jupyter Notebook implementing an **Explainable Federated Learning (FL)** model with **Homomorphic Encryption (HE)** using **TensorFlow** and **Keras** for skin cancer classification.

---

## 📜 Abstract

Skin diseases pose a major public health challenge, with skin cancer being the most prevalent malignancy. Early detection is critical for improving patient outcomes, yet traditional diagnostic methods rely on expert evaluation, which is limited by clinician variability, image quality, and accessibility constraints.

Deep learning (DL)-based models offer automated diagnostic capabilities but require large-scale centralized data aggregation, which conflicts with privacy regulations and poses security risks.

**Federated Learning (FL)** enables collaborative training across multiple hospitals without sharing raw patient data. However, it introduces communication overhead, security vulnerabilities, and interpretability challenges.

To address these issues, we propose an **Encrypted FedAvg-Based Explainable Federated Learning** approach utilizing a **Lightweight Deep Learning Multi-Scale Convolutional Neural Network (LWMS-CNN)** for efficient, privacy-preserving, and interpretable skin cancer diagnosis.

- **Homomorphic Encryption (HE)** ensures secure model aggregation and prevents privacy breaches.
- **SHapley Additive exPlanations (SHAP)** enhances interpretability, allowing clinicians to understand AI-driven predictions.

> 🧪 Experimental results on the **HAM10000** dataset show our model achieves **98.6% accuracy**, with only a **0.3% performance tradeoff** after encryption — ensuring robust security with reliable diagnostics.

---

<h2 style="text-align: center;">🖼️ Used Federated Algorithm with Homomorphic Encryption</h2>
<p align="center">
  <img src="https://github.com/asifhasan24/FL_Skin/blob/main/images/final fig.jpg" width="600"/>
</p>

---


Below is the updated "How to Run" section with instructions to download the required image folders from Kaggle:

---

## How to Run

### Install Dependencies
In your terminal, navigate to the repository directory and run:
```bash
pip install -r requirements.txt
```

### Download the Image Folders
Download the folders **"HAM10000_images_part_1"** and **"HAM10000_images_part_2"** from the following Kaggle dataset:
[https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

Place these folders in your repository directory (or update the paths in `data_preprocessing.py` accordingly).

### Preprocessing
Run the data preprocessing script to load the CSV, perform oversampling, split the data into training and test sets, and save the client data:
```bash
python data_preprocessing.py
```

### Start the Server (Aggregator)
Start the server script. It waits for encrypted updates from all clients, aggregates them homomorphically, and publishes the aggregated encrypted update:
```bash
python server.py
```

### Run the Clients
In parallel, run the client training script for each client (IDs 1–5). Note that client 1 will generate and share the HE context at the very beginning; thereafter, all clients follow the same procedure.

```bash
python client.py 1
python client.py 2
python client.py 3
python client.py 4
python client.py 5
```
### Explainability
To generate explainability reports, each client may run:

```bash
python client_explain.py 1
python client_explain.py 2
python client_explain.py 3
python client_explain.py 4
python client_explain.py 5
```



---


## ✅ Key Contributions

1. 🚀 **Lightweight Multi-Scale CNN**: Outperforms state-of-the-art models with significantly fewer trainable parameters, achieving an optimal balance between efficiency and performance.
2. 🔒 **Federated Learning for Privacy**: Trains collaboratively across devices/institutions without sharing raw data, addressing privacy and regulatory concerns.
3. 🛡️ **Homomorphic Encryption**: Securely aggregates model updates, protecting sensitive medical data during training.
4. 🧩 **SHAP-based Explainability**: Adds interpretability to the AI predictions, aiding clinicians in decision-making and boosting trust in model outputs.

