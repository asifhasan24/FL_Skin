# 🧠 Explainable Federated Learning with Homomorphic Encryption for Skin Cancer Diagnosis

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

## ✅ Key Contributions

1. 🚀 **Lightweight Multi-Scale CNN**: Outperforms state-of-the-art models with significantly fewer trainable parameters, achieving an optimal balance between efficiency and performance.
2. 🔒 **Federated Learning for Privacy**: Trains collaboratively across devices/institutions without sharing raw data, addressing privacy and regulatory concerns.
3. 🛡️ **Homomorphic Encryption**: Securely aggregates model updates, protecting sensitive medical data during training.
4. 🧩 **SHAP-based Explainability**: Adds interpretability to the AI predictions, aiding clinicians in decision-making and boosting trust in model outputs.

---

<h2 style="text-align: center;">🖼️ CNN Model Architecture</h2>
<p align="center">
  <img src="https://github.com/asifhasan24/FL_Skin/blob/main/Picture1.png" width="600"/>
</p>


---

## 🗂️ Repository Structure

```bash
.
├── notebook.ipynb                 # Main Jupyter Notebook
├── CNN Model.pdf                  # PDF of CNN architecture
├── Diagram.pdf                    # PDF of FL + HE system diagram
├── Federated Algo.pdf             # PDF of the FL algorithm
├── models/                        # Model weights and checkpoints
├── utils/                         # Helper scripts for encryption, SHAP, etc.
└── README.md                      # This file
