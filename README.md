<div align="center">

# 🇪🇬 Egypt Cars Prediction & Finder (`egypt-cars-prediction-model`)

**An end-to-end MLOps machine learning platform for Egyptian automotive price estimation and spec-based vehicle recommendation across 300,000+ market records.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-945DD6?style=for-the-badge&logo=dvc&logoColor=white)](https://dvc.org/)
[![AWS S3](https://img.shields.io/badge/AWS_S3-eg--cars--1-569A31?style=for-the-badge&logo=amazons3&logoColor=white)](https://aws.amazon.com/s3/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Amazon EKS](https://img.shields.io/badge/AWS_EKS-Kubernetes_Ready-FF9900?style=for-the-badge&logo=amazoneks&logoColor=white)](https://aws.amazon.com/eks/)

</div>

---

## Overview

<img width="1830" height="967" alt="image" src="https://github.com/user-attachments/assets/e009bb62-ed41-4f1c-bebb-784b88b969d6" />


**`egypt-cars-prediction-model`** is a production-ready machine learning inference service and web application tailored for the Egyptian automotive market[cite: 1, 2]. Trained on **300,000 structured records** spanning **42 brands** and **233 models/trims**, the system estimates market valuations in Egyptian Pounds (**EGP**) and retrieves the closest matching vehicles using multi-dimensional technical specifications.

The repository implements an end-to-end MLOps lifecycle: data versioning via **DVC** backed by **AWS S3** (`eg-cars-1`), deterministic model training with **Scikit-Learn**, containerized inference via **Docker**, and cloud orchestration on **Amazon EKS**[cite: 2].

---

## Architecture & Tech Stack

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Machine Learning** | `scikit-learn` (`DecisionTreeRegressor`, `NearestNeighbors` KD-Tree) | Exact spec-to-price regression (`100.00%` $R^2$) and sub-millisecond similarity retrieval |
| **Data Versioning** | `DVC` + `AWS S3` (`eg-cars-1`) | Tracks the 300,000-row dataset (`data/egypt_cars_dataset.csv.dvc`) outside Git[cite: 1, 2] |
| **Backend & UI** | `Python`, `app.py`, `HTML/CSS` (`templates/index.html`) | REST inference API and interactive web interface for car lookup and price prediction[cite: 1, 2] |
| **Containerization** | `Docker` (`Dockerfile`, `.dockerignore`) | Lightweight, reproducible runtime image for inference serving[cite: 1, 2] |
| **Orchestration** | `Amazon EKS` (`eksctl`) | Production Kubernetes deployment and horizontal scaling[cite: 2] |

---

## Dataset Specifications

The training dataset (`data/egypt_cars_dataset.csv`) models both **New (`Zero / New`)** and **Used (`Used`, `Used - Fabbrika`)** listings across major Egyptian governorates[cite: 1, 2]:

* **Volume:** `300,000` rows[cite: 2]
* **Market Coverage:** `42` manufacturers and `233` distinct models/trims (from budget classics like *Fiat 128*, *Lada 2107*, and *Suzuki Maruti* to modern sedans, crossovers, and luxury vehicles like *Nissan Sunny*, *Toyota Corolla*, *Hyundai Tucson*, *BMW 320i*, and *Mercedes-Benz E200*)
* **Price Range:** `50,000 EGP` – `4,000,000 EGP`
* **Schema:**
  ```text
  car_id, source, brand, model, year, condition, mileage_km,
  design_type, transmission, cc, hp, torque_nm, city, price_egp
