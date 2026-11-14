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

### AWS EC2 High-Availability Architecture (`AWSEC2` Branch)

```text
                                  [ Users / Clients ]
                                           │
                                    HTTP (Port 80)
                                           ▼
                            ┌─────────────────────────────┐
                            │   Internet Gateway (IGW)    │
                            └──────────────┬──────────────┘
                                           │
┌──────────────────────────────────────────┼──────────────────────────────────────────┐
│ AWS VPC (10.0.0.0/16)                    │                                          │
│                                          ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │  Public Route Table (0.0.0.0/0 ──► IGW)                                     │   │
│   └──────────────────────┬───────────────────────────────┬──────────────────────┘   │
│                          │                               │                          │
│  ┌───────────────────────▼───────────┐   ┌───────────────▼───────────────────────┐  │
│  │ Availability Zone A               │   │ Availability Zone B                   │  │
│  │ Public Subnet 1 (10.0.1.0/24)     │   │ Public Subnet 2 (10.0.2.0/24)         │  │
│  │                                   │   │                                       │  │
│  │  ┌─────────────────────────────┐  │   │  ┌─────────────────────────────────┐  │  │
│  │  │ Application Load Balancer   │──┼───┼──│ Application Load Balancer (ALB) │  │  │
│  │  │ (ALB Node - Port 80)        │  │   │  │ (ALB Node - Port 80)            │  │  │
│  │  └──────────────┬──────────────┘  │   │  └────────────────┬────────────────┘  │  │
│  └─────────────────┼─────────────────┘   └───────────────────┼───────────────────┘  │
│                    │                                         │                      │
│                    └────────────────────┬────────────────────┘                      │
│                                         │                                           │
│                                         ▼                                           │
│                     ┌───────────────────────────────────────┐                       │
│                     │   ALB Target Group (HTTP : 8000)      │                       │
│                     │   Health Check: GET /                 │                       │
│                     └───────────────────┬───────────────────┘                       │
│                                         │                                           │
│   ┌─────────────────────────────────────┼───────────────────────────────────────┐   │
│   │ Auto Scaling Group (ASG: Min 2, Desired 2, Max 4)                           │   │
│   │ Launch Template + IAM Instance Profile (S3 Read Access)                     │   │
│   │                                     │                                       │   │
│   │  ┌──────────────────────────────────▼─┐ ┌────────────────────────────────┐  │   │
│   │  │ EC2 Instance 1 (AZ-a / Subnet 1)   │ │ EC2 Instance 2 (AZ-b / Subnet 2)│  │   │
│   │  │ • Docker Container (Port 8000)     │ │ • Docker Container (Port 8000) │  │   │
│   │  │ • app.py + artifacts/car_model.pkl │ │ • app.py + artifacts/car_model │  │   │
│   │  └──────────────────┬─────────────────┘ └────────────────┬───────────────┘  │   │
│   └─────────────────────┼────────────────────────────────────┼──────────────────┘   │
└─────────────────────────┼────────────────────────────────────┼──────────────────────┘
                          │                                    │
                          └─────────────────┬──────────────────┘
                                            │ Outbound HTTPS (443) via IGW
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │ External Services (Bootstrap via User Data)  │
                     │ • GitHub (AWSEC2 branch code & Dockerfile)   │
                     │ • AWS S3 Bucket: eg-cars-1 (DVC dataset)     │
                     └──────────────────────────────────────────────┘

```

---

### Component & Security Group Matrix

| Component | Configuration Details | Traffic / Security Rules |
| --- | --- | --- |
| **VPC & Subnets** | CIDR `10.0.0.0/16` with **2 Public Subnets** (`10.0.1.0/24` in AZ-a, `10.0.2.0/24` in AZ-b) | Auto-assign public IPv4 enabled on both subnets so EC2 instances can pull from GitHub & S3 via IGW |
| **Internet Gateway** | Attached to VPC; linked via Route Table (`0.0.0.0/0` $\rightarrow$ `igw-xxx`) | Routes inbound ALB traffic and outbound `git clone` / `dvc pull` / `docker build` traffic |
| **ALB Security Group** | `eg-cars-alb-sg` | **Inbound:** HTTP `80` from `0.0.0.0/0`<br>

<br>**Outbound:** Custom TCP `8000` to `eg-cars-ec2-sg` |
| **EC2 Security Group** | `eg-cars-ec2-sg` | **Inbound:** Custom TCP `8000` **only** from `eg-cars-alb-sg` (plus SSH `22` from your IP if needed)<br>

<br>**Outbound:** All traffic (`0.0.0.0/0`) |
| **IAM Instance Role** | Attached to Launch Template (`AmazonS3ReadOnlyAccess` or scoped to `eg-cars-1`) | Allows `dvc pull` on EC2 boot without hardcoding AWS access keys in `userdata.sh` |
| **ASG & Target Group** | Spans both public subnets; Target Group listens on `HTTP:8000` | Automatically replaces unhealthy containers/instances and scales out on CPU > 70% |

---

### `userdata.sh` Bootstrap Flow (Launch Template)

When the ASG launches an EC2 instance on the **`AWSEC2`** branch, `userdata.sh` executes this exact pipeline:

```bash
#!/bin/bash
set -e

apt-get update -y
apt-get install -y git docker.io python3-pip
systemctl enable --now docker

cd /opt
git clone -b AWSEC2 https://github.com/AbdullrahmanEissa/egypt-cars-prediction-model.git
cd egypt-cars-prediction-model

pip3 install dvc[s3]
dvc pull

docker build -t egypt-cars-prediction-model:latest .
docker run -d --restart always -p 8000:8000 --name eg-cars-api egypt-cars-prediction-model:latest

```

---

### Push Diagram to `AWSEC2` Branch

If you want to create the `AWSEC2` branch and save this diagram directly into `README.md` or `ARCHITECTURE.md`:

```bash
git checkout -b AWSEC2
git add ARCHITECTURE.md
git commit -m "docs(infra): add AWS VPC, ALB, and ASG architecture diagram"
git push -u origin AWSEC2

```
