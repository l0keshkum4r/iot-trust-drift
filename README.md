# 🛡️ IoT Trust & Drift Analytics Platform

> **Hybrid Approach for Dynamic Trust Evaluation, Behavioral Drift Detection, and Explainable Risk Assessment in IoT Networks**

An intelligent cybersecurity analytics platform that continuously evaluates the trustworthiness of IoT devices, detects behavioral drift, identifies anomalies, and provides explainable security insights from network telemetry.

The platform combines machine learning, policy-based security analysis, dynamic trust scoring, and explainable AI to help security analysts understand **what happened, why it happened, and how much risk it introduces**.

---

## ✨ Key Capabilities

### 🔍 Anomaly Detection

Detect abnormal device behavior using machine learning models trained on device-specific activity patterns.

* Isolation Forest–based anomaly detection
* Device-type-aware analysis
* Real-time risk identification
* Attack and compromise detection

### 📈 Behavioral Drift Detection

Monitor long-term changes in device behavior and identify concept drift before it becomes a security issue.

* Baseline behavior learning
* Continuous drift monitoring
* Feature-level drift analysis
* Early warning indicators

### 🛡️ Security Policy Engine

Evaluate network activity against predefined security rules and compliance policies.

* Device-specific security policies
* Rule-based threat identification
* Policy violation reporting
* Risk categorization

### ⭐ Dynamic Trust Scoring

Generate interpretable trust scores for every IoT device.

* Trust score range: **0–100**
* Multi-factor trust computation
* Continuous trust updates
* Risk-aware scoring model

### 🧠 Explainable AI

Provide human-readable explanations for every alert, anomaly, drift event, and trust score.

* Evidence-based reporting
* Feature importance analysis
* Analyst-friendly explanations
* Transparent decision making

### 📊 Interactive Security Dashboard

Visualize device trust, security posture, anomalies, drift trends, and policy violations through an intuitive dashboard.

---

## 🏗️ System Architecture

```text
                Network Telemetry
                        │
                        ▼
              Feature Engineering
                        │
                        ▼
        ┌─────────────────────────────┐
        │     Anomaly Detection       │
        └─────────────────────────────┘
                        │
                        ▼
        ┌─────────────────────────────┐
        │     Drift Detection         │
        └─────────────────────────────┘
                        │
                        ▼
        ┌─────────────────────────────┐
        │     Policy Evaluation       │
        └─────────────────────────────┘
                        │
                        ▼
        ┌─────────────────────────────┐
        │     Trust Scoring Engine    │
        └─────────────────────────────┘
                        │
                        ▼
        ┌─────────────────────────────┐
        │ Explainability & Evidence   │
        └─────────────────────────────┘
                        │
                        ▼
               API & Dashboard
```

---

## ⚙️ Analytics Pipeline

The platform executes a multi-stage cybersecurity analytics workflow:

### 1️⃣ Feature Engineering

Transforms raw network telemetry into structured device-level features.

### 2️⃣ Anomaly Detection

Identifies suspicious activity using unsupervised machine learning.

### 3️⃣ Drift Detection

Measures behavioral deviation from historical baselines.

### 4️⃣ Policy Evaluation

Checks compliance with security rules and device profiles.

### 5️⃣ Trust Computation

Calculates a dynamic trust score for every device.

### 6️⃣ Explainability Generation

Creates analyst-friendly evidence reports and risk explanations.

---

## 📊 Generated Outputs

| Output File            | Description                        |
| ---------------------- | ---------------------------------- |
| `anomaly_scores.csv`   | Device anomaly scores              |
| `drift_results.csv`    | Behavioral drift metrics           |
| `policy_results.csv`   | Security policy evaluation results |
| `trust_scores.csv`     | Dynamic trust scores               |
| `evidence_reports.csv` | Explainable AI reports             |

---

## 🚀 Quick Start

### 1. Create a Python Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Analytics Pipeline

```bash
python run_pipeline.py
```

Generated outputs will be stored in:

```text
data/
├── anomaly_scores.csv
├── drift_results.csv
├── policy_results.csv
├── trust_scores.csv
└── evidence_reports.csv
```

### 3. Start the API

```bash
python api/server.py
```

API available at:

```text
http://localhost:8001
```

### 4. Launch the Dashboard

```bash
cd dashboard

npm install
npm start
```

Open:

```text
http://localhost:3000
```

---

## 🧪 Synthetic Dataset Generation

Generate realistic IoT network traffic containing both normal and malicious behavior:

```bash
python simulators/generate_dataset.py
```

Supported attack simulations include:

* Distributed Denial of Service (DDoS)
* Port Scanning
* Data Exfiltration
* Device Compromise Scenarios
* Traffic Flooding Attacks

---

## 🔬 Research Objectives

This project explores the intersection of:

* IoT Security
* Trust Management Systems
* Concept Drift Detection
* Explainable Artificial Intelligence (XAI)
* Behavioral Analytics
* Cyber Threat Detection
* Adaptive Security Frameworks

---

## 🎯 Future Enhancements

* Deep Learning-Based Trust Models
* Online Incremental Learning
* Federated Trust Evaluation
* Edge-Based Analytics
* Graph Neural Networks for Device Relationships
* Real-Time Streaming Analytics
* Threat Intelligence Integration
* Digital Twin–Based Security Monitoring


