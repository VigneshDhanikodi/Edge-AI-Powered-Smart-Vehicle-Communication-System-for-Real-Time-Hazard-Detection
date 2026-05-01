<div align="center">

# 🚗 Edge-AI Powered Smart Vehicle Communication System
### Using Deep Learning & Hybrid V2V Protocols

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Jetson](https://img.shields.io/badge/Hardware-Jetson%20Nano%20Orin-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit)
[![Status](https://img.shields.io/badge/Status-Project%20Report-blueviolet?style=for-the-badge)]()

---

> **Research Report:** *Edge-AI Powered Smart Vehicle Communication System for Real-Time Hazard Detection* [cite: 1, 2]
>
> A decentralized V2V framework that classifies **real-time road hazards** into distinct categories using YOLOv11n — deployed on **Jetson Nano Orin** boards with a hybrid communication stack (WebSocket/MQTT/HTTP) ensuring zero reliance on cloud infrastructure[cite: 67, 70, 71, 73].

</div>

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [🌐 Overview](#-overview) |
| 2 | [🏗️ Repository Structure](#️-repository-structure) |
| 3 | [📦 Dataset Design](#-dataset-design) |
| 4 | [⚡ Communication Framework](#-communication-framework) |
| 5 | [🔤 Object Detection Pipeline](#-object-detection-pipeline) |
| 6 | [🧠 Model Architectures](#-model-architectures) |
| 7 | [🔧 V2V Telemetry Metrics](#-v2v-telemetry-metrics) |
| 8 | [🏋️ Hardware & Training Configuration](#️-hardware--training-configuration) |
| 9 | [📊 Results](#-results) |
| 10 | [🔬 Performance Trade-offs](#-performance-trade-offs) |
| 11 | [👁️ Class-Wise Detection Analysis](#️-class-wise-detection-analysis) |
| 12 | [🚀 Quickstart](#-quickstart) |
| 13 | [📁 Deployment Outline](#-deployment-outline) |
| 14 | [📜 Citation](#-citation) |

---

## 🌐 Overview

[cite_start]Cloud-based autonomous vehicle systems face critical challenges including high latency, poor connectivity in semi-urban areas, and expensive infrastructure[cite: 66, 158]. [cite_start]This project frames road safety as a **decentralized edge-computing problem** and applies state-of-the-art YOLO architectures and Wi-Fi protocols to solve it[cite: 67, 185].

### Key Contributions

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ✅  Jetson Nano Orin powered  —  fully decentralized Edge-AI framework │
│  ✅  YOLOv11n engine  —  TensorRT accelerated, INT8/FP16 quantized      │
│  ✅  Multi-protocol stack  —  WebSocket / MQTT / HTTP integration       │
│  ✅  Sub-50ms latency  —  V2V transmission in Line-of-Sight conditions  │
│  ✅  85% mAP@0.5 accuracy  —  trained robustly on the KITTI dataset     │
│  ✅  Full ablation study  —  streaming resolution vs throughput latency   │
│  ✅  3.2M parameter model  —  maintaining an ultra-light ~6.2 MB size   │
│  ✅  Dual-car prototype  —  validated on physical robotic vehicle nodes │
└─────────────────────────────────────────────────────────────────────────┘
```
[cite_start]*(Based on the system design and evaluation metrics detailed in the report[cite: 161, 388, 419, 474, 571, 577].)*

---

## 🏗️ Repository Structure

```
edge-ai-vehicle-system/
│
├── 📓 notebooks/
│   └── hazard_detection_eval.ipynb       ← Full evaluation & telemetry
│
├── 🐍 src/
│   ├── pipeline.py                       ← Real-time object detection (YOLOv11n)
│   ├── v2v_comms.py                      ← WebSocket/MQTT/HTTP sockets
│   └── display_interface.py              ← OLED/LCD visual feedback logic
│
├── 📊 data/                              ← Sample KITTI calibration sets
├── 🤖 models/                            ← Compiled TensorRT engine (.engine)
├── 📈 results/                           ← Latency & frame-drop logs
├── 🖼️ figures/                           ← Generated plots & confusion matrices
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📦 Dataset Design

### Real-World Training Guarantee

[cite_start]The perception model was developed utilizing the **KITTI dataset**, ensuring the system learns from high-quality, real-world road scenes from urban driving environments rather than synthetic data[cite: 419, 564].

| Class Category | Targets | Detection Priority | Challenge Level |
|:------|:---------|:-----------|:-----------|
| `Vehicles` | Cars, Trucks, Vans, Trams | High | Low (Large Scale) |
| `Pedestrians` | People walking/standing | Critical | High (Small Scale) |
| `Cyclists` | Bicycles, Motorcycles | Critical | High (Scale Variance) |
| `Obstacles` | Static road elements | Medium | Moderate |

### Component Sources

```
╔══════════════════════════════════════════════════════════════════════╗
║  SOURCE 1 ─ Core Processing Unit                                     ║
║  NVIDIA Jetson Nano Orin Developer Kit (Vibration-damped mount)      ║
╠══════════════════════════════════════════════════════════════════════╣
║  SOURCE 2 ─ Perception Layer                                         ║
║  720p/1080p Camera module (Chassis mounted & calibrated)             ║
╠══════════════════════════════════════════════════════════════════════╣
║  SOURCE 3 ─ V2V Network                                              ║
║  IEEE 802.11ac Wi-Fi (Dual-band 2.4/5 GHz)                           ║
╚══════════════════════════════════════════════════════════════════════╝
```
[cite_start]*(Derived from the hardware methodology[cite: 285, 287, 382, 383, 485].)*

---

## ⚡ Communication Framework

A hybrid protocol stack handles different data types prioritizing **urgent threat data** over background logging.

### V2V Protocols

| Protocol | Description | Application |
|:---------|:------------|:--------|
| `WebSocket` | Bidirectional, low-latency stream | Urgent hazard alerts (object class, distance) |
| `MQTT` | Asynchronous pub/sub architecture | Event logging & system telemetry |
| `HTTP` | Request-response standard | System diagnostics & remote configuration |

[cite_start]*(Protocol specifications sourced from the methodology[cite: 389, 390, 391].)*

---

## 🔤 Object Detection Pipeline

Visual data is treated as a **stream of sequential frames**, optimized through a real-time quantum of detection steps:

```
  Raw Camera Feed
        │
        ▼
  ① Input Scaler        1080p  ──────►  640x480 resolution
  ② Preprocessing       Lens Distortion ──►  Checkerboard Calibration
  ③ Core Inference      YOLOv11n  ────►  Feature Extraction & Fusion
  ④ Quantization        FP32  ────────►  FP16 / INT8
  ⑤ Hardware Accel      PyTorch  ─────►  TensorRT Engine
  ⑥ Confidence Gate     Score < 0.5  ─►  Drop
  ⑦ Box Suppression     NMS > 0.4  ───►  Merge
        │
        ▼
  Hazard Class & Distance  (Sent to V2V WebSocket)
```
[cite_start]*(Pipeline parameters derived from Chapter 3[cite: 424, 425, 427, 428, 492].)*

---

## 🧠 Model Architectures

### Architecture Overview

| Model | Parameters | Size | Target FPS | Optimization |
|:------|----------:|:----:|:----:|:------------:|
| `YOLOv11n` | ~3.2M | ~6.2 MB | 30-60 FPS | TensorRT + INT8 |

### Edge-AI vs Cloud ADAS — Head-to-Head

| Aspect | Cloud-Based ADAS | Edge-AI (Proposed) |
|:-------|:-----|:----|
| Processing Location | Remote Servers | On-device (Jetson Orin) |
| Network Dependency | High (Requires 4G/5G) | **Zero (Local Inference)** |
| Latency | High & Variable | **Deterministic & Low** |
| Infrastructure Cost | High | **Low (Embedded hardware)** |
| Privacy | Data leaves vehicle | **Data stays local** |

> **Why Edge-AI wins on vehicle safety:**
> V2V hazard detection requires instantaneous response. [cite_start]Relying on remote cloud servers introduces bandwidth constraints and unpredictable jitter[cite: 251, 252]. [cite_start]Processing directly on the Jetson Nano Orin utilizing TensorRT guarantees the necessary deterministic sub-100ms latency[cite: 257, 424].

---

## 🔧 V2V Telemetry Metrics

Metrics extracted from the custom TCP/IP and WebSocket client-server chat system implementation across dual Jetson nodes:

| # | Metric | Observed Value | Condition |
|:-:|:--------|:----:|:--------|
| 0 | `WebSocket Latency` | 5–20 ms | Baseline |
| 1 | `MQTT Logging Delay` | 10–20 ms | Asynchronous |
| 2 | `Communication Range` | 30–50 meters | 5 GHz Wi-Fi band |
| 3 | `Packet Delivery Ratio` | > 95% | Line-of-sight (LOS) |
| 4 | `Reconnection Time` | < 1 second | After signal dropout |
| 5 | `Client Avg Ping` | 1.02 ms | Network calibration phase |
| 6 | `Server Avg Ping` | 1.43 ms | Network calibration phase |
| 7 | `Client Jitter` | 13.70 ms | Real-time transmission |

[cite_start]*(Values sourced exactly from Table 4-1 Chat System Performance Metrics[cite: 539, 545].)*

---

## 🏋️ Hardware & Training Configuration

### YOLOv11n Model Profile

| Hyperparameter | Value |
|:---------------|:-----------:|
| Input Dimension | 640 x 640 |
| Convolutional Layers | ~60 - 70 |
| FLOPS | ~8.7 GFLOPS |
| NMS Threshold | 0.4 |
| Confidence Threshold | 0.5 |
| Dataset | KITTI |

[cite_start]*(Parameters from Table 4-3 and Section 3.2[cite: 428, 564, 577].)*

### Training Convergence Curve

```
  Loss
  │
1.4 ─┐  Initial Training Phase (Epoch 0)
     │╲
     │ ╲  Steady Convergence
     │  ╲─────────────────────────╮
     │                            ╰──── min_loss ≈ 0.8
  ────┼──────────────────────────────────► Epoch
      0    20   40   50
```
[cite_start]*(Based on the visual representation of the `train/box_loss` curve decreasing steadily over 50 epochs[cite: 583, 586, 589, 591, 593, 614, 616].)*

---

## 📊 Results

### Main Object Detection Results

| Rank | Metric | Score | Characteristic |
|:----:|:------|:--------:|:---------|
| 🥇 1 | **mAP@0.5** | **$\approx0.85$** | **Primary accuracy standard** |
| 🥈 2 | Precision | ~0.70 | False positive mitigation |
| 🥉 3 | F1-score (peak) | ~0.67 | Balance of PR curve |
| 4 | Recall | ~0.65 | Missed detection rate |
| 5 | mAP@0.5:0.95 | $\approx0.60$ | Stringent IoU threshold |

[cite_start]*(Performance Summary extracted from Table 4-4[cite: 580].)*

### Per-Class Detection Accuracy

| Error Class | Accuracy | Characteristics | Challenge |
|:------------|:---------:|:------|:--------|
| `Car` | 0.92 | Large bounding box | Low |
| `Tram` | 0.92 | Distinct features | Low |
| `Truck` | 0.89 | Variable size | Minor |
| `Van` | 0.89 | Similar to cars | Minor |
| `Cyclist` | 0.75 | Scale variations | Moderate |
| `Pedestrian` | 0.72 | Dataset imbalance | High |

[cite_start]*(Accuracy data points from Section 4.3 and Confusion Matrix[cite: 572, 674, 676, 680, 682, 683, 684].)*

---

## 🔬 Performance Trade-offs

Ablation testing on the **WebRTC video streaming** module to balance bandwidth efficiency and visual clarity (tested on 640p target):

### 1. Streaming Resolution vs Latency

| Resolution | Jitter (ms) | Packet Loss | Target Framerate | Overall Stability |
|:-----------------:|--------:|:-------------:|:-------------:|:-------------:|
| 1080p | 7.99 | 1.2% | 97% | **Too heavy for V2V** |
| **640p** | **6.22** | **1.8%** | **94%** | **Optimal Balance** |
| 480p | 6.70 | 3.6% | 90% | High packet loss |
| 360p | 5.88 | 3.2% | 91% | Diminished clarity |
| 240p | 5.56 | 1.5% | 93% | Poor detection |

[cite_start]*(Extracted from Table 4-2 Video Streaming Performance Metrics[cite: 561].)*

### 2. Operational Constraints Identified

* [cite_start]**Wi-Fi Handoffs:** Periodic latency spikes observed during handoffs or signal interference areas[cite: 720].
* [cite_start]**Broker Reliability:** The MQTT broker requires a manual restart after long idle periods, necessitating future automated watchdogs[cite: 721].
* [cite_start]**Resource Contention:** Concurrent video streaming and WebSocket messaging caused intermittent frame drops[cite: 723].
* [cite_start]**Non-Line-of-Sight (NLoS):** Latency and frame stability visibly drop when vehicles exceed 50 meters or lose direct sightlines[cite: 724].

---

## 👁️ Class-Wise Detection Analysis

### Normalized Confusion Matrix Insights

Analysis of the YOLOv11n confusion matrix reveals where the model excels and struggles:

```
  High Accuracy Example: "Car"
  ─────────────────────────────────────────────────
  Prediction      Accuracy Score   ████ Confidence
  ─────────────────────────────────────────────────
  True Car        0.92             ██████████████████
  Background      0.07             █
  ...
  ─────────────────────────────────────────────────

  Lower Accuracy Example: "Pedestrian"
  ─────────────────────────────────────────────────
  Prediction      Accuracy Score   ████ Confidence
  ─────────────────────────────────────────────────
  True Pedestrian 0.72             ████████████
  Background      0.27             ████
  ...
  ─────────────────────────────────────────────────
```
[cite_start]*(Derived from the normalized confusion matrix values showing 0.92 for cars and 0.72 for pedestrians with higher background confusion[cite: 674, 680, 700].)*

---

## 🚀 Quickstart

### Installation

```bash
git clone https://github.com/YourOrg/edge-ai-vehicle-system.git
cd edge-ai-vehicle-system
pip install -r requirements.txt
```

### Flash the Jetson Nano Orin

[cite_start]Ensure your board is running Ubuntu-based JetPack OS[cite: 489].

```bash
# Setup Python, OpenCV, PyTorch, and TensorRT environments
sudo apt-get update
sudo apt-get install python3-opencv
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install python-socketio paho-mqtt
```
[cite_start]*(Libraries listed in the software stack methodology[cite: 489, 490].)*

### Run the Dual-Node Experiment

```python
# On Car 1 (Transmitter/Detector)
python3 src/pipeline.py --role transmitter --model models/yolov11n.engine

# On Car 2 (Receiver)
python3 src/v2v_comms.py --role receiver --ip <CAR_1_IP>
```

### Environment

| Requirement | Specification |
|:------------|:-------|
| Hardware | Jetson Nano Orin Developer Kit |
| OS | JetPack OS (Ubuntu-based) |
| Vision | OpenCV |
| Deep Learning | PyTorch, TensorRT |
| Networking | `socket.io`, `paho-MQTT`, IEEE 802.11ac Wi-Fi |

---

## 📁 Deployment Outline

| Phase | Focus | Description |
|:----:|:------|:------------|
| **0** | Procurement | [cite_start]Source Jetson kits, CSI cameras, Wi-Fi dongles, and DC converters [cite: 482] |
| **1** | Assembly | [cite_start]Mount Jetson on vibration-damped brackets; secure & align cameras [cite: 485, 486] |
| **2** | OS Setup | [cite_start]Flash JetPack OS; install PyTorch and TensorRT [cite: 489] |
| **3** | Networking | [cite_start]Configure dual-band Wi-Fi and install `socket.io` / `paho-MQTT` [cite: 490] |
| **4** | Calibration | [cite_start]Use checkerboards to correct lens distortion [cite: 492] |
| **5** | Engine Build | [cite_start]Compile YOLOv11n to TensorRT with FP16/INT8 [cite: 424, 425] |
| **6** | Baseline Test | [cite_start]Run `measure_latency()` scripts to check ping/jitter [cite: 533] |
| **7** | Live Streaming | [cite_start]Initiate WebRTC 640p video transmission [cite: 549, 550] |
| **8** | Active Inference | Feed camera frames to YOLO pipeline; log metrics |
| **9** | V2V Alerting | [cite_start]Transmit high-priority bounding boxes via WebSocket [cite: 466] |
| **10** | Validation | [cite_start]Monitor CPU/GPU thermal load and transmission success rates [cite: 498] |

---

## 📜 Citation

```bibtex
@techreport{Vendhan2026EdgeAI,
  title       = {Edge-AI Powered Smart Vehicle Communication System for Real-Time Hazard Detection},
  author      = {Kavin Vendhan V S and Deepak Kumar V B and Vignesh D},
  institution = {Amrita Vishwa Vidyapeetham, Department of Electronics and Communication Engineering},
  year        = {2026},
  month       = {March},
  type        = {Bachelor of Technology Project Report},
  address     = {Coimbatore - 641112}
}
```
[cite_start]*(Citation block formatted using the official title page metadata[cite: 1, 5, 7, 9, 21, 23].)*

</div>
