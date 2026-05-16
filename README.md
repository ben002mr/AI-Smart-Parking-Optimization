# AI-Driven Smart Parking System for Urban Mobility Optimization
AI-Driven Smart Parking System using IoT and Cloud-based Predictive Approach for Urban Mobility Optimization. Presented at IEEE ICUIS 2025.

## 📄 Research Publication
Published in IEEE Xplore:  
🔗 https://ieeexplore.ieee.org/document/11380569

## 📌 Research Overview
This project presents an end-to-end autonomous sensing framework presented at **IEEE ICUIS 2025**. It bridges the gap between IoT edge sensing and cloud-based deep learning to predict parking availability.

## 🛠️ Project Structure & Files
* **📂 hardware/**: `Smart_Parking_ESP8266_Firmware.ino` — C++ logic for ultrasonic sensing and cloud data transmission.
* **📂 ml_model/**: `Parking_Occupancy_BiLSTM_Model.py` — Python implementation of the Bidirectional LSTM forecasting model.
* **📂 data/**: `Sample_Synthetic_Parking_Data.csv` — 130 days of synthetic occupancy data used for model training.
* **📂 docs/**: Includes the **IEEE Certificate of Presentation** and the full research paper.

## 🚀 Technical Highlights
* **Edge Layer:** ESP8266 + HC-SR04 sensors calibrated for vehicle detection (<10cm threshold).
* **AI Layer:** Bi-LSTM model featuring 256/128/64 units, optimized to a **14.82% MAPE**.
* **Cloud Layer:** Designed for **AWS Lambda** integration and **Google Maps API** for real-time user navigation.

## 👥 Contributions & Collaboration
This research was a collaborative effort. My primary contributions included:
* **System Architecture:** Designing the end-to-end flow from IoT sensors to Cloud.
* **AI Development:** Implementing and optimizing the **Bi-LSTM** model and data preprocessing pipeline.
* **Cloud Integration:** Configuring AWS IoT Core and Lambda for real-time inference.
* **Hardware Prototyping:** Developing the ESP8266 firmware and sensor calibration.

*Co-authors: Soundari V., Arjun T. T., Simon Jose Jesuraj E. D.*

---
### 🎓 Relevance to Autonomous Systems
This repository demonstrates a complete control loop: **Data Acquisition (Sensors) → Transmission (IoT) → Intelligent Processing (AI/Cloud) → Actionable Output (User Navigation).**

