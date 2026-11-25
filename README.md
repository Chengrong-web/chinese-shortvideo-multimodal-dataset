# A Multimodal Data Mining and Machine Learning Framework for Predicting Engagement and Content Circulation on Short-Video Platforms

[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview
This repository contains the anonymized empirical dataset ($N = 4,782$ validated records) and the reproducible Python computational pipeline accompanying the IEEE Access manuscript:

> **Title: A Multimodal Data Mining and Machine Learning Framework for Predicting Engagement and Content Circulation on Short-Video Platforms  
> **Authors: Cheng Rong, Zhang Jicheng (School of Digital Media, Geely University of China)  
> **Corresponding Author: Cheng Rong (`CR18728402460@163.com`)

## Repository Structure
- `data/guochao_multimodal_audited_dataset_N4782.csv`: Audited & anonymized dataset ($N=4,782$)
- `src/01_statistical_audit.py`: Benford's Law & Z-score (>3.5) bot filtering
- `src/02_multimodal_feature_extraction.py`: OpenCV, MediaPipe, PaddleOCR, Whisper AI & Chinese-RoBERTa-wwm pipeline
- `src/03_regression_and_ml_modeling.py`: Fixed-Effects OLS, Interaction Model & Random Forest (k=10)
- `requirements.txt`: Python dependencies
- `LICENSE`: MIT License (Code) & CC-BY-4.0 (Data)

## Dataset Description
The dataset covers a 90-day longitudinal observation window ($T_0$ and $T_1 = T_0 + 7$ days) of short videos associated with the **Guochao (国潮, China-Chic)** trend across Douyin ($n = 1,912$), Kuaishou ($n = 1,435$), and Bilibili ($n = 1,435$).

## Ethical Compliance & Privacy
All records were collected exclusively from publicly accessible surface metadata without private authentication. Creator handles and video IDs have been fully anonymized to comply with OSINT research standards and data protection regulations (GDPR/PIPL).
