# ECG Anomaly Detection with Transformers

This project explores a self-supervised Transformer-based approach for ECG anomaly detection. 
The model is trained exclusively on normal ECG signals and detects abnormal patterns using 
reconstruction error. A residual-based error localization method highlights waveform segments 
that deviate from normal morphology.

## Dataset

The dataset used in this project is not included in this repository due to copyright restrictions.
Please obtain the dataset from the official source and place it in a `dataset/` directory.
# bachelor-thesis
Deep learning models for ECG analysis often lack interpretability and require labeled data. This project uses a self-supervised Transformer trained on normal ECG signals to detect anomalies via reconstruction error and localize abnormal waveform segments for improved interpretability.
