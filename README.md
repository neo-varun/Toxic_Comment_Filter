# Toxic Comment Classifier

## Overview
This project is a Toxic Comment Classification Application built using PyTorch, Flask, and ONNX. It detects and classifies toxic language in text comments into six categories: Toxic, Severe Toxic, Obscene, Threat, Insult, and Identity Hate. The application leverages a DistilBERT-based model, with options for model optimization and ONNX export for efficient inference.

## Features
- **Multi-label Toxicity Detection** – Analyze text and get predictions for multiple toxicity categories
- **DistilBERT Model** – Utilizes a pre-trained transformer for robust language understanding
- **Interactive Web Interface** – User-friendly UI for training, prediction, and analytics
- **Model Training Interface** – Train and fine-tune the model directly through the web interface
- **Model Optimization** – Export to ONNX and prune the model for faster inference
- **Comprehensive Analytics** – View detailed model performance metrics
- **CUDA Support** – GPU acceleration for faster training and inference (if available)

## Prerequisites

### Dataset Download
Before running the application, you need to download the Jigsaw Toxic Comment Classification dataset from Kaggle:
1. Visit [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
2. Download `train.csv`
3. Place the file in the following structure:
```
data/
└── train.csv
```

### System Requirements
- Python 3.8+
- CUDA-capable GPU (optional but recommended for training)

## Installation & Setup

### Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Application
1. Activate your virtual environment
2. Launch the Flask server:
```bash
python app.py
```
3. Open your web browser and navigate to: `http://localhost:8000`

## How the Program Works

### Application Components
1. **Data Preprocessing (`data_preprocessing.py`)**
   - Cleans and tokenizes text data
   - Prepares input tensors for model training and inference
   - Handles multi-label targets for six toxicity categories

2. **Model Training (`model_training.py`)**
   - Fine-tunes DistilBERT for multi-label classification
   - Supports custom batch size, learning rate, and epochs
   - Tracks and saves model checkpoints and metrics

3. **Model Optimization (`model_optimization.py`)**
   - Prunes the trained model for efficiency
   - Exports the model to ONNX format for fast inference
   - Reports model size before and after optimization

4. **Web Interface (`app.py` + `templates/index.html`)**
   - Start training, view metrics, and optimize the model
   - Submit text for toxicity prediction (PyTorch or ONNX)
   - Compare model sizes and inference speeds

### Performance Metrics & Evaluation
The model's performance is evaluated using multiple metrics:

- **Accuracy**: Overall prediction accuracy across all toxicity categories
- **Precision**: Ability to identify true positives accurately
- **Recall**: Ability to find all relevant instances
- **F1 Score**: Harmonic mean of precision and recall

## Usage Guide

1. **Dataset Preparation**
   - Download and place `train.csv` in the `data/` directory
   - Verify data organization before training

2. **Model Training**
   - Click "Start Training" on the web interface
   - Monitor training progress and metrics
   - View detailed performance metrics after training

3. **Model Optimization**
   - Click "Optimize Model" to prune and export to ONNX
   - Compare model sizes and inference speeds

4. **Toxicity Detection**
   - Enter text in the web interface for prediction
   - Choose between PyTorch and ONNX models for inference
   - View predicted toxicity scores for each category

## Technologies Used
- **PyTorch** (Deep Learning)
- **Flask** (Web Server)
- **Transformers** (NLP Model)
- **ONNX Runtime** (Optimized Inference)
- **NumPy, scikit-learn, pandas, tqdm** (Data Processing & Metrics)

## License
This project is licensed under the MIT License.

## Author
Developed by Varun. Feel free to connect with me:
- Email: darklususnaturae@gmail.com
