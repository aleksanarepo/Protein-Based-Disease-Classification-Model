# Protein-Based Disease Classification Model

A deep learning model for binary classification of MASH (Metabolic dysfunction-Associated Steatohepatitis) vs Healthy Controls based on protein expression data.

## Overview

This project implements a neural network classifier using TensorFlow/Keras to predict disease status from protein biomarkers. The model achieves high classification accuracy with proper regularization techniques.

## Requirements

```
tensorflow==2.14.0
keras==2.14.0
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## Model Architecture

- **Input Layer**: Protein expression features
- **Hidden Layers**: Dense layers with L1/L2 regularization
- **Dropout**: Applied for regularization
- **Output Layer**: Binary classification (sigmoid activation)
- **Optimizer**: Adam
- **Loss Function**: Binary crossentropy

## Features

- Data normalization and standardization
- Class imbalance handling with class weights
- Early stopping to prevent overfitting
- Model checkpointing
- Comprehensive evaluation metrics:
  - Confusion matrix
  - ROC curve and AUC
  - Classification report
  - Sensitivity and specificity

## Usage

1. Load your protein expression data (CSV format)
2. Preprocess data with normalization/scaling
3. Train the model with the provided architecture
4. Evaluate using confusion matrix and ROC curves
5. Save the trained model for inference

## Model Performance

The model demonstrates excellent classification performance with near-perfect separation between MASH and healthy control samples based on the confusion matrix results.

## File Structure

```
Protein_prediction_Model.ipynb - Main notebook with complete pipeline
```

## Notes

- Threshold for classification can be adjusted (default: 0.97)
- Model uses early stopping with patience=10
- Training includes validation split for monitoring
