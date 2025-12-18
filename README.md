# MASH Protein Biomarker Prediction Model

A machine learning approach for distinguishing MASH (Metabolic Dysfunction-Associated Steatohepatitis) from healthy controls using 19 protein biomarkers.

## Overview

This repository contains a neural network-based classification model that achieves 97% accuracy in distinguishing MASH patients from healthy controls using a panel of 19 protein biomarkers. The model demonstrates the feasibility of protein-based diagnostic approaches for metabolic liver disease.

## Key Features

- **High Accuracy**: 97% classification accuracy on independent test dataset
- **19 Protein Biomarkers**: Comprehensive panel including acute phase proteins, liver function markers, and inflammatory cytokines
- **Multiple ML Approaches**: Implementation of neural networks, Ridge regression, and k-NN for comparison
- **Bootstrap Validation**: Statistical confidence intervals for performance metrics
- **SHAP Analysis**: Feature importance and model interpretability

## Dataset

- **Training Set**: 30 samples (augmented for healthy controls)
- **Independent Test Set**: 36 samples
- **Features**: 19 protein biomarkers
- **Classes**: Binary classification (MASH vs Healthy Controls)

### Protein Biomarkers

| Category | Proteins |
|----------|----------|
| Acute Phase Proteins | SAA1, SERPINA3 |
| Liver Function | ARG1, GSTA2, FABPA |
| Metabolic Markers | IGFbp, PPBP |
| Inflammatory Cytokines | IL27RA, OSMR |
| Other | HP, HPX, SERPINA7, PTPA, CLC1B, FRIH, TIMP2, FYN, S100A4, ANKRD |

## Model Architecture

- **Input Layer**: 19 features (protein biomarkers)
- **Hidden Layer**: 7 neurons with ReLU activation
- **Output Layer**: 2 neurons with softmax activation (binary classification)
- **Optimizer**: Adam
- **Loss Function**: Categorical cross-entropy

## Results

### Performance Metrics (Threshold = 0.97)
- **Accuracy**: 97.3%
- **Sensitivity**: 100%
- **Specificity**: 94.7%
- **Precision**: 94.7%
- **F1 Score**: 97.3%

### Confusion Matrix
```
                Predicted
              MASH    HC
Actual MASH    18     0
       HC       1    18
```

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.8+
- TensorFlow 2.14.0
- Keras 2.14.0
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- SHAP

## Usage

### Training the Model

```python
# Run the Jupyter notebook
jupyter notebook Protein_prediction_Model.ipynb
```

### Making Predictions

```python
from keras.models import load_model
import numpy as np

# Load trained model
model = load_model('mash_model.h5')

# Prepare your data (19 protein values)
sample = np.array([[...]])  # Your 19 protein measurements

# Make prediction
prediction = model.predict(sample)
print(f"MASH probability: {prediction[0][1]:.4f}")
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── Protein_prediction_Model.ipynb
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   └── mash_model.h5
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── shap_analysis.png
└── docs/
    └── methodology.md
```

## Methodology

1. **Data Preprocessing**: Feature scaling and normalization
2. **Feature Selection**: Ridge regression to identify key biomarkers
3. **Model Training**: Neural network with 20,000 epochs (converges ~2,000 epochs)
4. **Validation**: Independent test set (no data leakage)
5. **Bootstrap Analysis**: 1,000 iterations for confidence intervals
6. **Interpretability**: SHAP values for feature importance

## Limitations

- **Small Sample Size**: Limited to 30 training and 36 test samples
- **Augmented Data**: Healthy controls augmented due to limited availability
- **Single Population**: Validation needed across diverse demographics
- **Exploratory Study**: Proof-of-concept requiring larger validation cohorts

## Future Work

- External validation on independent cohorts
- Integration with clinical workflows
- Cost-effectiveness analysis
- Prospective clinical trials
- Regulatory approval pathway

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mash_protein_prediction,
  title={MASH Protein Biomarker Prediction Model},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/mash-protein-prediction}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Data collection and clinical validation team
- Proteomics facility for biomarker measurements
- Research funding sources

## Contact

For questions or collaborations:
- Email: your.email@institution.edu
- GitHub: [@yourusername](https://github.com/yourusername)

## Disclaimer

This model is for research purposes only and should not be used for clinical diagnosis without proper validation and regulatory approval.
