# Installation and Setup Guide

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/protein-prediction-model.git
cd protein-prediction-model
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Data
Place your data file in the `data/` directory:
```
data/
└── Protein_DLModel_training_with0syntheticINCLUD.csv
```

### 5. Run the Model

**Option A: Using Jupyter Notebook**
```bash
jupyter notebook Protein_prediction_Model_cleaned.ipynb
```

**Option B: Using Python Script**
```bash
python protein_prediction_model.py
```

## System Requirements

- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended)
- CPU or GPU (GPU recommended for faster training)

## Troubleshooting

### TensorFlow Installation Issues

**For Windows with GPU:**
```bash
pip install tensorflow==2.14.0[and-cuda]
```

**For macOS (M1/M2):**
```bash
pip install tensorflow-macos==2.14.0
pip install tensorflow-metal
```

**For Linux:**
```bash
pip install tensorflow==2.14.0
```

### Common Errors

**Error: "No module named 'tensorflow'"**
- Solution: Ensure you've activated your virtual environment and run `pip install -r requirements.txt`

**Error: "File not found: data/Protein_DLModel_training_with0syntheticINCLUD.csv"**
- Solution: Make sure your CSV file is in the `data/` folder

**Error: "DLL load failed" (Windows)**
- Solution: Install Microsoft Visual C++ Redistributable

## Verifying Installation

Run this command to verify TensorFlow is installed correctly:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

Expected output: `2.14.0`

## Development Setup

If you want to contribute or modify the code:

```bash
# Install development dependencies
pip install jupyter notebook ipython

# Start Jupyter
jupyter notebook
```

## Hardware Recommendations

### Minimum:
- CPU: 4 cores
- RAM: 8GB
- Storage: 2GB free space

### Recommended:
- CPU: 8+ cores or GPU (NVIDIA with CUDA support)
- RAM: 16GB
- Storage: 5GB free space

## Support

If you encounter issues:
1. Check the [Issues](https://github.com/yourusername/protein-prediction-model/issues) page
2. Ensure all dependencies are installed correctly
3. Verify your Python version is 3.10+
