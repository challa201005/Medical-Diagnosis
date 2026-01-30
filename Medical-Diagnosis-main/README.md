# Medical Diagnosis Project

A comprehensive machine learning project for medical diagnosis using deep learning and traditional ML techniques. This repository contains implementations for diagnosing various medical conditions including breast cancer, diabetes, and heart disease.

## 📁 Project Structure

```
medical-diagnosis-project/
├── breast/                 # Breast cancer diagnosis
│   ├── notebook.ipynb     # Jupyter notebook with implementation
│   └── dataset.csv        # Breast cancer dataset
├── diabetes/              # Diabetes prediction
│   ├── notebook.ipynb     # Jupyter notebook with implementation
│   └── dataset.csv        # Diabetes dataset
├── heart_disease/         # Heart disease prediction
│   ├── notebook.ipynb     # Jupyter notebook with implementation
│   └── dataset.csv        # Heart disease dataset
├── loss_plots/            # Training visualization
│   ├── breast_loss.png    # Loss plots for breast cancer model
│   ├── diabetes_loss.png  # Loss plots for diabetes model
│   └── heart_loss.png     # Loss plots for heart disease model
└── README.md             # Project documentation
```

---
## 📌 Features
- **Jupyter Notebooks** for each disease
- **Supervised Learning Models**: Logistic Regression, Decision Trees, Neural Networks, etc.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Loss Plot Visualization** for model training performance
- Modular structure for easy extension and readability

---
## 📚 Technologies Used
- [Python 3.x](https://www.python.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [TensorFlow](https://www.tensorflow.org/)

---
## 📊 Datasets
Each notebook includes:
- Data loading and preprocessing
- Feature analysis and selection
- Model training and testing
- Performance visualization

Datasets are stored alongside each notebook and are publicly available or simulated for academic use.

---
## 📈 Loss Plots
All training loss plots are saved in the `loss_plots/` directory to visually inspect the training progress of neural network models.

---
## 🚀 Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/KALYANRAM-005/Medical-Diagnosis.git
   cd Medical-Diagnosis
   ```

2. Install required dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn tensorflow jupyter
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open any notebook from the disease folders and run the cells

---
## 🔬 Medical Conditions Covered

### 1. Breast Cancer Diagnosis
- **Objective**: Binary classification to detect malignant vs benign tumors
- **Features**: Cell nucleus characteristics (radius, texture, perimeter, area, etc.)
- **Approach**: Deep learning with TensorFlow/Keras

### 2. Diabetes Prediction
- **Objective**: Predict diabetes onset in patients
- **Features**: Medical indicators like glucose levels, blood pressure, BMI
- **Approach**: Machine learning classification

### 3. Heart Disease Prediction
- **Objective**: Assess risk of heart disease
- **Features**: Cardiovascular risk factors and clinical measurements
- **Approach**: Ensemble methods and neural networks

---
## 🎯 Model Evaluation Metrics

Each model is evaluated using:
- **Accuracy**: Overall classification performance
- **Precision**: True positive rate
- **Recall**: Sensitivity to positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed classification results

---
## ⚠️ Disclaimer

This project is for educational purposes only and should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice.

---

**Note**: All models show training progress in the loss plots directory, demonstrating convergence and helping identify potential overfitting or underfitting issues.
