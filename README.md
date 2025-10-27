# 🫀 Heart Attack Prediction using Machine Learning

A comprehensive machine learning project that predicts the risk of heart attacks using patient clinical data. This project implements multiple ML algorithms and provides interactive visualizations through Power BI.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Results & Insights](#results--insights)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)

## 🎯 Overview

Heart disease is one of the leading causes of death globally. Early prediction and diagnosis can save lives. This project uses machine learning algorithms to predict the likelihood of a heart attack based on various patient health metrics such as age, cholesterol levels, blood pressure, and other clinical parameters.

## ✨ Features

- **Data Preprocessing**: Handled missing values, outliers, and data normalization
- **Exploratory Data Analysis (EDA)**: Comprehensive analysis with visualizations
- **Feature Engineering**: Created new features and selected the most relevant predictors
- **Multiple ML Models**: 
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier
- **Model Evaluation**: Used accuracy, precision, recall, F1-score, and AUC-ROC metrics
- **Interactive Visualizations**: Power BI dashboard for risk distribution analysis

## 📊 Dataset

The dataset contains patient records with the following features:

- **Age**: Age of the patient
- **Sex**: Gender (Male/Female)
- **Chest Pain Type**: Type of chest pain experienced
- **Resting Blood Pressure**: Blood pressure in mm Hg
- **Cholesterol**: Serum cholesterol in mg/dl
- **Fasting Blood Sugar**: Blood sugar level > 120 mg/dl
- **Resting ECG**: Electrocardiographic results
- **Max Heart Rate**: Maximum heart rate achieved
- **Exercise Induced Angina**: Exercise-induced angina (Yes/No)
- **ST Depression**: Depression induced by exercise relative to rest
- **Slope**: Slope of peak exercise ST segment
- **Number of Major Vessels**: Vessels colored by fluoroscopy (0-3)
- **Thalassemia**: Blood disorder type
- **Target**: Heart attack risk (0 = Low Risk, 1 = High Risk)

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `matplotlib` & `seaborn` - Data visualization
  - `scikit-learn` - Machine learning models and preprocessing
  - `xgboost` - Gradient boosting framework
  - `Power BI` - Interactive dashboards and visualizations

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/heart-attack-prediction.git
cd heart-attack-prediction
```

2. **Create a virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. **Run the Jupyter Notebook**
```bash
jupyter notebook heart_attack_prediction.ipynb
```

2. **Or run the Python script**
```bash
python heart_attack_prediction.py
```

3. **View Power BI Dashboard**
   - Open the `heart_attack_dashboard.pbix` file in Power BI Desktop

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 85.2% | 0.83 | 0.87 | 0.85 | 0.90 |
| Random Forest | 88.7% | 0.86 | 0.89 | 0.87 | 0.93 |
| **XGBoost** | **92.1%** | **0.91** | **0.93** | **0.92** | **0.95** |

**XGBoost emerged as the best performing model** with the highest accuracy and AUC-ROC score.

## 🔍 Results & Insights

### Key Findings:
1. **Most Important Predictors**:
   - Cholesterol levels
   - Age
   - Maximum heart rate
   - Blood pressure
   - Chest pain type

2. **Risk Distribution**:
   - Males showed higher risk compared to females
   - Risk increases significantly after age 50
   - Patients with Type 2 chest pain had the highest risk

3. **Model Insights**:
   - XGBoost handled non-linear relationships better
   - Feature engineering improved model performance by 7%
   - Cross-validation confirmed model stability

## 📁 Project Structure

```
heart-attack-prediction/
│
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned and preprocessed data
│
├── notebooks/
│   └── heart_attack_prediction.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── visualization.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost_model.pkl
│
├── visualizations/
│   ├── eda_plots/
│   └── heart_attack_dashboard.pbix
│
├── requirements.txt
├── README.md
└── LICENSE
```

## 🔮 Future Enhancements

- [ ] Deploy model as a web application using Flask/Streamlit
- [ ] Implement deep learning models (Neural Networks)
- [ ] Add real-time prediction capability
- [ ] Integrate with healthcare APIs
- [ ] Implement SHAP values for model interpretability
- [ ] Add support for more clinical parameters
- [ ] Create mobile application interface

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👤 Author

**Your Name**
- GitHub: [@AnshulBhardwaj](https://github.com/anshulbhardwaj123)
- LinkedIn: [AnshulBhardwaj](https://www.linkedin.com/in/anshulbhardwaj1)
- Email: anshul123.124@gmail.com

## 🙏 Acknowledgments

- Dataset source: [UCI Machine Learning Repository / Kaggle]
- Inspiration from healthcare ML research papers
- Thanks to the open-source community

---

⭐ If you found this project helpful, please consider giving it a star!

**Note**: This model is for educational purposes only and should not be used as a substitute for professional medical advice.
