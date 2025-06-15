
# 🏥 Obesity Prediction ML Project

This project is an end-to-end machine learning solution to predict obesity levels based on lifestyle and biometric factors using the ObesityDataSet. It includes data preprocessing, exploratory analysis, model training, and a deployed web app using Streamlit.

---

## 📁 Dataset

**File:** `ObesityDataSet.csv`  
**Source:** Provided locally  
**Target Column:** `NObeyesdad` (Obesity level classification)

---

## 🧱 Project Structure

### 1. `Obesity_Prediction_Project.ipynb`
A modular notebook that organizes the project into the following classes:

- **DataLoader**
  - Loads the dataset
  - Displays summary statistics and data info
  - Checks missing values
  - Plots feature distributions
  - Displays class balance

- **DataPreprocessor**
  - Handles missing values
  - Encodes categorical features
  - Scales numeric features
  - Provides quick data inspection (e.g. `.head()`, `.describe()`)

### 2. `obesity_streamlit_app.py`
A user-friendly web interface built with Streamlit. It:

- Shows dataset preview, summary, and class distribution
- Lets users input their own data
- Predicts obesity level using an SVM model
- Explains each input feature with full names and tooltips

---

## 🧪 Model

- **Model used:** Support Vector Classifier (SVC)
- **Input Features:** 16
- **Target Labels:** Multiple obesity levels (e.g., Normal_Weight, Obesity_Type_I, etc.)
- **Evaluation:** Handled within notebook

---

## 🔢 Feature Descriptions

| Feature | Description |
|--------|-------------|
| **Age** | Age of the individual |
| **Gender** | Biological gender (Male/Female) |
| **Height** | Height in meters |
| **Weight** | Weight in kilograms |
| **CALC** | Frequency of alcohol consumption |
| **FAVC** | Frequent consumption of high caloric food (Yes/No) |
| **FCVC** | Frequency of vegetable consumption |
| **NCP** | Number of main meals |
| **SCC** | Consumption of food between meals (Yes/No) |
| **SMOKE** | Smoking habit (Yes/No) |
| **CH2O** | Daily water consumption (liters) |
| **family_history_with_overweight** | Family history of overweight (Yes/No) |
| **FAF** | Physical activity frequency |
| **TUE** | Time spent using technology devices |
| **NObeyesdad** | Obesity classification (target variable) |

---

## 🚀 How to Run

1. Clone the repo or copy project files
2. Install dependencies:
```bash
pip install streamlit pandas scikit-learn matplotlib seaborn
```
3. Launch the web app:
```bash
streamlit run obesity_streamlit_app.py
```

---

## 📌 Notes

- All feature labels in the app are user-friendly with clear names
- Includes basic EDA options (raw data, class distribution)
- Fully self-contained and easy to deploy

---

## 👨‍💻 Author

Project created and organized with support from ChatGPT (OpenAI) + your development and customization.

