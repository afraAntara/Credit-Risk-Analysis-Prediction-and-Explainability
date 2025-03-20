# **Credit Risk Analysis, Prediction & Explainability**
🚀 **A machine learning project for predicting loan defaults with validation techniques and explainability using SHAP & LIME.**  

---

## **📌 Project Overview**
This project builds a **credit risk prediction model** using **machine learning** to classify loan applicants as **defaulters or non-defaulters**. The model is validated using **multiple validation techniques** and **interpreted using SHAP & LIME** to explain predictions.

---

## **📂 Dataset**
The dataset contains **loan applicant information** with features such as:  
- **Personal details** (Age, Income, Employment Length)  
- **Loan details** (Amount, Interest Rate, Grade, Intent)  
- **Credit history** (Default on File, Credit History Length)  
- **Credit history** (Default on File, Credit History Length)  **Loan status** (`0 = No Default, 1 = Default`)  

📌 **Dataset Source**: Public credit risk datasets from **Kaggle / Lending Club / Home Credit**.

---

## **🔍 Data Analysis**
We conducted **exploratory data analysis (EDA)** to understand the dataset structure and detect any inconsistencies.

### **✔ Steps in Data Analysis**
1. **Missing Value Handling**  
   - Filled missing values for `person_emp_length` with **median**.  
   - Replaced missing values for `loan_int_rate` with **mean**.  

2. **Outlier Removal**  
   - Removed **age > 80** (unrealistic for loan applicants).  
   - Removed **employment length > 60** (not practical).  

3. **Feature Engineering**  
   - **Loan-to-Income Ratio** = `loan_amnt / person_income`  
   - **Loan-to-Employment Length Ratio** = `person_emp_length / loan_amnt`  
   - **Interest Rate-to-Loan Ratio** = `loan_int_rate / loan_amnt`  

4. **Class Imbalance Handling**  
   - Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance default & non-default cases.  

---

## **🧪 Model Training & Validation**
To ensure **robust performance**, we trained multiple models and validated them with different techniques.

### **✔ Models Used**
- **Random Forest**  
- **XGBoost**  
- **Stacking Classifier** (combining models for better predictions)  

### **✔ Validation Techniques**
| Validation Method      | Purpose |
|------------------------|---------|
| **K-Fold Cross-Validation** | Splits data into `K` parts for multiple evaluations. |
| **Stratified K-Fold** | Ensures each fold maintains class distribution. |
| **Monte Carlo Validation** | Randomly resamples data multiple times. |
| **Bootstrapping** | Estimates model uncertainty with repeated sampling. |

📊 **Validation results are plotted for easy comparison.**  
```python
# Example visualization
methods = ["K-Fold", "Stratified K-Fold", "Monte Carlo", "Bootstrapping"]
accuracies = [k_fold_accuracy, strat_k_fold_accuracy, monte_carlo_accuracy, bootstrap_mean]

plt.figure(figsize=(10,5))
plt.bar(methods, accuracies, color='teal')
plt.xlabel("Validation Method")
plt.ylabel("Accuracy")
plt.title("Comparison of Model Validation Methods")
plt.xticks(rotation=45)
plt.show()
