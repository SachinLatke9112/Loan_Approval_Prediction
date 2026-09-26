# Loan Approval Prediction

This project is a Machine Learning approach to predicting whether a loan will be approved or not based on an applicant's profile. It uses various applicant details, such as their gender, marital status, income, and credit history, to make the prediction.

## Overview

The repository contains a Jupyter Notebook (`Loan_Approval_Pridiction.ipynb`) that outlines the entire data science pipeline from data loading and cleaning to exploratory data analysis (EDA), feature engineering, and model training.

## Dataset

The analysis is performed on a loan dataset (`Copy of loan.csv`). The dataset includes 13 variables:
- `Loan_ID`
- `Gender`
- `Married`
- `Dependents`
- `Education`
- `Self_Employed`
- `ApplicantIncome`
- `CoapplicantIncome`
- `LoanAmount`
- `Loan_Amount_Term`
- `Credit_History`
- `Property_Area`
- `Loan_Status` (Target Variable)

## Key Steps in the Notebook

1. **Data Preprocessing & Cleaning:**
   - Filled missing values for categorical features (`Gender`, `Married`, `Dependents`, `Self_Employed`, `Loan_Amount_Term`, `Credit_History`) with their mode.
   - Filled missing values for continuous features (`LoanAmount`, `loanAmount_log`) with their mean.

2. **Feature Engineering:**
   - Log transformations applied to `LoanAmount` to normalize the distribution (`loanAmount_log`).
   - Combined `ApplicantIncome` and `CoapplicantIncome` into `TotalIncome`, and applied log transformation (`TotalIncome_log`).

3. **Exploratory Data Analysis (EDA):**
   - Histograms to view data distributions.
   - Count plots using Seaborn to analyze the relationship between target class and categorical attributes.

4. **Data Encoding & Scaling:**
   - Handled categorical data using `LabelEncoder`.
   - Scaled the final feature set using `StandardScaler`.

5. **Machine Learning Models:**
   - Data is split into 80% training and 20% testing sets.
   - The following classification models were evaluated using the Accuracy metric:
     - Random Forest Classifier
     - Gaussian Naive Bayes
     - Decision Tree Classifier
     - K-Nearest Neighbors (KNN)

## Requirements

To run this notebook, you will need to install the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install them using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to Run

1. Make sure you have the `Copy of loan.csv` dataset in the same directory as the Jupyter notebook.
2. Open the notebook using Jupyter:
   ```bash
   jupyter notebook Loan_Approval_Pridiction.ipynb
   ```
3. Run all the cells in the notebook to view the data analysis, visualizations, and model accuracies.
