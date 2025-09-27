# Home Credit Default Prediction

## 📌 Project Overview

This project focuses on predicting whether a loan applicant will be able to repay a loan or not, using data provided by **Home Credit Group**. The goal is to build a robust machine learning pipeline that improves risk assessment and helps financial institutions make better lending decisions.

The dataset contains information about clients, their loan applications, previous loans, and credit bureau history. By analyzing these features, we aim to predict the probability of default.

---

## 🎯 Objectives

* Perform **data cleaning & preprocessing** (handling missing values, encoding, feature engineering).
* Conduct **exploratory data analysis (EDA)** to understand patterns.
* Build machine learning models to predict loan repayment probability.
* Evaluate models using metrics such as **AUC-ROC, Precision, Recall, and F1-score**.
* Deploy the final model for inference.

---

## 📂 Project Structure

```
Home_Credit_Prediction/
│── data/                 # Raw and processed datasets
│── notebooks/            # Jupyter notebooks for EDA & experiments
│── src/                  # Source code (data processing, utils, models)
│   ├── utils/            # Helper functions
│   ├── components/       # Data transformation, model trainer
│   ├── pipelines/        # Training and prediction pipelines
│── artifacts/            # Saved models, preprocessors, and outputs
│── test/                 # Unit tests
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
│── app.py                # Streamlit / Flask app for model inference
```

---

## ⚙️ Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/home-credit-prediction.git
cd home-credit-prediction
```

2. Create a virtual environment:

```bash
python -m venv myenv
myenv\Scripts\activate   # On Windows
source myenv/bin/activate # On Mac/Linux
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

* **Source**: [Kaggle - Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
* **Size**: ~300 MB across multiple CSV files
* **Key Files**:

  * `application_train.csv` → Client & loan application info (training data)
  * `application_test.csv` → Test data (without target)
  * `bureau.csv` → Credit history from other institutions
  * `previous_application.csv` → Previous loans from Home Credit
  * `POS_CASH_balance.csv`, `credit_card_balance.csv`, `installments_payments.csv` → Repayment history

---

## 🛠️ Methodology

1. **Data Preprocessing**

   * Handle missing values
   * Encode categorical variables
   * Feature scaling & transformation

2. **Exploratory Data Analysis (EDA)**

   * Visualize distributions & correlations
   * Identify key risk indicators

3. **Feature Engineering**

   * Aggregation of credit bureau & repayment history
   * Domain-specific features (ratios, flags, averages)

4. **Modeling**

   * Logistic Regression
   * Random Forest
   * Gradient Boosting (XGBoost, LightGBM, CatBoost)
   * Neural Networks (optional)

5. **Model Evaluation**

   * Cross-validation
   * ROC-AUC, Precision, Recall, F1-score

6. **Deployment**

   * Save best model using `pickle/joblib`
   * Build prediction API (Flask/Streamlit)

---

## 📈 Results

* Best performing model: **LightGBM**
* Achieved **ROC-AUC: ~0.78–0.80** on validation data
* Feature importance shows **income, credit history, and repayment behavior** as key predictors

---

## 🚀 Usage

Run training pipeline:

```bash
python src/pipelines/training_pipeline.py
```

Run prediction pipeline:

```bash
python src/pipelines/prediction_pipeline.py --input sample_input.csv
```

Start Streamlit app:

```bash
streamlit run app.py
```

---

## 🧑‍💻 Tech Stack

* **Languages**: Python
* **Libraries**: Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, Matplotlib, Seaborn
* **Deployment**: Flask / Streamlit
* **Version Control**: Git & GitHub

---

## 📌 Future Improvements

* Hyperparameter optimization with Optuna
* Use of deep learning models for feature extraction
* Model interpretability with SHAP / LIME
* Dockerization for production deployment

---

## 🙌 Acknowledgements

* [Kaggle Home Credit Competition](https://www.kaggle.com/competitions/home-credit-default-risk)
* Open-source community for datasset
  
