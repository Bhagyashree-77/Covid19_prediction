# COVID-19 Death Prediction Model

This project focuses on creating a machine learning model to predict the death situation of COVID-19 patients based on various health and demographic features. The dataset is preprocessed to handle missing values and outliers, and Logistic Regression is used to train the model.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Requirements](#requirements)
- [File Descriptions](#file-descriptions)
- [Acknowledgments](#acknowledgments)

## Overview
This notebook implements:
- Data cleaning and preprocessing.
- Exploratory data analysis (EDA) with visualizations.
- Handling of missing values and outliers.
- Logistic Regression model training and evaluation.
- ROC Curve for model performance.
- Saving and loading the trained model using `pickle`.

## Dataset
- **Source**: `Covid Dataaa.csv`
- **Shape**: 1,048,575 rows and 21 columns

### Columns Overview
- **USMER, MEDICAL_UNIT, SEX, PATIENT_TYPE**: Categorical demographic and medical features.
- **DATE_DIED**: Date of death ("9999-99-99" represents alive patients).
- **INTUBED, PNEUMONIA, AGE, PREGNANT, DIABETES, etc.**: Various health conditions and risk factors.
- **CLASIFFICATION_FINAL, ICU**: Final classification and ICU admission status.

## Data Preprocessing
1. **Missing Values Handling**:
   - Features like `PNEUMONIA`, `DIABETES`, etc., had invalid values (e.g., `99`) treated as NaN and removed.
   - `DATE_DIED` was converted into a binary feature `DEATH`.

2. **Feature Encoding**:
   - `DEATH`: 1 for deceased patients, 2 for alive.

3. **Visualization**:
   - Count plots for features like `SEX` and `PREGNANT`.

## Model Training
### Logistic Regression
- **Model**: Logistic Regression
- **Split**: Data was split into training (80%) and testing (20%).
- **Accuracy**: Evaluated on the test set.
- **Confusion Matrix**: Visualized for better insight into predictions.

## Evaluation
- **Metrics**:
  - Accuracy: Computed using `logreg.score`.
  - F1 Score: Computed for the test set.
  - ROC Curve: Plotted to evaluate the model's ability to classify outcomes.

## Usage
1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook or script to preprocess the dataset and train the model.
4. Use the trained model for predictions using the provided input format.

### Example
```python
input_data = (2,1,2,1,"21/6/2020",97,2,68,97,1,2,2,2,1,2,2,2,2,2,3,97)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = logreg.predict(input_data_reshaped)
if prediction[0] == 0:
    print("The person is detected COVID positive")
else:
    print("The person is detected COVID negative")
```

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies with:
```bash
pip install -r requirements.txt
```

## File Descriptions
- `Covid Dataaa.csv`: Raw dataset.
- `notebook.ipynb`: Jupyter notebook containing the code for preprocessing, training, and evaluation.
- `trained_model.sav`: Serialized trained Logistic Regression model.
- `requirements.txt`: List of required libraries.

## Acknowledgments
- Thanks to the contributors of the dataset for enabling this analysis.
- Libraries like pandas, numpy, matplotlib, and scikit-learn were crucial for this project.

