# Heart-Disease-Prediction-ML
# Heart Disease Prediction using Machine Learning

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![numpy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

This project aims to develop an accurate and efficient machine learning model to predict the risk of heart disease based on patient medical history. It utilizes a dataset from Kaggle, containing 271 records and 13 attributes related to health history.

## Table of Contents

* [Dataset](#dataset)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Technologies](#technologies)
* [Model Selection and Evaluation](#model-selection-and-evaluation)
* [Contributing](#contributing)
* [License](#license)

## Dataset

The dataset is sourced from Kaggle and contains 271 records with 13 attributes related to patient health history.

## Features

* **Age:** Age of the patient.
* **BP:** Blood pressure (resting).
* **Cholesterol:** Serum cholesterol in mg/dl.
* **Max HR:** Maximum heart rate achieved.
* **ST Depression:** ST depression induced by exercise relative to rest.
* **Chest Pain:** Chest pain type (typical angina, atypical angina, non-anginal pain, asymptomatic).
* **EKG results:** Resting electrocardiographic results (normal, ST-T wave abnormality, left ventricular hypertrophy).
* **Slope of ST:** The slope of the peak exercise ST segment.
* **Thallium test:** Results of the thallium stress test.
* **Number of vessels fluoro:** Number of major vessels (0-3) colored by fluoroscopy.
* **Sex:** Patient's sex (1 = male; 0 = female).
* **Exercise Angina:** Exercise-induced angina (1 = yes; 0 = no).
* **FBS over 120:** Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).
* **Target:** 1 = heart disease; 0 = no heart disease.

## Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/yourusername/heart-disease-prediction.git](https://www.google.com/search?q=https://github.com/yourusername/heart-disease-prediction.git)
    cd heart-disease-prediction
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  Install the required dependencies:

    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

4. Place the dataset file into the root directory of the project.

## Usage

1.  Run the Jupyter Notebook or Python script:

    ```bash
    jupyter notebook heart_disease_prediction.ipynb #or
    python heart_disease_prediction.py
    ```

2.  The notebook/script will:

    * Load and preprocess the dataset.
    * Train a machine learning model.
    * Evaluate the model's performance.
    * Display the results.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Model Selection and Evaluation

The script explores various machine learning models, including:

* Logistic Regression
* Random Forest
* Support Vector Machines (SVM)
* K-Nearest Neighbors (KNN)
* Naive Bayes 
* Decision Trees
* XGBoost
  

The model is evaluated using metrics such as:

* Accuracy
* Precision
* Recall
* F1-score


## Results

Based on the evaluation, the **XGBoost** model demonstrated the best performance, achieving an accuracy of over 80%. Random forest and decision tree also performed well.

* **XGBoost:** Accuracy > 87%
* **Random Forest:** Accuracy > 81%
* **Decision Tree:** Accuracy > 83%

## Contributing

Contributions are welcome! Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature`).
3.  Make your changes.
4.  Commit your changes (`git commit -am 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature`).
6.  Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
