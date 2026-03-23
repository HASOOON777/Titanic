🚢 Titanic Survival Predictor & Analytics Dashboard

📌 Overview

This project is an interactive web application built with Streamlit and Scikit-Learn. It uses a Machine Learning model (Logistic Regression) trained on the famous Titanic dataset to predict the survival probability of a passenger based on their personal information and ticket details.

The app also features an interactive analytics dashboard to explore the historical data and model performance.

🎥 App Demo

(Replace this with your actual GIF file if you named it differently)

🚀 Features

Real-time Predictions: Input passenger features (Age, Class, Gender, Fare, etc.) and instantly get survival probabilities.

Interactive Analytics: View dynamic charts (Survival by Gender, Age Distribution, Ticket Class) using Matplotlib and Seaborn.

Model Evaluation: Visualizations of the Confusion Matrix and Missing Values Heatmap.

Optimized Preprocessing: Handled missing data via Imputation, encoded categorical variables, and scaled numerical features.

🛠️ Tech Stack

Data Manipulation: pandas, numpy

Machine Learning: scikit-learn (Logistic Regression, SimpleImputer, StandardScaler, LabelEncoder)

Web App UI: streamlit

Data Visualization: matplotlib, seaborn

📂 Project Structure

titanic_app.py: The main Streamlit application script.

titanic_project.py: The script used for data preprocessing and model training.

train.csv: The raw dataset.

*.pkl: Serialized machine learning models, scalers, and encoders.

*.png: Saved plots (Confusion Matrix, Heatmap).

requirements.txt: List of dependencies required to run the app.

💻 How to Run Locally

Clone this repository:

git clone [https://github.com/HASOOON777/titanic-survival-predictor.git](https://github.com/HASOOON777/titanic-survival-predictor.git)
cd titanic-survival-predictor


Install the required dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run titanic_app.py


👨‍💻 Author

Hassoooon AI 
LinkedIn | GitHub