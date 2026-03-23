import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

# Custom Title
st.title("🚢 Titanic Survival Predictor & Analytics")
st.write("Welcome! This app uses a Machine Learning model (**Logistic Regression**) to predict whether a passenger would have survived the Titanic disaster.")

# Load saved models and preprocessors
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('logistic_model.pkl')
        scaler = joblib.load('scaler.pkl')
        sex_encoder = joblib.load('sex_encoder.pkl')
        feature_cols = joblib.load('feature_columns.pkl')
        return model, scaler, sex_encoder, feature_cols
    except FileNotFoundError:
        return None, None, None, None

# Load raw data for visualizations
@st.cache_data
def load_raw_data():
    try:
        return pd.read_csv('train.csv')
    except FileNotFoundError:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        return pd.read_csv(url)

model, scaler, sex_encoder, feature_cols = load_assets()
df_raw = load_raw_data()

if model is None:
    st.error("⚠️ Saved model files (.pkl) not found! Please run the training script first.")
else:
    # Sidebar for user inputs
    st.sidebar.header("Passenger Features")
    
    pclass = st.sidebar.selectbox("Ticket Class (Pclass)", [1, 2, 3], help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class")
    sex = st.sidebar.selectbox("Sex", ["male", "female"])
    age = st.sidebar.slider("Age", 0.0, 100.0, 25.0, help="Age in years")
    sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.sidebar.number_input("Parents/Children Aboard", 0, 10, 0)
    fare = st.sidebar.slider("Passenger Fare ($)", 0.0, 500.0, 32.0)
    embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"], help="C = Cherbourg, Q = Queenstown, S = Southampton")
    
    st.markdown("---")
    
    # Create Tabs for better UI organization (Swapped order)
    tab1, tab2 = st.tabs(["📊 Data & Model Analytics", "🔮 Make a Prediction"])
    
    # ==========================================
    # TAB 1: Visualizations (Plots) - Now it is the first tab
    # ==========================================
    with tab1:
        st.subheader("Interactive Data Exploration")
        st.write("Let's explore the actual historical data of the Titanic passengers:")
        
        # Row 1 of charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Survival by Gender**")
            fig_sex, ax_sex = plt.subplots(figsize=(5, 4))
            sns.countplot(data=df_raw, x='Sex', hue='Survived', palette='Set2', ax=ax_sex)
            ax_sex.set_xticklabels(['Male', 'Female'])
            ax_sex.legend(title='Survived', labels=['No', 'Yes'])
            st.pyplot(fig_sex)
            
        with col2:
            st.write("**Survival by Ticket Class**")
            fig_pclass, ax_pclass = plt.subplots(figsize=(5, 4))
            sns.countplot(data=df_raw, x='Pclass', hue='Survived', palette='Set1', ax=ax_pclass)
            ax_pclass.legend(title='Survived', labels=['No', 'Yes'])
            st.pyplot(fig_pclass)
            
        st.markdown("---")
        
        # Row 2 of charts
        st.write("**Age Distribution of Passengers**")
        fig_age, ax_age = plt.subplots(figsize=(10, 4))
        sns.histplot(data=df_raw, x='Age', hue='Survived', kde=True, palette='coolwarm', multiple="stack", ax=ax_age)
        ax_age.legend(title='Survived', labels=['Yes', 'No']) # reversed for histplot stack order
        st.pyplot(fig_age)

        st.markdown("---")
        st.subheader("Model Static Performance Plots")
        
        col3, col4 = st.columns(2)
        with col3:
            # Display Confusion Matrix
            if os.path.exists("confusion_matrix.png"):
                st.image("confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
            else:
                st.warning("⚠️ Confusion matrix image not found.")
                
        with col4:
            # Display Heatmap
            if os.path.exists("missing_values_heatmap.png"):
                st.image("missing_values_heatmap.png", caption="Missing Values Heatmap (Before Cleaning)", use_container_width=True)
            else:
                st.warning("⚠️ Heatmap image not found.")

    # ==========================================
    # TAB 2: Prediction Logic - Now it is the second tab
    # ==========================================
    with tab2:
        st.subheader("Ready to see the prediction?")
        
        if st.button("Predict Survival", type="primary"):
            # 1. Feature Engineering (FamilySize)
            family_size = sibsp + parch + 1
            
            # 2. Encoding categorical data (Sex)
            sex_encoded = sex_encoder.transform([sex])[0]
            
            # 3. Encoding categorical data (Embarked - Dummy Variables)
            embarked_Q = 1 if embarked == "Q" else 0
            embarked_S = 1 if embarked == "S" else 0
            
            # 4. Constructing the final input dictionary
            input_data = {
                'Pclass': pclass,
                'Sex': sex_encoded,
                'Age': age,
                'SibSp': sibsp,
                'Parch': parch,
                'Fare': fare,
                'FamilySize': family_size,
                'Embarked_Q': embarked_Q,
                'Embarked_S': embarked_S
            }
            
            # Convert to DataFrame ensuring the exact column order as during training
            input_df = pd.DataFrame([input_data])[feature_cols]
            
            # 5. Feature Scaling
            input_scaled = scaler.transform(input_df)
            
            # 6. Prediction
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0]
            
            survival_prob = probability[1] * 100
            death_prob = probability[0] * 100
            
            # 7. Display Results
            if prediction[0] == 1:
                st.success(f"🎉 **Prediction: SURVIVED!**")
                st.info(f"The model is **{survival_prob:.2f}%** confident that this passenger would survive.")
                st.balloons()
            else:
                st.error(f"💀 **Prediction: DID NOT SURVIVE.**")
                st.warning(f"The model is **{death_prob:.2f}%** confident that this passenger would NOT survive.")

            
    st.markdown("---")
    st.caption("Developed with ❤️ using Streamlit & Scikit-Learn")
    # streamlit run titanic_app.py