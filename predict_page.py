import streamlit as st
from joblib import load
import numpy as np

def load_model():
    try:
        data = load('saved_steps.joblib')
        print("Model and preprocessing transformers loaded successfully.")
        return data
    except FileNotFoundError:
        print("Error: File 'saved_steps.joblib' not found.")
        return None
    except Exception as e:
        print(f"Error loading joblib file: {e}")
        return None

data = load_model()

if data is not None:
    regressor = data.get("model")
    le_country = data.get("le_country")
    le_education = data.get("le_education")
else:
    # Handle case where model loading failed
    regressor = None
    le_country = None
    le_education = None

def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education_levels = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education_levels)
    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        if regressor is not None and le_country is not None and le_education is not None:
            # Encode input features using LabelEncoder
            country_encoded = le_country.transform([country])[0]
            education_encoded = le_education.transform([education])[0]
            
            # Ensure experience is float type
            experience = float(experience)
            
            X = np.array([[country_encoded, education_encoded, experience]])
            salary = regressor.predict(X)
            st.subheader(f"The estimated salary is ${salary[0]:.2f}")
        else:
            st.error("Error: Model or preprocessing transformers not loaded. Cannot make prediction.")

