import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Prediction App")

# User input
age = st.slider("Age", 0, 100, 30)
fare = st.number_input("Fare Paid", 0.0, 600.0, 32.0)
sex = st.selectbox("Sex", ["Male", "Female"])
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Process inputs
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
sex = 1 if sex == "Male" else 0
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_map[embarked]

features = np.array([[age, fare, sex, pclass, sibsp, parch, family_size, is_alone, embarked]])

if st.button("Predict Survival"):
    pred = model.predict(features)
    if pred[0] == 1:
        st.success("🎉 Passenger would have survived!")
    else:
        st.error("❌ Passenger would not have survived.")
