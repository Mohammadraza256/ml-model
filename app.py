import streamlit as st
import numpy as np
import pickle

# Title
st.title("🌸 Iris Flower Species Predictor")
st.write("Enter flower measurements below to predict the species:")

# Inputs
sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    species = ["setosa", "versicolor", "virginica"][prediction]
    st.success(f"The predicted species is: **{species.capitalize()}**")
