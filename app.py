import pandas as pd
import numpy as np
import pickle
import gradio as gr

with open ("/content/social_network_rf_model.pkl","rb") as file:
  model = pickle.load(file)

df = pd.read_csv("Social_Network_Ads.csv")
age_mean = df["Age"].mean()
age_std = df["Age"].std()
salary_mean = df["EstimatedSalary"].mean()
salary_std = df["EstimatedSalary"].std()

def predict_purchase(gender, age, salary):

    gender = 1 if gender == "Male" else 0
    age_scaled = (age - age_mean) / age_std
    salary_scaled = (salary - salary_mean) / salary_std

    input_data = np.array([[gender, age_scaled, salary_scaled]])
    prediction = model.predict(input_data)[0]
    return "Will Purchase" if prediction == 1 else "Will NOT Purchase"

app = gr.Interface(
    fn=predict_purchase,
    inputs=[
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Number(label="Age"),
        gr.Number(label="Estimated Salary")
    ],
    outputs="text",
    title="Purchase Prediction App"
)

app.launch(share=True)
