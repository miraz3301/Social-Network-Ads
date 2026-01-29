import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pickle
import joblib
import gradio as gr

# 1. Data Loading

df = pd.read_csv('/content/Social_Network_Ads.csv')
print("Shape: ", df.shape)
df.head()

# 2. Data Preprocessing

 # i.
if 'User ID' in df.columns:
    df = df.drop('User ID', axis=1)

 # ii.
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})

 # iii.
if df.isnull().sum().sum() > 0:
    df.fillna(df.mean(), inplace=True)
else:
    print("No missing values found.")

 # iv.
numerical_cols = ['Age', 'EstimatedSalary']
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Outliers detected in {col}: {len(outliers)}")
 # v.
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("\nProcessed Data Head:")
df.head()

# 3. Pipeline Creation

X = df.drop(columns=['Purchased'])
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('model', RandomForestClassifier(random_state=42))
])

# 4. Primary Model Selection:

  #  I have selected Random Forest algorithm. Cause:
  #  a. It handles non-linear data.
  #  b. It reduces overfitting.
  #  c. It high accuracy for tabular classification
  #  d. It works well with scaled features

# 5. Model Training

pipeline.fit(X_train, y_train)

# 6. Cross-Validation

scores = cross_val_score(pipeline, X, y, cv=5)
print("Mean Accuracy:", scores.mean())
print("Standard Deviation:", scores.std())

# 7. Hyperparameter Tuning

params = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 5, 10]
}
gd = GridSearchCV(pipeline, params, cv=5)
gd.fit(X_train, y_train)
print("Best Params:", gd.best_params_)
print("Best Score:", gd.best_score_)

# 8. Best Model Selection

best_model = gd.best_estimator_

# 9. Model Performance Evaluation

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save trained model
with open("social_network_rf_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# 10. Web Interface with Gradio

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

app.launch()