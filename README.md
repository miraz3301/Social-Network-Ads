# Social Network Ads Purchase Prediction

This project uses **Machine Learning** to predict whether a user will purchase a product based on their **Age**, **Gender**, and **Estimated Salary**.

A trained Random Forest model is deployed using **Gradio** on **Hugging Face Spaces**.
🔗 https://huggingface.co/spaces/miraz3301/Social-Network-Ads

---

## Project Features
- Data preprocessing & scaling
- Machine Learning model training (Random Forest)
- Model evaluation (Accuracy, Precision, Recall, F1 Score)
- Interactive web app using Gradio
- Deployed on Hugging Face Spaces

---

## Dataset
**Social Network Ads Dataset:**
https://www.kaggle.com/datasets/rakeshrau/social-network-ads

Features:
- Gender
- Age
- Estimated Salary  
Target:
- Purchased (0 = No, 1 = Yes)

---

## Machine Learning Workflow
1. Load and explore dataset
2. Encode categorical features
3. Normalize Age & Salary
4. Split dataset into train/test sets
5. Train Random Forest Classifier
6. Evaluate model performance
7. Save trained model
8. Deploy using Hugging Face + Gradio

---

## Live Demo
🔗 **Hugging Face App**  
https://huggingface.co/spaces/miraz3301/Social-Network-Ads

---

## How to Run Locally

### Install dependencies:
```bash
pip install -r requirements.txt
```
### Run the App
```
python app.py
```
