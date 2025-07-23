import pandas as pd
import numpy as np
import streamlit as st
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load data
data = pd.read_csv(r"C:\Users\MOUKTHIKA\OneDrive\Desktop\adult.csv")

# Optional: print columns for debugging
print("Columns in dataset:", data.columns.tolist())

# Clean missing values
data['workclass'] = data['workclass'].replace({'?': 'Others'})
data['occupation'] = data['occupation'].replace({'?': 'Others'})
data['native-country'] = data['native-country'].replace({'?': 'Others'})

# Drop unnecessary columns
data = data.drop(['fnlwgt', 'education.num', 'capital.gain', 'capital.loss'], axis=1)

# Encode categorical features
le = LabelEncoder()
for column in data.select_dtypes(include='object').columns:
    data[column] = le.fit_transform(data[column])

# Split data
X = data.drop('income', axis=1)
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# Best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n✅ Best model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# Save best model
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# --- Streamlit App ---
st.title("Income Classification App")

age = st.slider("Age", 17, 90)
hours_per_week = st.slider("Hours per Week", 1, 99)
education = st.selectbox("Education", data['education'].unique())
occupation = st.selectbox("Occupation", data['occupation'].unique())
relationship = st.selectbox("Relationship", data['relationship'].unique())
sex = st.selectbox("Sex", data['sex'].unique())
race = st.selectbox("Race", data['race'].unique())
workclass = st.selectbox("Workclass", data['workclass'].unique())
native_country = st.selectbox("Native Country", data['native-country'].unique())
marital_status = st.selectbox("Marital Status", data['marital.status'].unique())

# Create input data
input_data = pd.DataFrame([[
    age,
    le.transform([workclass])[0],
    le.transform([education])[0],
    le.transform([marital_status])[0],
    le.transform([occupation])[0],
    le.transform([relationship])[0],
    le.transform([race])[0],
    le.transform([sex])[0],
    hours_per_week,
    le.transform([native_country])[0],
]])

# Load model and predict
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

st.subheader("Prediction Result:")
st.write("Income >50K" if prediction[0] == 1 else "Income <=50K")
