
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("C:/Users/MOUKTHIKA/OneDrive/Desktop/adult.csv")

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace('-', '_').str.replace('.', '_')

    # Handle missing values
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    # Simplify column names for use
    df['native_country'] = df['native_country'].replace({'?': 'Others'})

    # Label Encoding for categorical columns
    categorical_cols = df.select_dtypes(include='object').columns
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    return df, encoder

def main():
    st.title("Income Prediction App (>50K or <=50K)")
    st.markdown("Upload your data or use the default dataset")

    df, encoder = load_and_prepare_data()
    st.write("Dataset Preview", df.head())

    X = df.drop('income', axis=1)
    y = df['income']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_choice = st.selectbox("Choose a model", ["Logistic Regression", "Random Forest"])

    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"Model trained. Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
