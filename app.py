import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Custom yellow background
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffe0;;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style="text-align:center; padding: 2rem 0;">
        <h1 style="color:#333366; font-size:3rem; font-family:sans-serif;">ChronicCare AI</h1>
        <p style="font-size:1.2rem; color:#444;">Empowering Chronic Kidney Disease Analysis & Prediction</p>
        <hr style="border-top: 2px solid #333366; width:60%; margin:auto;">
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Chronic Kidney Disease Interactive Dashboard")

# Load CSV
df = pd.read_csv('/Users/nivetham/Documents/Chronic/Chronic_Kidney_Dsease_data (1).csv')
st.write("Data Preview:")
st.dataframe(df.head())

# Select columns for pairplot
cols = st.multiselect("Select columns for pairplot", df.columns.tolist(), default=['Age', 'Gender', 'Ethnicity', 'SocioeconomicStatus', 'Diagnosis'])
hue_col = st.selectbox("Select hue column", df.columns.tolist(), index=df.columns.get_loc('Diagnosis') if 'Diagnosis' in df.columns else 0)

if len(cols) > 1 and hue_col in cols:
    fig = sns.pairplot(df[cols], hue=hue_col)
    st.pyplot(fig)
else:
    st.write("Please select at least two columns and include the hue column.")

# Show value counts for a selected column
selected_col = st.selectbox("Select column for value counts", df.columns.tolist())
st.write(df[selected_col].value_counts())

# Show correlation heatmap
if st.button("Show Correlation Heatmap"):
    numeric_df = df.select_dtypes(include=[np.number])
    fig2, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, ax=ax)
    st.pyplot(fig2)

# Machine Learning Section
st.header("Train Models")
target_col = st.selectbox("Select target column", df.columns.tolist(), index=df.columns.get_loc('Diagnosis') if 'Diagnosis' in df.columns else 0)
feature_cols = st.multiselect("Select feature columns", [col for col in df.columns if col != target_col], default=['Age', 'Gender', 'Ethnicity', 'SocioeconomicStatus'])

if st.button("Train Models"):
    X = df[feature_cols].values
    Y = df[target_col].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(X_train, Y_train)

    st.write(f"Logistic Regression Training Accuracy: {log.score(X_train, Y_train):.2f}")
    st.write(f"Decision Tree Training Accuracy: {tree.score(X_train, Y_train):.2f}")
    st.write(f"Random Forest Training Accuracy: {forest.score(X_train, Y_train):.2f}")

    pred = forest.predict(X_test)
    st.write("Random Forest Predictions:", pred)
    st.write("Actual Values:", Y_test)