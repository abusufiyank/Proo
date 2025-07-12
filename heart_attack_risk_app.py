import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set page configuration
st.set_page_config(page_title="Heart Attack Risk Predictor", layout="centered")

# Custom background and styles
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(to right, #f8f9fa, #e0f7fa);
            color: #000;
        }
        .sidebar .sidebar-content {
            background-color: #f1f1f1;
        }
        .prediction-box {
            padding: 1rem;
            margin-top: 1rem;
            background-color: #ffffff;
            border-left: 6px solid #ff6f61;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            font-size: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# Title
st.title("🫀 Heart Attack Risk Prediction App")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("Heart_Attack_Risk_Levels_Dataset.csv")
    df = df.drop(columns=['Result', 'Recommendation'])
    label_encoder = LabelEncoder()
    df['Risk_Level'] = label_encoder.fit_transform(df['Risk_Level'])
    return df, label_encoder

df, label_encoder = load_data()

# Split data
X = df.drop(columns=['Risk_Level'])
y = df['Risk_Level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train_scaled, y_train)

# Sidebar input form
st.sidebar.header("📝 Input Patient Data")

def user_input():
    age = st.sidebar.slider("Age", 20, 100, 55)
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    heart_rate = st.sidebar.slider("Heart Rate", 40, 200, 70)
    sbp = st.sidebar.slider("Systolic Blood Pressure", 90, 200, 130)
    dbp = st.sidebar.slider("Diastolic Blood Pressure", 50, 130, 85)
    blood_sugar = st.sidebar.slider("Blood Sugar (mg/dL)", 70, 400, 150)
    ck_mb = st.sidebar.slider("CK-MB (ng/mL)", 0.0, 10.0, 2.0)
    troponin = st.sidebar.slider("Troponin (ng/mL)", 0.0, 5.0, 0.01)

    gender_value = 1 if gender == "Male" else 0

    return pd.DataFrame([{
        'Age': age,
        'Gender': gender_value,
        'Heart rate': heart_rate,
        'Systolic blood pressure': sbp,
        'Diastolic blood pressure': dbp,
        'Blood sugar': blood_sugar,
        'CK-MB': ck_mb,
        'Troponin': troponin
    }])

input_df = user_input()

# Make prediction
if st.sidebar.button("Predict Risk Level"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    st.subheader("🔍 Predicted Risk Level")
    st.markdown(
        f"""
        <div class="prediction-box">
            <strong>🩺 Risk Level Prediction:</strong> <span style='color:#d32f2f; font-weight:bold;'>{predicted_label}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# Display model performance
st.subheader("📊 Model Evaluation on Test Data")

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy:** {acc:.2f}")

report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Visual: Average Risk Level by Age Group and Gender
st.subheader("📈 Risk Level Trends by Age and Gender")

# Prepare data
df_viz = df.copy()
df_viz['AgeGroup'] = pd.cut(df_viz['Age'], bins=[20, 35, 50, 65, 80, 100], 
                            labels=["20-35", "36-50", "51-65", "66-80", "81-100"])

df_viz['Gender'] = df_viz['Gender'].replace({1: 'Male', 0: 'Female'})
avg_risk = df_viz.groupby(['AgeGroup', 'Gender'], observed=True)['Risk_Level'].mean().reset_index()

fig2, ax2 = plt.subplots(figsize=(8,5))
sns.barplot(data=avg_risk, x='AgeGroup', y='Risk_Level', hue='Gender', palette='Set2', ax=ax2)
ax2.set_title('Average Risk Level by Age Group and Gender')
ax2.set_ylabel("Average Risk Level (0 = Low, 1 = High)")
st.pyplot(fig2)