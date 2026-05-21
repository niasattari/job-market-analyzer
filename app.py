import streamlit as st
import joblib

st.set_page_config(page_title="Job Category Predictor", page_icon="🔍")


@st.cache_resource
def load_model():
    return joblib.load('best_model.pkl')


model = load_model()

st.title("🔍 Job Category Predictor")
st.caption("Enter job details to predict the role category.")

workplace = st.selectbox("Workplace", ["Remote", "On-site", "Hybrid"])
location = st.text_input("Location", placeholder="e.g. Berlin, Munich, Remote")
department = st.text_input(
    "Department", placeholder="e.g. Engineering, Marketing")
job_type = st.selectbox(
    "Job Type", ["Full-time", "Part-time", "Contract", "Internship"])

if st.button("Predict Category", type="primary"):
    features = f"{workplace} {location} {department} {job_type}"
    prediction = model.predict([features])[0]
    proba = model.predict_proba([features])[0]
    confidence = max(proba) * 100

    st.success(f"**Predicted Category: {prediction}**")
    st.metric("Confidence", f"{confidence:.1f}%")

    st.divider()
    st.subheader("All category probabilities")
    import json
    import pandas as pd
    with open('categories.json') as f:
        cats = json.load(f)
    prob_df = pd.DataFrame({'Category': cats, 'Probability': proba})
    prob_df = prob_df.sort_values('Probability', ascending=True)
    st.bar_chart(prob_df.set_index('Category'))
