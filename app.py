import streamlit as st
import joblib

# load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Email Phishing Detection App")

st.write("Enter email text to check if it is phishing or legitimate.")

email = st.text_area("Enter Email Text")

if st.button("Detect"):

    email_vector = vectorizer.transform([email])
    prediction = model.predict(email_vector)

    if prediction[0] == 1:
        st.error("⚠️ Phishing Email Detected")
    else:
        st.success("✅ Legitimate Email")