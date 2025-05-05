import streamlit as st
import pandas as pd
import joblib
import json
import re

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain

# ==== GOOGLE GEMINI CONFIG ====
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyDTjCC5GTBSS5MXWJzYzoPueYcmcv58Wqw"

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
template = """
Extract the following details from the sentence and return in JSON format:
- Gender (Male/Female)
- Age (Integer)
- Tenure (Integer)
- Balance (Float)
- NumOfProducts (Integer)
- HasCrCard (0 or 1)
- IsActiveMember (0 or 1)
- EstimatedSalary (Float)

Sentence: {text}

Respond only with JSON like:
{{
  "Gender": "Female",
  "Age": 42,
  "Tenure": 3,
  "Balance": 60000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 50000
}}
"""

prompt = PromptTemplate(input_variables=["text"], template=template)
chain = LLMChain(llm=llm, prompt=prompt)

# ==== STREAMLIT APP ====
st.title("🔮 Churn Prediction App")
st.write("Upload your dataset and predict churn from a sentence using Gemini + ML model")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file (churn dataset)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Sample Data", data.head())

    # Train model from uploaded CSV
    from sklearn.ensemble import RandomForestClassifier

    # Preprocess
    data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
    X = data[["Gender", "Age", "Tenure", "Balance", "NumOfProducts",
              "HasCrCard", "IsActiveMember", "EstimatedSalary"]]
    y = data["Churn"]

    model = RandomForestClassifier()
    model.fit(X, y)

    # Ask sentence input
    sentence = st.text_area("Enter user description (natural language)", height=100)

    if st.button("🔍 Predict Churn"):
        if sentence:
            try:
                # Gemini chain output
                parsed_output = chain.run(sentence)
                st.subheader("🔎 Gemini Extracted Details")
                st.code(parsed_output, language="json")

                # Extract JSON safely
                match = re.search(r"\{.*\}", parsed_output, re.DOTALL)
                if match:
                    user_data = json.loads(match.group(0))
                else:
                    st.error("Couldn't parse JSON from Gemini output.")
                    st.stop()

                features = [
                    1 if user_data["Gender"].lower() == "male" else 0,
                    int(user_data["Age"]),
                    int(user_data["Tenure"]),
                    float(user_data["Balance"]),
                    int(user_data["NumOfProducts"]),
                    int(user_data["HasCrCard"]),
                    int(user_data["IsActiveMember"]),
                    float(user_data["EstimatedSalary"])
                ]

                # Predict
                prediction = model.predict([features])[0]
                result = "🚫 Churn" if prediction == 1 else "✅ No Churn"
                st.success(f"**Prediction:** {result}")

            except Exception as e:
                st.error(f"Something went wrong: {e}")
