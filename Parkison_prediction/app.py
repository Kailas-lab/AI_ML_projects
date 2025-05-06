import streamlit as st
import numpy as np
import joblib
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import gdown

# --- Load model and scaler ---
#model = joblib.load("svc_model.pkl")
#scaler = joblib.load("scaler.pkl")

# Google Drive File ID for model
# ------------------------------------------------------


file_id = "1UGSJtjEUgoeReAcw-Jjl1QMapCKP4c4M"
url = f"https://drive.google.com/uc?id={file_id}"
output = "svc_model.pkl"

# Download the model if it doesn't exist
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Load the model
model = joblib.load(output)
# ----------------------------------------------------------
file_id = "1ydeGT8f4maGkpTg-tfay9B60ESOgwQlW"
url = f"https://drive.google.com/uc?id={file_id}"
output = "scaler.pkl"
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

scaler = joblib.load(output)

# ----------------------------------------------------------

# --- Configure Gemini API ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyDTjCC5GTBSS5MXWJzYzoPueYcmcv58Wqw"  # Replace with st.secrets if deploying

# --- Initialize Gemini model ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# --- Prompt Template ---
template = """
You are an assistant that extracts medical voice measurements from casual patient speech.

From the patient's message, extract the following **22 voice features** in this order:
Fo, Fhi, Flo, Jitter(%), Jitter(Abs), RAP, PPQ, DDP, Shimmer, Shimmer(dB),
Shimmer:APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE.

Only return a comma-separated list of 22 numeric values, no text explanation.

Patient message:
{sentence}
"""

prompt = PromptTemplate(input_variables=["sentence"], template=template)
chain = LLMChain(llm=llm, prompt=prompt)

# --- Streamlit UI ---
st.set_page_config(page_title="Parkinson's Predictor", layout="centered")
st.title("🧠 Parkinson's Disease Prediction")
st.write("Describe your symptoms and the model will try to predict whether it's Parkinson's related.")

sentence = st.text_area("Enter your sentence (e.g. 'My voice trembles and I can't speak steadily'):")

if st.button("Analyze and Predict"):
    if sentence.strip():
        with st.spinner("Extracting features and predicting..."):
            response = chain.run(sentence)
            st.subheader("🔍 Extracted Features")
            st.code(response)

            try:
                values = [float(x.strip()) for x in response.split(',')]
                if len(values) != 22:
                    st.error("❌ Extraction failed: Expected 22 values.")
                else:
                    input_array = np.asarray(values).reshape(1, -1)
                    input_scaled = scaler.transform(input_array)
                    prediction = model.predict(input_scaled)

                    if prediction[0] == 1:
                        st.error("🔴 Prediction: Positive for Parkinson's")
                    else:
                        st.success("🟢 Prediction: Negative for Parkinson's")
            except Exception as e:
                st.error(f"⚠️ Error during prediction: {e}")
    else:
        st.warning("Please enter a sentence to analyze.")
