# AI-Powered Prediction & Recipe Apps

This repository contains three distinct applications that leverage **Streamlit**, **LangChain**, and **Google Gemini API** to solve real-world problems using AI. Each app demonstrates how natural language processing (NLP) and machine learning models can be combined to offer intelligent solutions in different domains.

---

## **1. Churn Prediction App (Streamlit + LangChain + Gemini + ML)**

**Overview:**  
This app predicts whether a customer will churn based on a natural language description of their characteristics. Users can upload a CSV dataset containing customer information, and the app uses **Google Gemini** (via **LangChain**) to extract features from input sentences (e.g., gender, age, balance). These features are then fed into a **RandomForestClassifier** to predict whether the customer is likely to churn.

**Key Features:**
- Upload a CSV file to train the model with customer data.
- Extract features from sentences like "A 35-year-old male with 1 product and a balance of 50,000."
- Displays predictions: **Churn** or **No Churn** based on the model.

**Technologies:**
- **Streamlit** for the web interface.
- **LangChain** and **Google Gemini** for extracting structured data from text.
- **scikit-learn** for training the **RandomForestClassifier**.

---

## **2. AI Chef - Main Course Recipe Generator (Streamlit + LangChain + Gemini)**

**Overview:**  
This application generates creative **main course recipes** based on the list of ingredients provided by the user. By entering ingredients, the app uses **Gemini via LangChain** to create a detailed recipe, including preparation steps, estimated cook time, and nutritional information per serving.

**Key Features:**
- Accepts ingredients in a comma-separated format.
- Generates a main course recipe using only the provided ingredients.
- Includes cooking steps and nutritional info per serving.

**Technologies:**
- **Streamlit** for the web interface.
- **LangChain** and **Google Gemini** for recipe generation.
- **Markdown** formatting for clean, readable output.

---

## **3. Parkinson's Disease Prediction App (Streamlit + LangChain + ML)**

**Overview:**  
This app predicts whether a person has **Parkinson's disease** based on written or spoken symptoms. It extracts key features from user input using **Google Gemini** via **LangChain**, and then feeds those features into a **Support Vector Classifier (SVC)** model to predict the likelihood of Parkinson’s.

**Key Features:**
- Accepts natural language input (typed or spoken) about symptoms.
- Extracts relevant features using **Gemini LLM**.
- Uses a **SVC model** to predict if the person has Parkinson's.
- Displays the extracted data and prediction result.

**Technologies:**
- **Streamlit** for the web interface.
- **LangChain** and **Google Gemini** for extracting features from input.
- **scikit-learn** for the **SVC model**.

---

## **Installation and Setup**

### Prerequisites:
- **Python 3.8+**
- **Gemini API Key** (Google Generative AI)

### Installation:
1. Clone the repository.
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
