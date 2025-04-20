import streamlit as st
import pandas as pd
import pickle
import numpy as np

import pickle
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# Load all pickled model
def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def load_scaler(filename):
    with open(filename, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def load_encoder(filename):
    with open(filename, 'rb') as f:
        encoder = pickle.load(f)
    return encoder

def load_map(filename):
    with open(filename, 'rb') as f:
        map_config = pickle.load(f)
    return map_config

def load_clip(filename):
    with open(filename, 'rb') as f:
        clip_config = pickle.load(f)
    return clip_config

def preprocess_input(user_input, map_config, clip_config, encoder, scaler):
    # convert to df
    columns = ['person_age', 'person_gender', 'person_education', 'person_income',
               'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
               'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
               'credit_score', 'previous_loan_defaults_on_file']

    df = pd.DataFrame([user_input], columns=columns)

    # clip extreme values
    for col, rules in clip_config.items():
        df[col] = df[col].clip(**rules)

    # mapping
    for col, mapping in map_config.items():
        if col == 'person_gender':
            df[col] = df[col].str.lower().str.replace(" ", "")
        df[col] = df[col].map(mapping)
    
    # encode
    ohe_cols = ['person_home_ownership', 'loan_intent']
    to_encode = encoder.transform(df[ohe_cols])
    encoded_df = pd.DataFrame(to_encode, columns=encoder.get_feature_names_out(ohe_cols))

    df = pd.concat([df.drop(ohe_cols, axis=1), encoded_df], axis=1)

    # scaling
    scale_cols = ['person_age', 'person_income', 'person_emp_exp',
                  'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                  'cb_person_cred_hist_length', 'credit_score']

    df[scale_cols] = scaler.transform(df[scale_cols])

    return df

def predict_with_model(model, user_input):
    prediction = model.predict(user_input)
    return prediction[0]

def main():
    model_filename = 'loan_approval_xgb.pkl'
    model = load_model(model_filename)
    
    scaler_filename = 'scaler.pkl'
    scaler = load_scaler(scaler_filename)

    encoder_filename = 'encoder.pkl'
    encoder = load_encoder(encoder_filename)

    map_filename = 'mapping.pkl'
    map = load_map(map_filename)
    
    clip_filename = 'clip_config.pkl'
    clip = load_clip(clip_filename)

    st.title("💰 Loan Approval Status Prediction 💳")

    person_age = st.number_input("Age", min_value=0, max_value=100, value=30)
    person_gender = st.selectbox("Gender", options=['male', 'female'])
    person_education = st.selectbox("Education", options=['High School', 'Bachelor', 'Associate', 'Master', 'Doctorate'])
    person_income = st.number_input("Annual Income", min_value=0)
    person_emp_exp = st.number_input("Years of Work Experience", min_value=0, max_value=80, value=5)
    person_home_ownership = st.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
    loan_amnt = st.number_input("Loan Amount", min_value=0)
    loan_intent = st.selectbox("Loan Intent", ['VENTURE', 'PERSONAL', 'EDUCATION', 'MEDICAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
    loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0)
    loan_percent_income = st.number_input("Loan % of Income", min_value=0.0)
    cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0)
    credit_score = st.number_input("Credit Score", min_value=0)
    previous_loan_defaults_on_file = st.selectbox("Any Previous Loan Defaults?", ['Yes', 'No'])
    
    user_input = [person_age,
                  person_gender,
                  person_education,
                  person_income,
                  person_emp_exp,
                  person_home_ownership,
                  loan_amnt,
                  loan_intent,
                  loan_int_rate,
                  loan_percent_income,
                  cb_person_cred_hist_length,
                  credit_score,
                  previous_loan_defaults_on_file]
    input_df = preprocess_input(user_input, map, clip, encoder, scaler)

    if st.button("Predict Loan Approval"):
        prediction = model.predict(input_df)
        result = "✅ Approved" if prediction[0] == 1 else "❌ Rejected"
        st.subheader(f"Prediction: {result}")

if __name__ == "__main__":
    main()

