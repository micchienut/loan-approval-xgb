import pandas as pd
import pickle
import numpy as np

import pickle

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
               'loan_int_rate', 'cb_person_cred_hist_length', 'credit_score', 'previous_loan_defaults_on_file']

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
                  'loan_amnt', 'loan_int_rate', 'cb_person_cred_hist_length',
                  'credit_score']

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

    person_age = 35
    person_gender = 'male'
    person_education = 'Master'
    person_income = 80000
    person_emp_exp = 4
    person_home_ownership = 'RENT'
    loan_amnt = 12000
    loan_intent = 'EDUCATION'
    loan_int_rate = 15
    cb_person_cred_hist_length = 5
    credit_score = 556
    previous_loan_defaults_on_file = 'No'
    
    user_input = [person_age,
                  person_gender,
                  person_education,
                  person_income,
                  person_emp_exp,
                  person_home_ownership,
                  loan_amnt,
                  loan_intent,
                  loan_int_rate,
                  cb_person_cred_hist_length,
                  credit_score,
                  previous_loan_defaults_on_file]
    input_df = preprocess_input(user_input, map, clip, encoder, scaler)
    
    prediction = model.predict(input_df)
    result = "Loan approved" if prediction[0] == 1 else "Loan rejected"
    print(result)

if __name__ == "__main__":
    main()