import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

import pickle

class DataHandler:
    def __init__(self, file_path):
        # Initialization
        self.file_path = file_path
        self.df, self.X, self.y, self.cat, self.num = [None] * 5
    
    # Read data
    def load_data(self):
        self.df = pd.read_csv(self.file_path)
    
    # Split X and y
    def split_X_y(self, target):
        self.X = self.df.drop(target, axis=1)
        self.y = self.df[target]

    # Age & emp experience clipping
    def clip_data(self):
        self.clip_config = {
            'person_age': {'upper': 120},
            'person_emp_exp': {'upper': 80}
        }

        self.df['person_age'] = self.df['person_age'].clip(**self.clip_config['person_age'])
        self.df['person_emp_exp'] = self.df['person_emp_exp'].clip(**self.clip_config['person_emp_exp'])

    # Categorical data mapping
    def mapping(self):
        self.map_config = {
            'person_gender': {'male': 0, 'female': 1},
            'previous_loan_defaults_on_file': {'Yes': 1, 'No': 0},
            'person_education': {
                'High School': 0,
                'Associate': 1,
                'Bachelor': 2,
                'Master': 3,
                'Doctorate': 4
            }
        }

        self.df['person_gender'] = self.df['person_gender'].str.lower().str.replace(" ", "").map(self.map_config['person_gender'])
        self.df['previous_loan_defaults_on_file'] = self.df['previous_loan_defaults_on_file'].map(self.map_config['previous_loan_defaults_on_file'])
        self.df['person_education'] = self.df['person_education'].map(self.map_config['person_education'])
    
    # Drop column
    def drop(self, col):
        self.df = self.df.drop(col, axis=1)

    # Derive loan_percent_income
    def derive_percent(self):
        self.df['loan_percent_income'] = (self.df['loan_amnt']/self.df['person_income'])*100
    
    def check_data(self):
        print(self.df.head())

    def save_clip_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.clip_config, f)

    def save_map_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.map_config, f)

class ModelHandler:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.createModel()
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_pred = [None] * 5
        self.scaler = RobustScaler()
        self.encoder = OneHotEncoder(sparse_output=False)
        
        # Split categorical & numerical data in X
        self.scale_col = ['person_age', 'person_income', 'person_emp_exp',
                    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                    'cb_person_cred_hist_length', 'credit_score']
        self.ohe_col = ['person_home_ownership', 'loan_intent']
    
    def check_X(self):
        print(self.X_train.head())
        print(self.X_test.head())

    def getMedian(self, col):
        return np.nanmedian(self.X_train[col])
    
    def createModel(self):
        self.model = XGBClassifier()
    
    def one_hot(self):
        encoder_ohe = self.encoder.fit_transform(self.X_train[self.ohe_col])
        encoded_df = pd.DataFrame(encoder_ohe, columns=(self.encoder.get_feature_names_out(self.ohe_col)),
                                       index=self.X_train.index)
        self.X_train = pd.concat([self.X_train, encoded_df], axis=1)
        self.X_train = self.X_train.drop(self.ohe_col, axis=1)
        
        encoder_ohe_test = self.encoder.transform(self.X_test[self.ohe_col])
        encoded_df_test = pd.DataFrame(encoder_ohe_test, columns=(self.encoder.get_feature_names_out(self.ohe_col)),
                                       index=self.X_test.index)
        self.X_test = pd.concat([self.X_test, encoded_df_test], axis=1)
        self.X_test = self.X_test.drop(self.ohe_col, axis=1)
    
    def robust_scaler(self):
        for c in self.scale_col:
            self.X_train[c] = self.scaler.fit_transform(self.X_train[c].values.reshape(-1, 1))
            self.X_test[c] = self.scaler.transform(self.X_test[c].values.reshape(-1, 1))
    
    def fillNA(self, col, number):
        self.X_train[col].fillna(number)
        self.X_test[col].fillna(number)
    
    def makePrediction(self):
        self.y_pred = self.model.predict(self.X_test)

    def createReport(self):
        print('XGB Classification Report')
        print(classification_report(self.y_test, self.y_pred, target_names=['Rejected', 'Approved']))
    
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
    
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
    
    def save_model_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def save_encoder_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.encoder, f)

    def save_scaler_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.scaler, f)

# Main program
file_path = 'Dataset_A_loan.csv'
data_handler = DataHandler(file_path)
data_handler.load_data() # read data
data_handler.drop('loan_percent_income')
data_handler.derive_percent()
data_handler.clip_data() # clip extreme values
data_handler.mapping() # categorical mapping to numerical
# data_handler.check_data() # check df.head() to see progress
data_handler.split_X_y('loan_status') # split X and y on loan_status
X = data_handler.X
y = data_handler.y

model_handler = ModelHandler(X, y)
model_handler.split_data() # split X_train X_test y_train y_test
model_handler.one_hot() # apply one hot to ohe_col
med_income = model_handler.getMedian('person_income') # get median from person income
model_handler.fillNA('person_income', med_income) # fill null in person_income
model_handler.robust_scaler() # apply scaling
# model_handler.check_X() # check X_train and X_test after applying functions

model_handler.train_model() # train XGB
model_handler.makePrediction() # predict from trained model
model_handler.createReport()

model_handler.save_model_to_file('loan_approval_oop_xgb.pkl')
model_handler.save_scaler_to_file('scaler_oop.pkl')
model_handler.save_encoder_to_file('encoder_oop.pkl')
data_handler.save_clip_to_file('clip_oop.pkl')
data_handler.save_map_to_file('map_oop.pkl')