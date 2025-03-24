from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle


with open("model.pkl",'rb') as f:
    dt = pickle.load(f)

with open('preprocessor.pkl','rb')as pf:
    preprocessor = pickle.load(pf)

with open("label_encoder.pkl", "rb") as lf:
    le = pickle.load(lf)  # Load the label encoder used for 'loan_status'


app = FastAPI()

class LoanData(BaseModel):
    no_of_dependents: float
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: float
    cibil_score: float
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float



@app.post("/")
def loan(data:LoanData):
    #convert to pandas dataframe
    input_data = pd.DataFrame([data.model_dump()])
    pre_processed_data = preprocessor.transform(input_data)
    prediction = dt.predict(pre_processed_data)
    predicted_label = le.inverse_transform(prediction)  # Converts numeric encoded value to string

    return {"prediction": predicted_label[0]}




