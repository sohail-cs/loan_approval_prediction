import requests
import streamlit as st

st.title('Loan Approval Predication')

st.text('''WELCOME TO LOAN APPROVAL PREDICATION MODEL
Where the user enters details and the model based on the information predicts if user is eligible for a loan
        ''')


#Get user details
user_details = {
    'no_of_dependents': float(st.number_input("How many dependents do you have?", min_value=0, step=1)),
    'education': st.radio("What is your educational level?", (" Graduate", " Not Graduate")),
    'self_employed': st.radio("Are you self-employed?", (" Yes", " No")),
    'income_annum': float(st.number_input("What is your annual income?", min_value=0, step=1)),
    'loan_amount': float(st.number_input("How much loan are you applying for?", min_value=0, step=1)),
    'loan_term': float(st.number_input("What is the loan term (in months)?", min_value=0, step=1)),
    'cibil_score': float(st.number_input("What is your CIBIL score?", min_value=0, step=1)),
    'residential_assets_value': float(st.number_input("What is the value of your residential assets?", min_value=0, step=1)),
    'commercial_assets_value': float(st.number_input("What is the value of your commercial assets?", min_value=0, step=1)),
    'luxury_assets_value': float(st.number_input("What is the value of your luxury assets?", min_value=0, step=1)),
    'bank_asset_value': float(st.number_input("What is the value of your bank assets?", min_value=0, step=1))
}

if st.button("Submit"):
    #Check if all user inputs are filled
    if all(values != '' for values in user_details.values()):
        #Server's url
        url = 'http://127.0.0.1:8000'

        try:
            response = requests.post(url,json=user_details)
            st.write("YOU ENTERED:")
            st.write(f"{user_details}")
            if response.status_code == 200:
                st.text("CONNECTION ESTABLISHED WITH THE SERVER")
                st.text("RESPONSE FROM SERVER")
                st.write(response.json())

            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except:
            st.error("FAILED TO CONNECT")


    else:
        st.text("PLEASE ENTER ALL DETAILS")




