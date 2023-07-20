import streamlit as st
import pandas as pd
import joblib
# from data_preprocessing import data_preprocessing, encoder_Credit_Mix, encoder_Payment_Behaviour, encoder_Payment_of_Min_Amount
# from prediction import prediction

col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png", width=130)
with col2:
    st.header('Credit Scoring App (Prototype) :sparkles:')

data = pd.DataFrame()

col1, col2, col3 = st.columns(3)

with col1:
    Credit_Mix = st.selectbox(label='Credit_Mix', options=encoder_Credit_Mix.classes_, index=1)
    data["Credit_Mix"] = [Credit_Mix]

with col2:
    Payment_of_Min_Amount = st.selectbox(label='Payment_of_Min_Amount', options=encoder_Payment_of_Min_Amount.classes_, index=1)
    data["Payment_of_Min_Amount"] = [Payment_of_Min_Amount]

with col3:
    Payment_Behaviour = st.selectbox(label='Payment_Behaviour', options=encoder_Payment_Behaviour.classes_, index=5)
    data["Payment_Behaviour"] = Payment_Behaviour

col1, col2, col3, col4 = st.columns(4)

with col1:
    # st.header("Kolom 1")
    Age = int(st.number_input(label='Age', value=23))
    data["Age"] = Age

with col2:
    Num_Bank_Accounts = int(st.number_input(label='Num_Bank_Accounts', value=3))
    data["Num_Bank_Accounts"] = Num_Bank_Accounts

with col3:
    Num_Credit_Card = int(st.number_input(label='Num_Credit_Card', value=4))
    data["Num_Credit_Card"] = Num_Credit_Card

with col4:
    Interest_Rate = float(st.number_input(label='Interest_Rate', value=3))
    data["Interest_Rate"] = Interest_Rate


col1, col2, col3, col4 = st.columns(4)

with col1:
    Num_of_Loan = int(st.number_input(label='Num_of_Loan', value=4))
    data["Num_of_Loan"] = Num_of_Loan

with col2:
    # st.header("Kolom 1")
    Delay_from_due_date = int(st.number_input(label='Delay_from_due_date', value=3))
    data["Delay_from_due_date"] = Delay_from_due_date

with col3:
    Num_of_Delayed_Payment = int(st.number_input(label='Num_of_Delayed_Payment', value=7))
    data["Num_of_Delayed_Payment"] = Num_of_Delayed_Payment

with col4:
    Changed_Credit_Limit = float(st.number_input(label='Changed_Credit_Limit', value=11.27))
    data["Changed_Credit_Limit"] = Changed_Credit_Limit

col1, col2, col3, col4 = st.columns(4)

with col1:
    Num_Credit_Inquiries = float(st.number_input(label='Num_Credit_Inquiries', value=5))
    data["Num_Credit_Inquiries"] = Num_Credit_Inquiries

with col2:
    Outstanding_Debt = float(st.number_input(label='Outstanding_Debt', value=809.98))
    data["Outstanding_Debt"] = Outstanding_Debt

with col3:
    Monthly_Inhand_Salary = float(st.number_input(label='Monthly_Inhand_Salary', value=1824.8))
    data["Monthly_Inhand_Salary"] = Monthly_Inhand_Salary

with col4:
    Monthly_Balance = float(st.number_input(label='Monthly_Balance', value=186.26))
    data["Monthly_Balance"] = Monthly_Balance

col1, col2, col3 = st.columns(3)

with col1:
    Amount_invested_monthly = float(st.number_input(label='Amount_invested_monthly', value=236.64))
    data["Amount_invested_monthly"] = Amount_invested_monthly

with col2:
    Total_EMI_per_month = float(st.number_input(label='Total_EMI_per_month', value=49.5))
    data["Total_EMI_per_month"] = Total_EMI_per_month

with col3:
    Credit_History_Age = float(st.number_input(label='Credit_History_Age', value=216))
    data["Credit_History_Age"] = Credit_History_Age

with st.expander("View the Raw Data"):
    st.dataframe(data=data, width=800, height=10)

encoder_Credit_Mix = joblib.load("model/encoder_Credit_Mix.joblib")
encoder_Payment_Behaviour = joblib.load("model/encoder_Payment_Behaviour.joblib")
encoder_Payment_of_Min_Amount = joblib.load("model/encoder_Payment_of_Min_Amount.joblib")
pca_1 = joblib.load("model/pca_1.joblib")
pca_2 = joblib.load("model/pca_2.joblib")
scaler_Age = joblib.load("model/scaler_Age.joblib")
scaler_Amount_invested_monthly = joblib.load("model/scaler_Amount_invested_monthly.joblib")
scaler_Changed_Credit_Limit = joblib.load("model/scaler_Changed_Credit_Limit.joblib")
scaler_Credit_History_Age = joblib.load("model/scaler_Credit_History_Age.joblib")
scaler_Delay_from_due_date = joblib.load("model/scaler_Delay_from_due_date.joblib")
scaler_Interest_Rate = joblib.load("model/scaler_Interest_Rate.joblib")
scaler_Monthly_Balance =joblib.load("model/scaler_Monthly_Balance.joblib")
scaler_Monthly_Inhand_Salary = joblib.load("model/scaler_Monthly_Inhand_Salary.joblib")
scaler_Num_Bank_Accounts = joblib.load("model/scaler_Num_Bank_Accounts.joblib")
scaler_Num_Credit_Card = joblib.load("model/scaler_Num_Credit_Card.joblib")
scaler_Num_Credit_Inquiries = joblib.load("model/scaler_Num_Credit_Inquiries.joblib")
scaler_Num_of_Delayed_Payment = joblib.load("model/scaler_Num_of_Delayed_Payment.joblib")
scaler_Num_of_Loan = joblib.load("model/scaler_Num_of_Loan.joblib")
scaler_Outstanding_Debt = joblib.load("model/scaler_Outstanding_Debt.joblib")
scaler_Total_EMI_per_month = joblib.load("model/scaler_Total_EMI_per_month.joblib")

pca_numerical_columns_1 = [
    'Num_Bank_Accounts',
    'Num_Credit_Card',
    'Interest_Rate',
    'Num_of_Loan',
    'Delay_from_due_date',
    'Num_of_Delayed_Payment',
    'Changed_Credit_Limit',
    'Num_Credit_Inquiries',
    'Outstanding_Debt',
    'Credit_History_Age'
]

pca_numerical_columns_2 = [
    "Monthly_Inhand_Salary",
    "Monthly_Balance", 
    "Amount_invested_monthly", 
    "Total_EMI_per_month"
]

def data_preprocessing(data):
    """PPreprocessing data

    Args:
        data (Pandas DataFrame): Dataframe that contain all the data to make prediction 
        
    return:
        Pandas DataFrame: Dataframe that contain all the preprocessed data
    """
    data = data.copy()
    df = pd.DataFrame()
    
    df["Age"] = scaler_Age.transform(np.asarray(data["Age"]).reshape(-1,1))[0]
    
    df["Credit_Mix"] = encoder_Credit_Mix.transform(data["Credit_Mix"])[0]
    df["Payment_of_Min_Amount"] = encoder_Payment_of_Min_Amount.transform(data["Payment_of_Min_Amount"])
    df["Payment_Behaviour"] = encoder_Payment_Behaviour.transform(data["Payment_Behaviour"])
    
    # PCA 1
    data["Num_Bank_Accounts"] = scaler_Num_Bank_Accounts.transform(np.asarray(data["Num_Bank_Accounts"]).reshape(-1,1))[0]
    data["Num_Credit_Card"] = scaler_Num_Credit_Card.transform(np.asarray(data["Num_Credit_Card"]).reshape(-1,1))[0]
    data["Interest_Rate"] = scaler_Interest_Rate.transform(np.asarray(data["Interest_Rate"]).reshape(-1,1))[0]
    data["Num_of_Loan"] = scaler_Num_of_Loan.transform(np.asarray(data["Num_of_Loan"]).reshape(-1,1))[0]
    data["Delay_from_due_date"] = scaler_Delay_from_due_date.transform(np.asarray(data["Delay_from_due_date"]).reshape(-1,1))[0]
    data["Num_of_Delayed_Payment"] = scaler_Num_of_Delayed_Payment.transform(np.asarray(data["Num_of_Delayed_Payment"]).reshape(-1,1))[0]
    data["Changed_Credit_Limit"] = scaler_Changed_Credit_Limit.transform(np.asarray(data["Changed_Credit_Limit"]).reshape(-1,1))[0]
    data["Num_Credit_Inquiries"] = scaler_Num_Credit_Inquiries.transform(np.asarray(data["Num_Credit_Inquiries"]).reshape(-1,1))[0]
    data["Outstanding_Debt"] = scaler_Outstanding_Debt.transform(np.asarray(data["Outstanding_Debt"]).reshape(-1,1))[0]
    data["Credit_History_Age"] = scaler_Credit_History_Age.transform(np.asarray(data["Credit_History_Age"]).reshape(-1,1))[0]
    
    df[["pc1_1", "pc1_2", "pc1_3", "pc1_4", "pc1_5"]] = pca_1.transform(data[pca_numerical_columns_1])
    
    # PCA 2
    data["Monthly_Inhand_Salary"] = scaler_Monthly_Inhand_Salary.transform(np.asarray(data["Monthly_Inhand_Salary"]).reshape(-1,1))[0]
    data["Monthly_Balance"] = scaler_Monthly_Balance.transform(np.asarray(data["Monthly_Balance"]).reshape(-1,1))[0]
    data["Amount_invested_monthly"] = scaler_Amount_invested_monthly.transform(np.asarray(data["Amount_invested_monthly"]).reshape(-1,1))[0]
    data["Total_EMI_per_month"] = scaler_Total_EMI_per_month.transform(np.asarray(data["Total_EMI_per_month"]).reshape(-1,1))[0]
    
    df[["pc2_1", "pc2_2"]] = pca_2.transform(data[pca_numerical_columns_2])
    
    return df

model = joblib.load("model/gboost_model.joblib")
result_target = joblib.load("model/encoder_target.joblib")

def prediction(data):
    """Making prediction

    Args:
        data (Pandas DataFrame): Dataframe that contain all the preprocessed data

    Returns:
        str: Prediction result (Good, Standard, or Poor)
    """
    result = model.predict(data)
    final_result = result_target.inverse_transform(result)[0]
    return final_result

if st.button('Predict'):
    new_data = data_preprocessing(data=data)
    with st.expander("View the Preprocessed Data"):
        st.dataframe(data=new_data, width=800, height=10)
    st.write("Credit Scoring: {}".format(prediction(new_data)))