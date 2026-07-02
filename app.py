
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")

st.title("🚗 Car Price Prediction using Linear Regression")

st.markdown("""
Place **car_data.csv** in the same folder as this app.
Dataset columns expected:
Year,Selling_Price,Present_Price,Kms_Driven,Fuel_Type,Seller_Type,Transmission,Owner
""")

try:
    df = pd.read_csv("car_data.csv")
except Exception as e:
    st.error("car_data.csv not found. Download it and place it beside app.py.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# preprocessing
data = df.copy()
data["Car_Age"] = 2026 - data["Year"]
data = data.drop("Year", axis=1)

data = pd.get_dummies(
    data,
    columns=["Fuel_Type","Seller_Type","Transmission"],
    drop_first=True
)

X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
score = r2_score(y_test, pred)

st.success(f"Model R² Score: {score:.3f}")

st.header("Enter Car Details")

col1,col2 = st.columns(2)

with col1:
    year = st.number_input("Manufacturing Year",2000,2026,2020)
    present = st.number_input("Present Price (Lakhs)",0.1,100.0,8.5)
    kms = st.number_input("Kilometers Driven",0,500000,35000)

with col2:
    fuel = st.selectbox("Fuel Type",["Petrol","Diesel","CNG"])
    seller = st.selectbox("Seller Type",["Dealer","Individual"])
    trans = st.selectbox("Transmission",["Manual","Automatic"])
    owner = st.selectbox("Previous Owners",[0,1,2,3])

if st.button("Predict Selling Price"):
    row = {
        "Present_Price":present,
        "Kms_Driven":kms,
        "Owner":owner,
        "Car_Age":2026-year,
        "Fuel_Type_Diesel":0,
        "Fuel_Type_Petrol":0,
        "Seller_Type_Individual":0,
        "Transmission_Manual":0
    }

    # match dummy columns
    if "Fuel_Type_Diesel" in X.columns and fuel=="Diesel":
        row["Fuel_Type_Diesel"]=1
    if "Fuel_Type_Petrol" in X.columns and fuel=="Petrol":
        row["Fuel_Type_Petrol"]=1
    if "Seller_Type_Individual" in X.columns and seller=="Individual":
        row["Seller_Type_Individual"]=1
    if "Transmission_Manual" in X.columns and trans=="Manual":
        row["Transmission_Manual"]=1

    input_df = pd.DataFrame([row])
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    price = model.predict(input_df)[0]

    st.metric("Estimated Selling Price", f"₹ {price:.2f} Lakhs")

st.header("Actual vs Predicted")
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(y_test, pred)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)
