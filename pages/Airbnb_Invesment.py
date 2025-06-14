import streamlit as st
from pymongo import MongoClient
import pandas as pd
import numpy as np
import duckdb
from modules import dashboard, predict, regis


# MongoDB
DB_NAME = st.secrets["mongo"]["DB_NAME"]
COLLECTION_NAME = st.secrets["mongo"]["COLLECTION_NAME"]
URI = st.secrets["mongo"]["URI"]
client = MongoClient(URI)
collection = client[DB_NAME][COLLECTION_NAME]


@st.cache_data
def load_mongo():
    df = pd.DataFrame(list(collection.find()))
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)
    df["price"] = (
        df["price"].replace(r"[\$,]", "", regex=True).astype(float)
    )  # แปลงคอลัมน์ราคาให้เป็นตัวเลข
    df["instant_bookable"] = df["instant_bookable"].map(
        {"t": True, "f": False}
    )  # แปลง 't'/'f' เป็น True/False
    df["host_is_superhost"] = df["host_is_superhost"].map(
        {"t": True, "f": False}
    )  # แปลง 't'/'f' เป็น True/False
    df["room_type"] = (
        df["room_type"].str.strip().str.lower().str.title()
    )  # ทำความสะอาดและจัดรูปแบบประเภทห้องพัก
    df["property_type"] = (
        df["property_type"].astype(str).str.strip().str.title()
    )  # ทำความสะอาดและจัดรูปแบบประเภทที่พัก
    df.fillna(0, inplace=True)  # เติมค่าว่าง (NaN) ทั้งหมดด้วย 0
    df["amenities_count"] = df.iloc[:, 32:66].sum(
        axis=1
    )  # นับจำนวนสิ่งอำนวยความสะดวกที่มี (สมมติว่าเป็นคอลัมน์ 0/1)
    if "property_type" in df.columns:
        conditions = [
            df["property_type"].str.lower().str.contains("apartment", na=False),
            df["property_type"].str.lower().str.contains("house", na=False),
            df["property_type"].str.lower().str.contains("condominium", na=False)
            | df["property_type"].str.lower().str.contains("condo", na=False),
            df["property_type"].str.lower().str.contains("hotel", na=False),
            df["property_type"].str.lower().str.contains("hostel", na=False),
        ]
    choices = ["Apartment", "House", "Condo", "Hotel", "Hostel"]
    df["property_grouped"] = np.select(conditions, choices, default="Other")
    return df


@st.cache_data
def load_duckdb():
    with duckdb.connect("db/airbnb.db") as con:
        df = con.query("SELECT * FROM airbnb").df()
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)
    df["price"] = df["price"].replace(r"[\$,]", "", regex=True).astype(float)
    df["instant_bookable"] = df["instant_bookable"].map({"t": True, "f": False})
    df["host_is_superhost"] = df["host_is_superhost"].map({"t": True, "f": False})
    df["room_type"] = df["room_type"].str.strip().str.lower().str.title()
    if "property_type" in df.columns:
        conditions = [
            df["property_type"].str.lower().str.contains("apartment", na=False),
            df["property_type"].str.lower().str.contains("house", na=False),
            df["property_type"].str.lower().str.contains("condominium", na=False)
            | df["property_type"].str.lower().str.contains("condo", na=False),
            df["property_type"].str.lower().str.contains("hotel", na=False),
            df["property_type"].str.lower().str.contains("hostel", na=False),
        ]
    choices = ["Apartment", "House", "Condo", "Hotel", "Hostel"]
    df["property_grouped"] = np.select(conditions, choices, default="Other")
    return df


def remove_outliers_grouped(df, target_col="price"):
    def iqr_filter(group):
        q1 = group[target_col].quantile(0.25)
        q3 = group[target_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return group[(group[target_col] >= lower) & (group[target_col] <= upper)]

    return df.groupby(
        ["neighbourhood", "room_type", "property_grouped", "bedrooms"],
        group_keys=False,
    ).apply(iqr_filter)


@st.cache_data
def load_area():
    with duckdb.connect("db/airbnb.db") as con:
        df = con.query("SELECT * FROM att_bkk").df()
    return df


st.set_page_config(page_title="Airbnb Investment", layout="wide")

st.sidebar.title("Airbnb Investment")
page = st.sidebar.radio(
    "Menu Investment :",
    ["Dashboard Investment", "Prediction Investment", "Registing"],
)

if page == "Dashboard Investment":
    dashboard.show(df_clean=remove_outliers_grouped(load_duckdb()), area_df=load_area())
elif page == "Prediction Investment":
    predict.show(df=remove_outliers_grouped(load_mongo()))
elif page == "Registing":
    regis.show(collection=collection)
