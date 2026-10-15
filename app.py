import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load("diamond_price_model.joblib")
    with open("model_info.json") as f:
        info = json.load(f)
    return model, info


model, info = load_model()
ranges = info["ranges"]
categories = info["categories"]

st.title("💎 Diamond Price Predictor")
st.write(
    f"Predicts the price of a diamond with a **{info['algorithm']}** model, chosen by "
    f"{info['folds']}-fold cross validation on {info['training_rows']:,} diamonds."
)

st.subheader("Diamond quality")
col1, col2, col3 = st.columns(3)
cut = col1.selectbox("Cut", categories["cut"], index=categories["cut"].index("Ideal"))
color = col2.selectbox("Colour (D = best)", categories["color"], index=categories["color"].index("G"))
clarity = col3.selectbox("Clarity (IF = best)", categories["clarity"], index=categories["clarity"].index("VS2"))

st.subheader("Size and shape")
carat = st.slider("Carat (weight)", float(ranges["carat"][0]), 3.0, 0.9, step=0.01)
col4, col5 = st.columns(2)
depth = col4.number_input("Depth (%)", float(ranges["depth"][0]), float(ranges["depth"][1]), 61.8, step=0.1)
table = col5.number_input("Table (%)", float(ranges["table"][0]), float(ranges["table"][1]), 57.0, step=0.5)
col6, col7, col8 = st.columns(3)
x = col6.number_input("Length x (mm)", 3.0, 11.0, 6.2, step=0.01)
y = col7.number_input("Width y (mm)", 3.0, 11.0, 6.15, step=0.01)
z = col8.number_input("Depth z (mm)", 1.0, 7.0, 3.82, step=0.01)

if st.button("Predict Price", type="primary", use_container_width=True):
    features = pd.DataFrame([{
        "carat": carat, "depth": depth, "table": table, "x": x, "y": y, "z": z,
        "cut": cut, "color": color, "clarity": clarity,
    }])
    price = float(model.predict(features)[0])
    st.metric("Predicted price", f"${price:,.0f}")
    st.caption(
        f"Typical error: about \\${info['cv_mean_rmse']:,.0f} "
        f"(cross-validation RMSE, ± \\${info['cv_std_rmse']:,.0f} across folds)."
    )

st.divider()
st.caption("AIC354 Machine Learning Fundamentals, Assignment 3 · Zain Aftab (FA24-BSE-155)")
