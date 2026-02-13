import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter

st.set_page_config(page_title="Preprocessing", page_icon="🧹", layout="centered")
st.title("🧹 Data Preprocessing")

# ---------------- CHECK DATA ----------------
if "df" not in st.session_state:
    st.error("❌ No dataset uploaded. Please upload dataset first.")
    st.stop()

# If already preprocessed
if "X" in st.session_state and "Y" in st.session_state:
    st.success("✨ Preprocessing already completed!")
    st.write("🧮 Features Preview")
    st.dataframe(st.session_state["X"].head())
    st.write("🎯 Target Preview")
    st.dataframe(st.session_state["Y"].head())
    st.stop()

df = st.session_state["df"].copy()
st.write("### Dataset Preview")
st.dataframe(df.head())

# ---------------- USER INPUT ----------------
target_col = st.selectbox(
    "🎯 Select Target Column",
    options=df.columns
)

drop_cols = st.multiselect(
    "🗑 Select Columns to Drop (ID / Irrelevant)",
    options=[c for c in df.columns if c != target_col]
)

# ---------------- IMBALANCE CHECK ----------------
def check_imbalance(y, threshold=0.4):
    counts = Counter(y.squeeze())
    ratio = min(counts.values()) / max(counts.values())
    return ratio < threshold, ratio, dict(counts)

# ---------------- APPLY ----------------
if st.button("Apply Preprocessing"):

    df = df.drop(columns=drop_cols)
    df = df.drop_duplicates()
    df = df.dropna(subset=[target_col])

    target = df[target_col]

    # -------- Detect Problem Type --------
    if pd.api.types.is_numeric_dtype(target):
        if target.nunique() <= 15:
            p_type = "classification"
        else:
            p_type = "regression"
    else:
        p_type = "classification"

    st.session_state["p_type"] = p_type
    st.info(f"🔎 Detected Problem Type: **{p_type.upper()}**")

    with st.spinner("⚙ Processing dataset..."):

        X = df.drop(columns=target_col)
        Y = df[target_col]

        # -------- Remove High Missing Columns --------
        X = X.loc[:, X.isna().mean() < 0.5]

        # -------- Store Categorical Options --------
        cat_options = {}
        for col in X.select_dtypes(include="object"):
            cat_options[col] = X[col].dropna().unique().tolist()

        st.session_state["cat_options"] = cat_options

        encoders = {}

        # -------- Feature Processing --------
        for col in X.columns:

            if pd.api.types.is_numeric_dtype(X[col]):

                # Fill missing
                X[col] = X[col].fillna(X[col].median())

                # Outlier clipping (lightweight)
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                X[col] = X[col].clip(lower, upper)

            else:
                X[col] = X[col].fillna(X[col].mode()[0])

                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le

        st.session_state["encoders"] = encoders

        # -------- Target Processing --------
        if p_type == "classification":

            target_encoder = LabelEncoder()
            Y = pd.DataFrame(
                target_encoder.fit_transform(Y.astype(str)),
                columns=[target_col]
            )

            st.session_state["target_encoder"] = target_encoder

            is_imbalanced, ratio, dist = check_imbalance(Y)

            st.session_state["is_imbalanced"] = is_imbalanced
            st.session_state["imbalance_ratio"] = ratio
            st.session_state["class_distribution"] = dist

            if is_imbalanced:
                st.warning(
                    f"⚠ Imbalanced dataset detected (ratio = {ratio:.2f}). "
                    "SMOTE will be applied during training."
                )
            else:
                st.success("✔ Dataset is balanced")

        else:
            # Regression target clipping
            Q1 = Y.quantile(0.25)
            Q3 = Y.quantile(0.75)
            IQR = Q3 - Q1
            Y = Y.clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

        st.session_state["X"] = X
        st.session_state["Y"] = Y

        st.success("✅ Preprocessing Complete!")
        st.write("### Processed Features")
        st.dataframe(X.head())
        st.write("### Processed Target")
        st.dataframe(Y.head())
