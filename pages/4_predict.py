import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Predict", page_icon="🎯", layout="centered")
st.title("🔮 Make Prediction")

# ---------------- CHECK MODEL ----------------
if "best_model" not in st.session_state or "X" not in st.session_state:
    st.error("❌ Model not trained yet. Please train model first.")
    st.stop()

model = st.session_state["best_model"]
X = st.session_state["X"]
X_cols = X.columns
p_type = st.session_state["p_type"]
encoders = st.session_state.get("encoders", {})
target_encoder = st.session_state.get("target_encoder", None)
cat_values = st.session_state.get("cat_options", {})

# ---------------- SESSION INIT ----------------
st.session_state.setdefault("input_values", {})
st.session_state.setdefault("prediction", None)
st.session_state.setdefault("confidence", None)
st.session_state.setdefault("input_df", None)
st.session_state.setdefault("prediction_history", pd.DataFrame())

st.write("### 📝 Enter values for prediction:")

input_values = {}

# ---------------- INPUT FORM ----------------
for col in X_cols:

    if col in encoders:  # categorical
        options = cat_values.get(col, list(encoders[col].classes_))
        default = st.session_state["input_values"].get(col, options[0])

        input_values[col] = st.selectbox(
            col,
            options=options,
            index=options.index(default) if default in options else 0
        )

    else:  # numerical
        default = st.session_state["input_values"].get(
            col,
            float(X[col].median())
        )

        input_values[col] = st.number_input(
            col,
            value=float(default)
        )

# ---------------- PREP INPUT ----------------
raw_input_df = pd.DataFrame([input_values])
encoded_df = raw_input_df.copy()

for col in encoded_df.columns:
    if col in encoders:
        encoded_df[col] = encoders[col].transform(
            encoded_df[col].astype(str)
        )

st.session_state["input_df"] = raw_input_df

# ---------------- PREDICT ----------------
if st.button("Predict"):

    try:
        st.session_state["input_values"] = input_values

        prediction = model.predict(encoded_df)[0]
        st.session_state["prediction"] = prediction

        confidence = None

        if p_type == "classification" and hasattr(model, "predict_proba"):
            probs = model.predict_proba(encoded_df)[0]
            confidence = float(np.max(probs))

        st.session_state["confidence"] = confidence

        st.success("✅ Prediction successful")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ---------------- DISPLAY RESULT ----------------
if st.session_state["prediction"] is not None:

    if p_type == "classification":

        pred_idx = int(st.session_state["prediction"])

        if target_encoder:
            label = target_encoder.inverse_transform([pred_idx])[0]
        else:
            label = pred_idx

        st.success(f"🎯 Predicted Class: **{label}**")

        if st.session_state["confidence"] is not None:
            st.info(f"📊 Confidence: **{st.session_state['confidence']:.2%}**")

    else:
        st.success(
            f"📌 Predicted Value: **{float(st.session_state['prediction']):.4f}**"
        )

# ---------------- SAVE PREDICTION ----------------
if st.button("💾 Save Prediction"):

    if st.session_state["prediction"] is None:
        st.warning("⚠ Please make a prediction first.")
    else:
        row = st.session_state["input_df"].copy()
        row["prediction"] = st.session_state["prediction"]

        if p_type == "classification" and st.session_state["confidence"] is not None:
            row["confidence"] = st.session_state["confidence"]

        st.session_state["prediction_history"] = pd.concat(
            [st.session_state["prediction_history"], row],
            ignore_index=True
        )

        st.success("📁 Prediction saved successfully!")

# ---------------- HISTORY ----------------
if not st.session_state["prediction_history"].empty:

    st.write("### 📊 Prediction History")
    st.dataframe(st.session_state["prediction_history"])

    st.download_button(
        label="📥 Download Prediction History",
        data=st.session_state["prediction_history"].to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv"
    )
