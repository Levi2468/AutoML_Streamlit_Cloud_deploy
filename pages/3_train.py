import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Train Model", page_icon="🏋️", layout="centered")
st.title("🔄 Model Training")

# ---------------- CHECK ----------------
if "X" not in st.session_state or "Y" not in st.session_state:
    st.error("❌ Please preprocess the dataset first.")
    st.stop()

if "best_model" in st.session_state:
    st.success("🏆 Model already trained")
    st.write(f"**{st.session_state['best_tuned_model_name']}**")
    st.write(f"Score: {st.session_state['best_tuned_score']:.4f}")
    st.stop()

X = st.session_state["X"]
Y = st.session_state["Y"]
p_type = st.session_state["p_type"]

# ---------------- DATA SIZE SAFETY ----------------
if X.shape[0] > 30000:
    st.warning("⚠ Dataset too large for free deployment. Limiting to 30,000 rows.")
    X = X.sample(30000, random_state=42)
    Y = Y.loc[X.index]

# ---------------- TRAIN TEST SPLIT ----------------
stratify = Y if p_type == "classification" else None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.25,
    random_state=42,
    stratify=stratify
)

# ---------------- SMOTE ----------------
if p_type == "classification" and st.session_state.get("is_imbalanced", False):
    min_class = min(st.session_state["class_distribution"].values())
    if min_class >= 10:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        st.info("⚖ SMOTE applied on training data")

# ---------------- MODELS ----------------
if p_type == "classification":
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False
        )
    }
else:
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
    }

# ---------------- TRAIN BASELINE ----------------
results = {}

with st.spinner("🔍 Training models..."):

    for name, model in models.items():

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if p_type == "classification":
                if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
                    score = roc_auc_score(
                        y_test,
                        model.predict_proba(X_test)[:, 1]
                    )
                else:
                    score = roc_auc_score(
                        y_test,
                        model.predict_proba(X_test),
                        multi_class="ovr"
                    )
            else:
                score = r2_score(y_test, y_pred)

            results[name] = score

        except Exception as e:
            st.warning(f"{name} failed: {e}")

# ---------------- RESULTS ----------------
results_df = (
    pd.DataFrame(results.items(), columns=["Model", "Score"])
    .sort_values("Score", ascending=False)
    .reset_index(drop=True)
)

st.success("✅ Training Complete")
st.dataframe(results_df)

# ---------------- SELECT BEST ----------------
best_model_name = results_df.iloc[0]["Model"]
best_score = results_df.iloc[0]["Score"]

best_model = models[best_model_name]
best_model.fit(X_train, y_train)

st.success("🏆 Best Model Selected")
st.write(f"**{best_model_name}**")
st.write(f"Score: {best_score:.4f}")

# ---------------- SAVE ----------------
st.session_state["best_model"] = best_model
st.session_state["best_tuned_model_name"] = best_model_name
st.session_state["best_tuned_score"] = best_score
