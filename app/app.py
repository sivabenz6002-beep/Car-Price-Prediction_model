import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>

.stApp{
background-color:#eef5ff;
}

.title{
font-size:40px;
font-weight:bold;
color:black;
}

.metric-card{
background:white;
padding:20px;
border-radius:10px;
text-align:center;
box-shadow:0px 3px 10px rgba(0,0,0,0.05);
}

.metric-title{
font-size:16px;
color:#333;
}

.metric-value{
font-size:28px;
font-weight:bold;
color:#1565c0;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🚗 Car Price Prediction Dashboard</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "car_price_model.pkl")

model = joblib.load(model_path)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------- PRICE DISTRIBUTION ----------------
    if "selling_price" in df.columns:

        st.subheader("Price Distribution")

        fig, ax = plt.subplots()

        sns.histplot(df["selling_price"], kde=True)

        st.pyplot(fig)

    # ---------------- MODEL EVALUATION ----------------
    if "selling_price" in df.columns:

        st.subheader("Model Performance")

        X = df.drop(columns=["selling_price"])
        y = df["selling_price"]

        y_pred = model.predict(X)

        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred) ** 0.5

        col1, col2, col3 = st.columns(3)

        col1.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">R² Score</div>
        <div class="metric-value">{r2:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">MAE</div>
        <div class="metric-value">{mae:.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        col3.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">RMSE</div>
        <div class="metric-value">{rmse:.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- ACTUAL VS PREDICTED ----------------
        st.subheader("Actual vs Predicted")

        fig2, ax2 = plt.subplots()

        ax2.scatter(y, y_pred)

        ax2.set_xlabel("Actual Price")
        ax2.set_ylabel("Predicted Price")

        st.pyplot(fig2)
# ---------------- FEATURE IMPORTANCE ----------------
        st.subheader("Feature Importance")

        try:

            # get trained random forest model
            rf_model = model.named_steps["model"]

            # get importance values
            importances = rf_model.feature_importances_

            # get transformed feature names from pipeline
            preprocessor = model.named_steps["preprocessor"]

            feature_names = preprocessor.get_feature_names_out()

            # create dataframe
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            fig, ax = plt.subplots()

            sns.barplot(
                x="Importance",
                y="Feature",
                data=importance_df.head(15),
                ax=ax
            )

            ax.set_title("Top Important Features")

            st.pyplot(fig)

        except Exception as e:

            st.warning("Feature importance could not be generated.")
            st.write(e)
        # ---------------- PREDICTIONS ----------------
        df["predicted_price"] = y_pred

        st.subheader("Prediction Results")

        st.dataframe(df.head())

        # ---------------- DOWNLOAD ----------------
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Predictions",
            csv,
            "car_price_predictions.csv",
            "text/csv"
        )