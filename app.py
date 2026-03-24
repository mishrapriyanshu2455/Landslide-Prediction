
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt


model = joblib.load("landslide_model.pkl")
features = joblib.load("features.pkl")


st.set_page_config(
    page_title="Landslide Risk Predictor",
    page_icon="🌍",
    layout="wide"
)


st.title("🌍 Landslide Risk Prediction System")
st.markdown("### Predict landslide risk using environmental parameters")

st.markdown("---")


st.sidebar.header("⚙️ Input Parameters")

terrain = st.sidebar.selectbox("Terrain Type", ["plains", "hills", "mountains"])
soil = st.sidebar.selectbox("Soil Type", ["clay", "sand", "rock"])

rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 150.0)
slope = st.sidebar.slider("Slope (degrees)", 0.0, 50.0, 20.0)
elevation = st.sidebar.slider("Elevation (m)", 0.0, 4000.0, 1000.0)
vegetation = st.sidebar.slider("Vegetation Index", 0.0, 1.0, 0.5)
earthquake = st.sidebar.slider("Earthquake Intensity", 0.0, 7.0, 2.0)
moisture = st.sidebar.slider("Moisture Retention", 0.0, 1.0, 0.5)



input_dict = {
    "rainfall": rainfall,
    "vegetation": vegetation,
    "slope": slope,
    "elevation": elevation,
    "earthquake_intensity": earthquake,
    "moisture_retention": moisture,
    "terrain_hills": 1 if terrain == "hills" else 0,
    "terrain_mountains": 1 if terrain == "mountains" else 0,
    "terrain_plains": 1 if terrain == "plains" else 0,
    "soil_type_clay": 1 if soil == "clay" else 0,
    "soil_type_rock": 1 if soil == "rock" else 0,
    "soil_type_sand": 1 if soil == "sand" else 0
}

input_df = pd.DataFrame([input_dict])
input_df = input_df[features]


col1, col2 = st.columns([1, 1])


with col1:
    st.subheader("📋 Input Summary")
    st.dataframe(input_df, use_container_width=True)


with col2:
    st.subheader("🔍 Prediction")

    if st.button("🚀 Predict Landslide Risk", use_container_width=True):

        prob = model.predict_proba(input_df)[0][1]
        prediction = 1 if prob > 0.35 else 0

        st.markdown("### Result")

        if prediction == 1:
            st.error(f"⚠️ High Risk ({prob:.2f})")
        else:
            st.success(f"✅ Low Risk ({prob:.2f})")

       
        st.progress(int(prob * 100))

        st.markdown("### 🧠 Interpretation")
        if prob > 0.7:
            st.write("Very high probability due to extreme environmental conditions.")
        elif prob > 0.4:
            st.write("Moderate risk — some factors are contributing significantly.")
        else:
            st.write("Low risk — conditions are relatively stable.")


st.markdown("---")
st.subheader("📊 Feature Importance")

try:
    rf_model = model.named_steps['classifier']
    importances = rf_model.feature_importances_

    df_imp = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_imp["Feature"], df_imp["Importance"])
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")

    st.pyplot(fig)

except Exception as e:
    st.error(f"Error loading feature importance: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "Built with ❤️ using **Machine Learning + Streamlit** | "
    "Optimized for recall in risk prediction"
)

