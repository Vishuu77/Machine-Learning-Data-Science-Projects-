import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from engine import load_sample_data, train_ml_model, run_python_code
import io

st.set_page_config(page_title="Data Science & ML Compiler", layout="wide", page_icon="⚡")

# Custom Dark Theme Styling
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stSidebar { background-color: #1a1d29; }
    h1, h2, h3 { color: #00E5A0; }
    .stButton>button { background-color: #00E5A0; color: #0e1117; font-weight: bold; border-radius: 6px; }
    .stButton>button:hover { background-color: #00b37d; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Data Science & Machine Learning Compiler")
st.markdown("Upload datasets, explore statistics, train ML models, and execute custom Python scripts live.")

# Sidebar Data Loading
st.sidebar.header("1. Data Input")
data_source = st.sidebar.radio("Choose Data Source", ["Sample Dataset", "Upload CSV"])

df = pd.DataFrame()
target_col = ""

if data_source == "Sample Dataset":
    sample_name = st.sidebar.selectbox("Select Sample", ["Iris", "Wine", "California Housing"])
    df, target_col = load_sample_data(sample_name)
    st.sidebar.success(f"Loaded {sample_name} ({df.shape[0]} rows, {df.shape[1]} cols)")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        target_col = st.sidebar.selectbox("Select Target Column", options=df.columns)

if not df.empty:
    # Tabs for Workflow
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer", "🤖 ML Model Trainer", "💻 Python DS Compiler", "🔮 Interactive Predictor"])
    
    with tab1:
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))
        
        st.dataframe(df.head(10), width="stretch")
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), width="stretch")
        
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
        st.pyplot(fig)
        
    with tab2:
        st.subheader("Train Machine Learning Model")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            task_type = st.selectbox("Task Type", ["Classification", "Regression"])
            if task_type == "Classification":
                model_name = st.selectbox("Model", ["Random Forest", "Logistic Regression", "SVM"])
            else:
                model_name = st.selectbox("Model", ["Random Forest Regressor", "Linear Regression", "SVR"])
        with col_m2:
            test_size = st.slider("Test Split Ratio", 0.1, 0.4, 0.2, 0.05)
            n_estimators = st.slider("Estimators (RF)", 10, 300, 100, 10) if "Random Forest" in model_name else 100
            max_depth = st.slider("Max Depth (RF)", 2, 20, 5) if "Random Forest" in model_name else None
            
        params = {"n_estimators": n_estimators, "max_depth": max_depth}
        
        if st.button("🚀 Train Model"):
            try:
                with st.spinner("Training model..."):
                    results = train_ml_model(df, target_col, task_type, model_name, params, test_size)
                    st.session_state["model_results"] = results
                st.success("Model trained successfully!")
            except Exception as e:
                st.error(f"Training failed: {e}")
                
        if "model_results" in st.session_state:
            res = st.session_state["model_results"]
            st.markdown("---")
            st.subheader("Evaluation Results")
            
            if res["task"] == "Classification":
                st.metric("Test Accuracy", f"{res['accuracy'] * 100:.2f}%")
                
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.markdown("**Confusion Matrix**")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
                    st.pyplot(fig)
                with col_c2:
                    if res["feature_importance"] is not None:
                        st.markdown("**Feature Importances**")
                        fig, ax = plt.subplots(figsize=(5, 4))
                        res["feature_importance"].head(10).plot(kind="barh", ax=ax, color="#00E5A0")
                        st.pyplot(fig)
            else:
                st.metric("RMSE", f"{res['rmse']:.4f}")
                st.metric("R² Score", f"{res['r2']:.4f}")
                
                if res["feature_importance"] is not None:
                    st.markdown("**Feature Coefficients / Importances**")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    res["feature_importance"].head(10).plot(kind="barh", ax=ax, color="#00E5A0")
                    st.pyplot(fig)
                    
            # Save model download
            model_bytes = io.BytesIO()
            import joblib
            joblib.dump(res["model"], model_bytes)
            st.download_button("📥 Download Trained Model (.pkl)", model_bytes.getvalue(), file_name="trained_model.pkl", mime="application/octet-stream")

    with tab3:
        st.subheader("Python DS Script Compiler & Sandbox")
        st.markdown("Write custom pandas/matplotlib/numpy code. The dataframe is available as variable `df`.")
        
        default_code = "# Example: Display top 5 rows and plot target distribution\nprint(df.head())\nif len(df.columns) > 0:\n    df.iloc[:, 0].hist()\n    plt.title('Distribution of First Column')"
        user_code = st.text_area("Python Code Editor", value=default_code, height=200)
        
        if st.button("▶ Run Script"):
            with st.spinner("Executing script..."):
                stdout, fig = run_python_code(user_code, df)
                
            st.markdown("**Console Output / Print Results:**")
            st.code(stdout if stdout else "Script executed with no output.", language="python")
            
            if fig is not None:
                st.markdown("**Generated Plot:**")
                st.pyplot(fig)
                
    with tab4:
        st.subheader("Interactive Predictor")
        if "model_results" in st.session_state:
            res = st.session_state["model_results"]
            features = res["features"]
            
            st.markdown("Provide feature values to get predictions from the trained model:")
            input_vals = {}
            col_preds = st.columns(2)
            for i, feat in enumerate(features):
                with col_preds[i % 2]:
                    default_val = float(df[feat].mean()) if feat in df.columns else 0.0
                    input_vals[feat] = st.number_input(f"{feat}", value=default_val)
                    
            if st.button("🔮 Predict"):
                input_df = pd.DataFrame([input_vals])
                pred = res["model"].predict(input_df)[0]
                st.success(f"Predicted Output: **{pred}**")
        else:
            st.info("Train a model in the 'ML Model Trainer' tab first to use the predictor.")
else:
    st.info("Please select a sample dataset or upload a CSV file from the sidebar to begin.")
