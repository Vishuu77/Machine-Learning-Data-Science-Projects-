import pandas as pd
import numpy as np
import io
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import joblib

def load_sample_data(name):
    if name == "Iris":
        data = load_iris(as_frame=True)
        df = data.frame
        target = "target"
    elif name == "Wine":
        data = load_wine(as_frame=True)
        df = data.frame
        target = "target"
    elif name == "California Housing":
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        # subsample for speed
        df = df.sample(n=3000, random_state=42).reset_index(drop=True)
        target = "MedHouseVal"
    else:
        df = pd.DataFrame()
        target = ""
    return df, target

def train_ml_model(df, target_col, task_type, model_name, params, test_size=0.2):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    
    X = df.drop(columns=[target_col])
    # Drop non-numeric columns for simplicity in this demo compiler
    X = X.select_dtypes(include=[np.number])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    if task_type == "Classification":
        if model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=params.get('n_estimators', 100), max_depth=params.get('max_depth', None), random_state=42)
        elif model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_name == "SVM":
            model = SVC(random_state=42)
        else:
            model = RandomForestClassifier(random_state=42)
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            
        results = {
            "task": "Classification",
            "model": model,
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm,
            "feature_importance": feature_importance,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "features": list(X.columns)
        }
        return results
        
    else: # Regression
        if model_name == "Random Forest Regressor":
            model = RandomForestRegressor(n_estimators=params.get('n_estimators', 100), max_depth=params.get('max_depth', None), random_state=42)
        elif model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "SVR":
            model = SVR()
        else:
            model = LinearRegression()
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        elif hasattr(model, "coef_"):
            feature_importance = pd.Series(model.coef_, index=X.columns).abs().sort_values(ascending=False)
            
        results = {
            "task": "Regression",
            "model": model,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "feature_importance": feature_importance,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "features": list(X.columns)
        }
        return results

def run_python_code(code_str, df):
    # Capture stdout and matplotlib plots
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    plt.figure()
    output_msg = ""
    fig = None
    
    local_vars = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns}
    
    try:
        exec(code_str, {"__builtins__": __builtins__}, local_vars)
        output_msg = new_stdout.getvalue()
        if plt.get_fignums():
            fig = plt.gcf()
        else:
            plt.close()
    except Exception as e:
        output_msg = f"Error executing code:\n{str(e)}"
    finally:
        sys.stdout = old_stdout
        
    return output_msg, fig
