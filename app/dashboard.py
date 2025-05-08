# app/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import page modules
from dashboard_pages.overview import show_overview_page
from dashboard_pages.customer_insights import show_customer_insights_page
from dashboard_pages.prediction_tool import show_prediction_tool_page
from dashboard_pages.model_performance import show_model_performance_page

# Import utilities
from utils.churn_utils import get_model_path, get_data_path, load_model_components

# Set page config
st.set_page_config(
    page_title="Telecom Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load CSS
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("style.css not found. Using default styling.")

# Load CSS
load_css()

# Load model and data
@st.cache_resource
def init_resources():
    try:
        model, scaler, expected_columns, thresholds = load_model_components()
        model_loaded = model is not None and scaler is not None and expected_columns is not None
        return model, scaler, expected_columns, thresholds, model_loaded
    except Exception as e:
        st.error(f"Error initializing resources: {str(e)}")
        return None, None, None, {'default': 0.5, 'f1_threshold': 0.5, 'business_threshold': 0.5}, False

@st.cache_data
def load_data():
    try:
        # Try multiple possible locations for the dataset
        possible_paths = [
            # Current directory path
            os.path.join('data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
            # Parent directory path
            os.path.join('..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
            # Absolute path relative to dashboard.py
            os.path.join(os.path.dirname(__file__), '..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
            # From parent directory of dashboard.py
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
            # App subdirectory
            os.path.join('app', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
        ]

        df = None
        # Try each path until one works
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break

        if df is None:
            # If no path worked, check if we have a saved path
            if hasattr(st.session_state, 'data_path') and st.session_state.data_path:
                if os.path.exists(st.session_state.data_path):
                    df = pd.read_csv(st.session_state.data_path)
                    st.sidebar.success(f"Loaded dataset from: {st.session_state.data_path}")

        if df is None:
            # If still no data, show error
            st.error("Could not find the dataset. Try using the file browser in the sidebar.")
            return None

        # Fix TotalCharges column - convert empty strings to NaN and then to numeric
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        # Convert target to Yes/No if it's 1/0
        if 'Churn' in df.columns and df['Churn'].dtype != 'object':
            df['Churn'] = df['Churn'].map({1: 'Yes', 0: 'No'})

        return df
    except Exception as e:
        st.error(f"Error loading the dataset: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Initialize resources - NOT indented inside load_data() function
model, scaler, expected_columns, thresholds, model_loaded = init_resources()
df = load_data()

# Dashboard title
st.markdown('<div class="main-header">Telecom Customer Churn Dashboard</div>', unsafe_allow_html=True)
st.markdown("This dashboard helps analyze and predict customer churn for a telecom company.")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Customer Insights", "Prediction Tool", "Model Performance"])

# Add filters to sidebar
st.sidebar.title("Data Filters")
with st.sidebar.expander("Filter Options", expanded=False):
    if df is not None:
        # Contract filter
        contract_filter = st.multiselect(
            "Contract Type",
            options=sorted(df['Contract'].unique()),
            default=sorted(df['Contract'].unique())
        )

        # Internet service filter
        internet_filter = st.multiselect(
            "Internet Service",
            options=sorted(df['InternetService'].unique()),
            default=sorted(df['InternetService'].unique())
        )

        # Apply filters
        if contract_filter and internet_filter:
            filtered_df = df[
                (df['Contract'].isin(contract_filter)) &
                (df['InternetService'].isin(internet_filter))
                ]
            st.write(f"Showing {len(filtered_df):,} of {len(df):,} customers")
        else:
            filtered_df = df
            st.write("No filters applied")
    else:
        filtered_df = None
        st.write("No data available to filter")

# Debug mode checkbox
debug_mode = st.sidebar.checkbox("Debug Mode")
if debug_mode:
    st.sidebar.subheader("Debug Information")
    st.sidebar.write(f"Model loaded: {model_loaded}")
    st.sidebar.write(f"Model path: {get_model_path()}")
    st.sidebar.write(f"Data path: {get_data_path()}")
    if thresholds:
        st.sidebar.write("Thresholds:")
        st.sidebar.write(f"- Default: {thresholds.get('default', 0.5)}")
        st.sidebar.write(f"- F1 Optimal: {thresholds.get('f1_threshold', 0.5)}")
        st.sidebar.write(f"- Business Optimal: {thresholds.get('business_threshold', 0.5)}")

# Display the selected page
if page == "Overview":
    show_overview_page(df, filtered_df)
elif page == "Customer Insights":
    show_customer_insights_page(df, filtered_df)
elif page == "Prediction Tool":
    show_prediction_tool_page(model, scaler, expected_columns, thresholds, model_loaded, df)
elif page == "Model Performance":
    show_model_performance_page(model, scaler, expected_columns, thresholds, model_loaded, df)

# Footer
st.markdown('<div class="footer">Telecom Customer Churn Analysis Dashboard - Created by Jenna Chiang</div>',
            unsafe_allow_html=True)