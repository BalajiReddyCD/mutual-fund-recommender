# app.py – Streamlit Dashboard for Mutual Fund Forecasting and Recommendation

import sys
import os

# Add src folder to sys.path (absolute import fix)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import api  # this should now work

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# Set Streamlit page configuration
st.set_page_config(page_title="Mutual Fund Forecast Comparison", layout="wide")
st.title(" Mutual Fund Forecast Model Comparison Dashboard")

# Enable dark theme and highlight headers
st.markdown("""
    <style>
    body {
        color: white;
        background-color: #0E1117;
    }
    .stApp {
        background-color: #0E1117;
    }
    .css-1d391kg, .css-1kyxreq {
        background-color: #1C1F26;
    }
    h1, h2, h3, h4 {
        color: #00B4D8;
    }
    .stSelectbox > div, .stMultiselect > div {
        background-color: #1C1F26;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load leaderboard
leaderboard_df = api.load_leaderboard()
if leaderboard_df is not None:
    st.subheader(" Model Leaderboard")

    scheme_codes = api.get_scheme_codes(leaderboard_df)
    default_scheme = str(api.get_best_scheme_code(leaderboard_df))
    all_option = "All"
    scheme_filter = st.selectbox(" Select Scheme Code (Best by default or choose any)",
                                 options=[default_scheme] + [s for s in scheme_codes if s != default_scheme] + [all_option],
                                 index=0)

    model_filter = st.multiselect(" Select Models to Compare",
                                  options=api.get_model_list(leaderboard_df),
                                  default=None)

    filtered_df = api.filter_leaderboard(leaderboard_df, scheme_code=scheme_filter, model_names=model_filter)

    # Summary cards for best models
    st.markdown("### Top Performing Models")
    top_models = leaderboard_df.sort_values("RMSE").head(3)
    for _, row in top_models.iterrows():
        st.info(f"**{row['Model']}** (Scheme: {row['Scheme_Code']}) → RMSE: {row['RMSE']} | MAE: {row['MAE']} | R²: {row['R2']}")

    # Monthly NAV summaries
    if not filtered_df.empty:
        for _, row in filtered_df.iterrows():
            nav_df = api.load_predictions(str(row['Scheme_Code']), row['Model'])
            if nav_df is not None:
                nav_summary = api.get_monthly_nav_summary(nav_df)
                st.markdown(f"### Monthly NAV Summary ({row['Model']} - {row['Scheme_Code']})")
                st.dataframe(nav_summary, use_container_width=True)

    st.dataframe(filtered_df.sort_values("RMSE").reset_index(drop=True), use_container_width=True)

    st.download_button(
        label=" Download Leaderboard CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name="filtered_model_leaderboard.csv",
        mime="text/csv"
    )
else:
    st.warning("Leaderboard file not found. Please run the models first.")

# Model file prediction viewer
model_files = api.get_model_prediction_files()

if model_files:
    selected_file = st.selectbox(" Select a Model Prediction File to View", model_files)
    df = api.load_prediction_file(selected_file)
    if df is not None:
        st.subheader(f" Prediction Results from {selected_file}")

        # Date Range Filter
        df['Date'] = pd.to_datetime(df['Date'])
        min_date = df['Date'].min().to_pydatetime()
        max_date = df['Date'].max().to_pydatetime()
        date_range = st.slider(" Filter Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
        df_filtered = df[(df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])]

        with st.expander(" Raw Prediction Table"):
            st.dataframe(df_filtered, use_container_width=True)

        fig = px.line(df_filtered, x='Date', y=["Actual_NAV", "Predicted_NAV"],
                      title=" Actual vs Predicted NAV", markers=True)
        fig.update_layout(legend=dict(x=0, y=1), xaxis_title="Date", yaxis_title="NAV")
        st.plotly_chart(fig, use_container_width=True)

        # Metrics block
        st.markdown("###  Metrics Insight")
        try:
            mae, rmse, r2 = api.calculate_metrics(df_filtered)
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", mae)
            col2.metric("RMSE", rmse)
            col3.metric("R²", r2)

            st.download_button(
                label="📥 Download Predictions CSV",
                data=df_filtered.to_csv(index=False).encode('utf-8'),
                file_name=selected_file,
                mime="text/csv"
            )
        except Exception as e:
            st.warning(f" Error calculating metrics: {e}")
else:
    st.info(" No prediction files found. Please generate model predictions to visualize.")
