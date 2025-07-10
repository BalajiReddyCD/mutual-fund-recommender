# app(streamlit_dashboard).py
import os
import glob
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Mutual Fund Forecast Dashboard", layout="wide")

# Load leaderboard and prediction files
def load_leaderboard():
    return pd.read_csv("outputs/models/model_leaderboard.csv")

def find_latest_file(model_name, scheme_code):
    pattern = f"outputs/models/{model_name}_preds_{scheme_code}_*.csv"
    files = glob.glob(pattern)
    if not files:
        print(f"[DEBUG] No files found for {model_name}, Scheme {scheme_code}")
        return None
    files.sort(reverse=True)
    return files[0]

def load_predictions(model_name, scheme_code):
    file_path = find_latest_file(model_name, scheme_code)
    if file_path:
        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    else:
        return pd.DataFrame()

def get_recommendations(leaderboard_path="outputs/models/model_leaderboard.csv", min_accuracy=85, max_rmse=0.5):
    df = pd.read_csv(leaderboard_path)
    grouped = df.groupby("Scheme_Code").agg({
        "Accuracy (%)": "mean",
        "RMSE": "mean",
        "Model": "nunique"
    }).rename(columns={"Model": "Model_Count"})

    recommended = grouped[
        (grouped["Accuracy (%)"] >= min_accuracy) &
        (grouped["RMSE"] <= max_rmse) &
        (grouped["Model_Count"] == 3)
    ].sort_values(by="Accuracy (%)", ascending=False)

    return recommended.reset_index()

# Sidebar filters
st.sidebar.header("Model Selection")
leaderboard = load_leaderboard()

model_options = ["All"] + sorted(leaderboard['Model'].unique().tolist())
model = st.sidebar.selectbox("Select Model", model_options)

scheme_options = ["All"] + sorted(leaderboard['Scheme_Code'].astype(str).unique().tolist())
scheme = st.sidebar.selectbox("Select Scheme Code", scheme_options)

metric = st.sidebar.selectbox("Metric to Filter", ["MAE", "RMSE", "R2", "Accuracy (%)"])
threshold = st.sidebar.slider("Minimum Accuracy (%)", 0, 100, 80)

# Filter leaderboard
filtered_df = leaderboard.copy()
if model != "All":
    filtered_df = filtered_df[filtered_df['Model'] == model]

if scheme != "All":
    filtered_df = filtered_df[filtered_df['Scheme_Code'].astype(str) == scheme]

filtered_df = filtered_df[filtered_df['Accuracy (%)'] >= threshold]

# Select top scheme per model (if model == All)
top_schemes_per_model = (
    filtered_df.loc[filtered_df.groupby("Model")["Accuracy (%)"].idxmax()]
    if model == "All" else filtered_df
)

# Tabs layout
tabs = st.tabs([
    "Leaderboard", 
    "Forecast Overlay", 
    "Model Comparison", 
    "Stationarity Check",
    "Recommendations"
])

# Tab 1: Leaderboard
with tabs[0]:
    st.subheader("Filtered Leaderboard (Top Scheme per Model)")
    st.dataframe(top_schemes_per_model, use_container_width=True)

    fig_bar = go.Figure()
    for _, row in top_schemes_per_model.iterrows():
        fig_bar.add_trace(go.Bar(
            x=[row["Model"]],
            y=[row[metric]],
            name=f"{row['Model']} (Scheme {row['Scheme_Code']})",
            text=[f"{row[metric]}"],
            hovertext=f"Scheme: {row['Scheme_Code']}<br>{metric}: {row[metric]}",
            textposition="auto"
        ))

    fig_bar.update_layout(
        title="Top Scheme Accuracy Comparison Across Models",
        xaxis_title="Model",
        yaxis_title=metric,
        barmode='group',
        showlegend=True
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Tab 2: Single model forecast overlay
with tabs[1]:
    st.subheader("Forecast Visualization")
    if model != "All" and scheme != "All":
        df_preds = load_predictions(model, scheme)

        if not df_preds.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_preds['Date'], y=df_preds['Actual_NAV'], name='Actual NAV'))
            fig.add_trace(go.Scatter(x=df_preds['Date'], y=df_preds['Predicted_NAV'], name='Predicted NAV', line=dict(dash='dash')))
            if 'Lower_CI' in df_preds.columns and 'Upper_CI' in df_preds.columns:
                fig.add_trace(go.Scatter(x=df_preds['Date'], y=df_preds['Lower_CI'], fill=None, mode='lines', name='Lower CI'))
                fig.add_trace(go.Scatter(x=df_preds['Date'], y=df_preds['Upper_CI'], fill='tonexty', mode='lines', name='Upper CI', fillcolor='rgba(0,100,80,0.2)'))

            fig.update_layout(title=f"{model} Forecast with Confidence Interval", xaxis_title="Date", yaxis_title="NAV")
            st.plotly_chart(fig, use_container_width=True)

            st.download_button("Download Predictions CSV", df_preds.to_csv(index=False).encode(), file_name=f"{model}_predictions_{scheme}.csv")
        else:
            st.warning("No predictions found for the selected model and scheme.")
    else:
        st.info("Please select both a specific model and scheme code to view forecast overlay.")

# Tab 3: All model forecast comparison
with tabs[2]:
    st.subheader(f"Forecast Overlay Comparison for Scheme {scheme}" if scheme != "All" else "Please select a scheme to compare")
    if scheme != "All":
        models = ["ARIMA", "LSTM", "Prophet"]
        overlay_fig = go.Figure()

        for m in models:
            df = load_predictions(m, scheme)
            if not df.empty:
                overlay_fig.add_trace(go.Scatter(x=df['Date'], y=df['Predicted_NAV'], mode='lines', name=f'{m} Forecast'))

        df_actual = load_predictions("ARIMA", scheme)  # Any model with 'Actual_NAV'
        if not df_actual.empty:
            overlay_fig.add_trace(go.Scatter(x=df_actual['Date'], y=df_actual['Actual_NAV'], mode='lines', name='Actual NAV', line=dict(color='black', width=2)))

        overlay_fig.update_layout(title=f"Overlay of Forecasts - Scheme {scheme}", xaxis_title="Date", yaxis_title="NAV")
        st.plotly_chart(overlay_fig, use_container_width=True)
    else:
        st.info("Please select a scheme to view model comparisons.")

# Tab 4: Stationarity Plot
with tabs[3]:
    st.subheader("ADF Stationarity Check - Top Scheme Codes")
    try:
        with open("outputs/models/adf_stationarity_plot.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600)
    except Exception as e:
        st.error(f"Error loading ADF plot: {e}")

# Tab 5: Fund Recommendations
with tabs[4]:
    st.subheader("Recommended Mutual Funds for Investment")
    recs_df = get_recommendations()

    if recs_df.empty:
        st.warning("No funds meet the recommendation criteria. Try lowering accuracy or RMSE threshold.")
    else:
        st.dataframe(recs_df, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=recs_df["Scheme_Code"].astype(str),
            y=recs_df["Accuracy (%)"],
            name="Avg Accuracy (%)",
            text=recs_df["Accuracy (%)"],
            textposition="auto"
        ))
        fig.update_layout(
            title="Recommended Funds by Accuracy",
            xaxis_title="Scheme Code",
            yaxis_title="Accuracy (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Smart Mutual Fund Forecasting Dashboard | MSc Data Science Final Project")
