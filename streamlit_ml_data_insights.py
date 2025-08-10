
import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Try optional imports
try:
    from ydata_profiling import ProfileReport
    HAS_PROFILE = True
except Exception:
    HAS_PROFILE = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

st.set_page_config(page_title="ML Data Insights", layout="wide")

# -------------------------- Helper functions ---------------------------

def read_file(uploaded_file):
    """Read CSV/Excel uploaded file into a pandas DataFrame"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def quick_overview(df):
    st.subheader("1. Quick Overview")
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
    with c2:
        missing = df.isna().sum().sum()
        st.metric("Total Missing", int(missing))
        st.metric("Duplicate Rows", int(df.duplicated().sum()))
    with c3:
        st.write("**Sample data**")
        st.dataframe(df.head(7))


def column_type_summary(df):
    types = df.dtypes.astype(str).value_counts()
    st.write("**Column types**")
    st.table(types)


def eda_profile(df):
    st.subheader("2. Automated Profiling")
    if HAS_PROFILE:
        with st.spinner('Generating profile report (may take a while for big datasets)'):
            profile = ProfileReport(df, minimal=False)
            profile_html = profile.to_html()
            st.components.v1.html(profile_html, height=800, scrolling=True)
    else:
        st.info("Optional package 'ydata-profiling' not installed. Showing simplified EDA.")
        st.write(df.describe(include='all').T)


def show_distribution(df, col):
    fig = px.histogram(df, x=col, nbins=30, marginal='box')
    st.plotly_chart(fig, use_container_width=True)


def correlation_heatmap(df, numeric_cols):
    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns for correlation map")
        return
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)


def detect_anomalies(df, numeric_cols):
    st.subheader("3. Anomaly Detection")
    if len(numeric_cols) == 0:
        st.info("No numeric columns to run anomaly detection")
        return None
    X = df[numeric_cols].copy()
    X = X.fillna(X.median())
    iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    preds = iso.fit_predict(X)
    df['_anomaly'] = (preds == -1)
    n_anom = df['_anomaly'].sum()
    st.write(f"Detected {int(n_anom)} anomalies out of {len(df)} rows")
    if n_anom > 0:
        st.dataframe(df[df['_anomaly']].head(50))
    return df


def run_clustering(df, numeric_cols, n_clusters=3):
    st.subheader("4. Segmentation (Clustering)")
    if len(numeric_cols) == 0:
        st.info("No numeric columns for clustering")
        return df
    X = df[numeric_cols].copy().fillna(df[numeric_cols].median())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(Xs)
    df['_cluster'] = labels
    st.write("Cluster sizes:")
    st.write(pd.Series(labels).value_counts())
    # Show 2D scatter for first two numeric features
    if X.shape[1] >= 2:
        fig = px.scatter(x=Xs[:,0], y=Xs[:,1], color=labels.astype(str), labels={'x':'feat1','y':'feat2'})
        st.plotly_chart(fig, use_container_width=True)
    return df


def feature_importance(df):
    st.subheader("5. Key Driver Analysis (Feature Importance)")
    st.write("Choose a target column to analyze drivers")
    cols = list(df.columns.drop(['_anomaly','_cluster'], errors='ignore'))
    target = st.selectbox("Target (predict) column", options=[None]+cols)
    if target:
        y = df[target]
        X = df[cols].drop(columns=[target]).select_dtypes(include=[np.number]).fillna(0)
        if X.shape[1] == 0:
            st.info("No numeric predictors available for feature importance")
            return
        # Decide classifier or regressor
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            task = 'regression'
        else:
            # convert to categorical if needed
            y = y.fillna('NA')
            try:
                y_enc = pd.factorize(y)[0]
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                task = 'classification'
                y = y_enc
            except Exception:
                st.info("Could not prepare target for supervised importance analysis")
                return
        # train-test split small
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        if task == 'regression':
            st.write(f"R2 on holdout: {r2_score(y_test, preds):.3f}")
        else:
            st.write(f"Accuracy on holdout: {accuracy_score(y_test, preds):.3f}")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.bar_chart(importances.head(20))

        if HAS_SHAP:
            st.write("Computing SHAP values (may take a while)...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            try:
                st.write("SHAP summary plot")
                shap.summary_plot(shap_values, X_test, show=False)
                st.pyplot(bbox_inches='tight')
            except Exception:
                st.write("Could not render SHAP plot in this environment")


def auto_recommendations(df):
    st.subheader("6. Automated Recommendations (Rule-based)")
    recs = []
    # Example rule: if a column named 'sales' exists and has downward trend
    cols = df.columns
    if 'sales' in (c.lower() for c in cols):
        # find actual column name
        sales_col = [c for c in cols if c.lower() == 'sales'][0]
        s = df[sales_col].dropna()
        if len(s) > 3:
            # simple check: recent mean vs earlier mean
            last_mean = s.tail(max(1,int(len(s)*0.2))).mean()
            first_mean = s.head(max(1,int(len(s)*0.2))).mean()
            if last_mean < first_mean * 0.95:
                recs.append(f"Sales show a downward trend (recent mean {last_mean:.2f} < earlier mean {first_mean:.2f}). Consider promotions, channel review, or product analysis.")
    # Example rule: if anomalies detected
    if '_anomaly' in df.columns and df['_anomaly'].sum() > 0:
        recs.append(f"{int(df['_anomaly'].sum())} anomalous rows detected. Inspect those records for data quality or exceptional events.")
    # Example rule: if clusters have imbalanced sizes
    if '_cluster' in df.columns:
        sizes = df['_cluster'].value_counts(normalize=True)
        if sizes.max() > 0.9:
            recs.append("Clustering suggests one dominant segment (>90%). Try different n_clusters or additional features.")
    if len(recs) == 0:
        st.write("No immediate automated rule-based recommendations found. Consider asking the AI assistant for further suggestions.")
    else:
        for r in recs:
            st.write(f"- {r}")
    return recs


def ai_recommendations_via_llm(summary_text, openai_api_key=None):
    if not HAS_OPENAI:
        st.info("OpenAI SDK not installed. Install openai to enable LLM recommendations.")
        return None
    if not openai_api_key:
        st.info("Provide OpenAI API key in the sidebar to enable LLM recommendations")
        return None
    openai.api_key = openai_api_key
    prompt = f"You are a business data analyst. The dataset context: {summary_text}. Provide 7 actionable, prioritized recommendations and a short 2-line rationale for each."
    try:
        resp = openai.ChatCompletion.create(
            model='gpt-4o-mini',
            messages=[{'role':'user','content':prompt}],
            max_tokens=500,
            temperature=0.2
        )
        return resp['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"LLM request failed: {e}")
        return None

# -------------------------- App layout ---------------------------

def main():
    st.title("ML Data Insights — Upload CSV / Excel and get deep insights")
    st.markdown("Upload a dataset and we'll run automated EDA, anomaly detection, segmentation, and give recommendations.")

    with st.sidebar:
        st.header("Options")
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv','xlsx','xls'])
        n_clusters = st.number_input("Clusters for segmentation (K)", min_value=2, max_value=10, value=3)
        run_profile = st.checkbox("Generate detailed profiling report (ydata-profiling)")
        show_shap = st.checkbox("Attempt SHAP explainability (if available)")
        if HAS_OPENAI:
            openai_key = st.text_input("OpenAI API Key (optional)", type='password')
        else:
            openai_key = None

    if not uploaded_file:
        st.info("Upload a CSV or Excel file to get started. Sample datasets: UCI or your own.")
        st.stop()

    df = read_file(uploaded_file)
    if df is None:
        st.stop()

    # Basic overview
    quick_overview(df)
    column_type_summary(df)

    # EDA
    if run_profile and HAS_PROFILE:
        eda_profile(df)
    else:
        eda_profile(df)

    # identify numeric cols
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Distributions for top numeric columns
    st.subheader("Distributions")
    if numeric_cols:
        col = st.selectbox("Choose numeric column to view distribution", options=numeric_cols)
        show_distribution(df, col)
    else:
        st.info("No numeric columns to show distributions")

    # Correlation heatmap
    correlation_heatmap(df, numeric_cols)

    # Anomaly detection
    df = detect_anomalies(df, numeric_cols) or df

    # Clustering
    df = run_clustering(df, numeric_cols, n_clusters=n_clusters) or df

    # Feature importance
    feature_importance(df)

    # Rule-based recommendations
    recs = auto_recommendations(df)

    # LLM recommendations
    if HAS_OPENAI and openai_key:
        st.subheader("AI (LLM) Recommendations")
        # create a tiny summary
        summary_text = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}. Key numeric cols: {', '.join(numeric_cols[:5])}. Detected anomalies: {int(df['_anomaly'].sum()) if '_anomaly' in df.columns else 0}."
        ai_recs = ai_recommendations_via_llm(summary_text, openai_key)
        if ai_recs:
            st.write(ai_recs)

    # Offer download of augmented dataset
    st.subheader("Download augmented dataset")
    to_download = df.copy()
    csv = to_download.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="augmented_dataset.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

    st.info("This app is a starter template. For production: add input validation, authentication, quotas, tests, and optimize heavy operations.")

if __name__ == '__main__':
    main()
