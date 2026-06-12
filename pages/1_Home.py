import streamlit as st

# ====================== PAGE CONFIG & CSS ======================
st.set_page_config(
    page_title="Home - Toronto Collision Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply same CSS styling for consistency with main app
st.markdown("""
    <style>
        /* Overall font size boost */
        html, body, [class*="css"] {
            font-size: 1.08rem !important;
        }

        /* Headings */
        h1 { font-size: 2.8rem !important; }
        h2 { font-size: 2.1rem !important; }
        h3 { font-size: 1.75rem !important; }
        h4 { font-size: 1.4rem !important; }

        /* Main content area - centered */
        :root{ --sidebar-width: 290px; --center-width: 900px; }
        .main .block-container {
            max-width: var(--center-width) !important;
            padding-top: 1.5rem;
            padding-left: 1.25rem;
            padding-right: 1.25rem;
            margin-left: calc( ( (100% - var(--sidebar-width) - var(--center-width)) / 2 ) + var(--sidebar-width) );
            margin-right: calc( (100% - var(--sidebar-width) - var(--center-width)) / 2 );
        }

        /* Sidebar improvements */
        section[data-testid="stSidebar"] {
            width: 290px !important;
        }
        
        .stSidebar .stMarkdown, .stSidebar p, .stSidebar label {
            font-size: 1.05rem !important;
        }

        /* Better line spacing and readability */
        p, li, label {
            line-height: 1.75 !important;
        }

        /* Metric containers */
        [data-testid="metric-container"] {
            font-size: 1.3rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# ====================== HOME PAGE CONTENT ======================
st.title("Toronto Motor Vehicle Collision Analysis")
st.markdown("### Predicting Number of Individuals Involved")

st.markdown("""
**Welcome to this interactive web application** designed to analyze and predict motor vehicle collisions in Toronto using real-world data.
""")

st.subheader("Project Overview")
st.markdown("""
This Streamlit application provides a complete end-to-end data science solution for Toronto Motor Vehicle Collision records. 
It enables users to explore collision patterns, understand contributing factors, and make predictions about **the number of individuals involved** in collisions.
""")

st.markdown("""
**Key Features:**
- **Data Cleaning Pipeline**: Automated cleaning and preprocessing of raw Toronto collision data
- **Exploratory Data Analysis (EDA)**: Interactive charts, temporal trends, geospatial analysis, and key insights
- **Machine Learning Models**: Comparison of XGBoost, Random Forest, and Neural Network
- **Prediction Tool**: Real-time predictions using the **best performing model (XGBoost)**
""")

st.subheader("How to Use This Application")
st.markdown("""
**Recommended Navigation Order:**
1. **Data Cleaning** — Load and clean the raw dataset
2. **Exploratory Analysis** — Discover important patterns and insights
3. **Prediction** — Make predictions using the trained XGBoost model

Use the **sidebar menu** on the left to navigate between different sections of the app.
""")

st.info("""
💡 **Tip**: For the best experience, please follow the sections in the recommended order. 
Make sure the processed data is ready before moving to the Prediction page.
""")

st.markdown("---")

# Optional Project Status
if st.checkbox("Show Project Status"):
    try:
        import pandas as pd
        df = pd.read_csv("data/processed/collisions_cleaned.csv")
        st.metric(label="Total Records in Dataset", value=f"{df.shape[0]:,}")
        st.caption(" Dataset loaded successfully")
    except:
        st.warning("Dataset not found. Please run Data Cleaning first.")
