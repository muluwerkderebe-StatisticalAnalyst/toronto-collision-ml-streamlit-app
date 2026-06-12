import streamlit as st
import os

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Toronto Motor Vehicle Collision Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== IMPROVED CUSTOM CSS ======================
st.markdown("""
    <style>
        /* Overall font size boost */
        html, body, [class*="css"] {
            font-size: 1.08rem !important;   /* Main body text - adjust between 1.05 to 1.15 */
        }

        /* Headings */
        h1 { font-size: 2.8rem !important; }
        h2 { font-size: 2.1rem !important; }
        h3 { font-size: 1.75rem !important; }
        h4 { font-size: 1.4rem !important; }

        /* Main content area - constrained/narrower center column
           Center the content within the available area to align equally
           between the left sidebar and the right deploy area. */
        :root{ --sidebar-width: 290px; --center-width: 900px; }
        .main .block-container {
            max-width: var(--center-width) !important;
            padding-top: 1.5rem;
            padding-left: 1.25rem;
            padding-right: 1.25rem;
            /* Center within the space excluding the sidebar */
            margin-left: calc( ( (100% - var(--sidebar-width) - var(--center-width)) / 2 ) + var(--sidebar-width) );
            margin-right: calc( (100% - var(--sidebar-width) - var(--center-width)) / 2 );
        }

        /* Sidebar improvements */
        section[data-testid="stSidebar"] {
            width: 290px !important;        /* Slightly wider sidebar if needed */
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

st.title("Toronto Motor Vehicle Collision Analysis")
st.markdown("### Predicting Number of Individuals Involved")

# Sidebar Navigation
st.sidebar.header("Navigation")
st.sidebar.markdown("""
Recommended Order:
1. Data Cleaning
2. Exploratory Analysis
3. Prediction
""")

st.sidebar.info("Use the pages menu on the left to navigate.")

# Check if processed data exists
processed_path = "data/processed/collisions_cleaned.csv"
if os.path.exists(processed_path):
    st.sidebar.success("Processed data is ready")
else:
    st.sidebar.warning("Please run Data Cleaning first")

# Main Home Page Content
st.subheader("Project Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    This application includes:
    - Data Cleaning Pipeline
    - Exploratory Data Analysis
    - Machine Learning Prediction
    """)
    # Move Target Variable under Machine Learning Prediction for clarity
    st.markdown("""
    **Target Variable:**
    - `NumberOfInvolvedPerson` — created from count of identical `ACCNUM`
    """)

with col2:
    # Reserved for additional quick links or visuals
    st.write("")

st.markdown("---")
st.write("Use the sidebar to navigate between different sections.")

# Optional: Show quick status
if st.checkbox("Show Project Status"):
    try:
        import pandas as pd
        df = pd.read_csv(processed_path)
        # Show a clear, prominent metric for dataset size
        st.metric(label="Dataset records", value=f"{df.shape[0]:,}")
        st.caption("Dataset loaded successfully")
    except:
        st.warning("Dataset not loaded yet.")