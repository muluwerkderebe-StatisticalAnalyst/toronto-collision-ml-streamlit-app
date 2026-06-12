import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_processed_data

# Page Configuration
st.set_page_config(page_title="Exploratory Analysis", layout="wide")
st.title("Exploratory Data Analysis")
st.markdown("### Understanding Toronto Motor Vehicle Collisions")

# ====================== IMPROVED DATA LOADING ======================
@st.cache_data(ttl=60)
def get_data():
    df = load_processed_data()
    if df is None:
        st.error("Processed data not found.")
        st.markdown("""
        **Please follow these steps:**
        1. Go to the **Data Cleaning** page
        2. Click on **"Run Data Cleaning Pipeline"**
        3. Return to this page after cleaning is complete.
        """)
        st.info("Looking for file: `data/processed/collisions_cleaned.csv`")
        st.stop()
    return df

df = get_data()

# ====================== TARGET VARIABLE EXPLANATION ======================
st.subheader("Target Variable: NumberOfInvolvedPerson")

st.markdown("""
**How the target variable was generated:**

The original dataset records **individual involvements** (one row per person), not one row per collision.  

Therefore, the target variable **`NumberOfInvolvedPerson`** was created by counting how many rows share the **same `ACCNUM`** (Accident Number).

- Each unique `ACCNUM` = One collision  
- Number of rows with the same `ACCNUM` = Number of individuals involved in that collision
""")

with st.expander("See Exact Code Used to Create Target Variable"):
    st.markdown("**Exact code from your Jupyter Notebook:**")
    st.code("""
# Formula: Count of identical ACCNUM values
df['NumberOfInvolvedPerson'] = df.groupby('ACCNUM')['ACCNUM'].transform('count')
    """, language="python")
    
    st.success("This is the exact line that creates your target variable.")

# Show example
st.write("**Example: Same ACCNUM → Multiple Individuals**")
example_df = df[['ACCNUM', 'NumberOfInvolvedPerson']].sort_values('ACCNUM').head(12)
st.dataframe(example_df, use_container_width=True)

st.info(" As shown above, `NumberOfInvolvedPerson` is the count of identical `ACCNUM` values.")

# ====================== SIDEBAR FILTERS ======================
st.sidebar.header("Filters")
district_filter = st.sidebar.multiselect(
    "Select District(s)", 
    options=sorted(df['DISTRICT'].unique()),
    default=sorted(df['DISTRICT'].unique())
)

road_class_filter = st.sidebar.multiselect(
    "Select Road Class(es)", 
    options=sorted(df['ROAD_CLASS'].unique()),
    default=sorted(df['ROAD_CLASS'].unique())
)

# Apply filters
filtered_df = df[
    (df['DISTRICT'].isin(district_filter)) & 
    (df['ROAD_CLASS'].isin(road_class_filter))
]

# ====================== KEY METRICS ======================
st.subheader("Key Dataset Metrics")

col1, col2, col3, col4 = st.columns(4)

# Corrected Metrics
total_collisions = filtered_df['ACCNUM'].nunique()
fatal_incidents = filtered_df[filtered_df['FATAL_NO'] >= 1]['ACCNUM'].nunique()

col1.metric("Total Collisions", f"{total_collisions:,}")
col2.metric("Avg Individuals Involved", f"{filtered_df['NumberOfInvolvedPerson'].mean():.2f}")
col3.metric("Max Individuals in One Collision", filtered_df['NumberOfInvolvedPerson'].max())
col4.metric("Fatal Incidents", f"{fatal_incidents:,}")

# ====================== TARGET VARIABLE DISTRIBUTION ======================
st.subheader("Distribution of Number of Individuals Involved")

col1, col2 = st.columns([2, 1])

with col1:
    fig_hist = px.histogram(
        filtered_df, 
        x='NumberOfInvolvedPerson',
        nbins=30,
        title="Distribution of Number of Individuals Involved",
        color_discrete_sequence=['steelblue']
    )
    fig_hist.update_layout(bargap=0.1, height=500)
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.write("**Insights:**")
    st.write("- Most collisions involve 2–6 people")
    st.write("- Right-skewed distribution")
    st.write("- Few high-severity multi-vehicle incidents")
    
    stats = filtered_df['NumberOfInvolvedPerson'].describe()
    st.dataframe(stats, use_container_width=True)

# ====================== COLLISION CHARACTERISTICS ======================
st.subheader("Average Individuals Involved by Collision Characteristics")

col1, col2, col3 = st.columns(3)

with col1:
    fig1 = px.bar(
        filtered_df.groupby('ACCLASS')['NumberOfInvolvedPerson'].mean().reset_index(),
        x='ACCLASS', y='NumberOfInvolvedPerson',
        title="By Accident Class",
        color='NumberOfInvolvedPerson'
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.bar(
        filtered_df.groupby('IMPACTYPE')['NumberOfInvolvedPerson'].mean().reset_index().sort_values('NumberOfInvolvedPerson', ascending=False),
        x='IMPACTYPE', y='NumberOfInvolvedPerson',
        title="By Impact Type",
        color='NumberOfInvolvedPerson'
    )
    fig2.update_xaxes(tickangle=45)
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    fig3 = px.bar(
        filtered_df.groupby('INVTYPE')['NumberOfInvolvedPerson'].mean().reset_index().nlargest(10, 'NumberOfInvolvedPerson'),
        x='INVTYPE', y='NumberOfInvolvedPerson',
        title="By Involvement Type (Top 10)",
        color='NumberOfInvolvedPerson'
    )
    fig3.update_xaxes(tickangle=45)
    st.plotly_chart(fig3, use_container_width=True)

# ====================== TEMPORAL ANALYSIS ======================
st.subheader("Temporal Patterns")

col1, col2 = st.columns(2)

with col1:
    hourly = filtered_df.groupby('TIME')['NumberOfInvolvedPerson'].mean().reset_index()
    fig_hour = px.line(hourly, x='TIME', y='NumberOfInvolvedPerson',
                      title="Average Individuals Involved by Hour of Day")
    st.plotly_chart(fig_hour, use_container_width=True)

with col2:
    fig_road = px.pie(
        filtered_df, 
        names='ROAD_CLASS', 
        title="Collisions by Road Class",
        hole=0.4
    )
    st.plotly_chart(fig_road, use_container_width=True)

# ====================== ENVIRONMENTAL CONDITIONS ======================
st.subheader("Environmental & Visibility Impact")

col1, col2 = st.columns(2)

with col1:
    visibility_avg = filtered_df.groupby('VISIBILITY')['NumberOfInvolvedPerson'].mean().reset_index()
    fig_vis = px.bar(visibility_avg, x='VISIBILITY', y='NumberOfInvolvedPerson',
                    title="Average Individuals by Visibility")
    fig_vis.update_xaxes(tickangle=45)
    st.plotly_chart(fig_vis, use_container_width=True)

with col2:
    light_avg = filtered_df.groupby('LIGHT')['NumberOfInvolvedPerson'].mean().reset_index()
    fig_light = px.bar(light_avg, x='LIGHT', y='NumberOfInvolvedPerson',
                      title="Average Individuals by Light Condition")
    fig_light.update_xaxes(tickangle=45)
    st.plotly_chart(fig_light, use_container_width=True)

# ====================== DISTRICT ANALYSIS ======================
st.subheader("Collisions by District")

district_count = filtered_df['DISTRICT'].value_counts().reset_index()
district_count.columns = ['District', 'Count']

fig_district = px.bar(district_count, x='District', y='Count',
                     title="Number of Collisions by District",
                     color='Count')
st.plotly_chart(fig_district, use_container_width=True)

# ====================== CORRELATION HEATMAP ======================
st.subheader("Correlation Analysis (Numerical Features)")

numeric_cols = ['NumberOfInvolvedPerson', 'TIME', 'FATAL_NO']
corr = filtered_df[numeric_cols].corr()

fig_corr = px.imshow(
    corr, 
    text_auto=True,
    aspect="auto",
    color_continuous_scale='RdBu_r',
    title="Correlation Heatmap"
)
st.plotly_chart(fig_corr, use_container_width=True)

# ====================== RAW DATA PREVIEW ======================
with st.expander("View Sample Data"):
    st.dataframe(filtered_df.sample(10), use_container_width=True)

st.success("Exploratory Analysis Complete! Use the filters on the left to explore different segments.")