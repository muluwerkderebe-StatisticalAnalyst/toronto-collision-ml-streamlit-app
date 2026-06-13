# 🚗 Toronto Motor Vehicle Collision Analysis & Prediction

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-5A5A5A?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

> **Live Demo:** [🚗 Toronto Collision Predictor →](https://toronto-collision-ml-app-app-6jtcxmwbo3hy9trevtuydp.streamlit.app/)

---

## 📋 Project Overview

This interactive **Streamlit web application** delivers a complete end-to-end data science workflow for analyzing **Toronto Motor Vehicle Collisions**.

The application enables users to:
- 🧹 Perform data cleaning and preprocessing on raw collision records
- 📊 Conduct in-depth Exploratory Data Analysis with interactive visualizations
- 🤖 Train and compare multiple machine learning models
- 🎯 Make real-time predictions on **the number of individuals involved** in collisions using the **best-performing model (XGBoost)**

| Detail | Info |
|---|---|
| **Dataset** | 18,957 Toronto collision records |
| **Target Variable** | `NumberOfInvolvedPerson` (derived from count of identical `ACCNUM`) |
| **Course** | CST2216 Individual Term Project |
| **Program** | Business Intelligence Systems Infrastructure (BISI) |
| **Institution** | Algonquin College, Ottawa |

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧹 **Data Cleaning Pipeline** | Automated handling of missing values, outliers, and feature engineering (temporal & geospatial) |
| 📊 **Exploratory Data Analysis** | Interactive charts showing temporal trends, geospatial patterns, and key risk factors |
| 🤖 **ML Modeling** | Built, trained, and compared XGBoost, Random Forest, and Neural Network regression models |
| 🎯 **Prediction Dashboard** | User-friendly interface for real-time predictions with the deployed XGBoost model |
| 🎨 **Professional UI/UX** | Clean, responsive design with optimized typography, centered layout, and improved readability |
| 🏗️ **Modular Code** | Well-structured, documented, and production-ready |

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Core Framework** | Streamlit (multi-page capable) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Deployment** | Streamlit Community Cloud |
| **Other** | Python 3, Git |

---

## 📊 Dataset Information

- **Source:** [Open Data Toronto – Toronto Police Service Motor Vehicle Collision Data](https://open.toronto.ca/)
- **Records:** 18,957 collision incidents
- **Key Features:** Temporal (date/time), geospatial, environmental, and vehicle-related variables
- **Target:** `NumberOfInvolvedPerson`

---

## 🎯 Machine Learning Approach

1. **Preprocessing & Feature Engineering** — PCA, sliding windows, temporal encoding, outlier handling
2. **Model Training** — Three regression models built and evaluated:

| Model | Notes |
|---|---|
|  **XGBoost** | Best-performing — selected for deployment |
| Random Forest | Strong baseline |
| Neural Network | Deep learning comparison |

3. **Evaluation** — Models compared on standard regression metrics (MAE, RMSE, R²)
4. **Deployment** — XGBoost model saved and served live via Streamlit

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/toronto-collision-involvement-prediction.git
cd toronto-collision-involvement-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

---

## 👤 Author

**Muluwerk Derebe**  
*Business Intelligence Systems Infrastructure (BISI) *  
Algonquin College, Ottawa, Ontario, Canada

> **Background:** Former Assistant Professor of Statistics with multiple peer-reviewed publications in advanced statistical modeling, longitudinal data analysis, and geospatial methods. Now transitioning into Data Analytics and Business Intelligence roles.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muluwerk-derebe-b959243a6/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/muluwerkderebe-StatisticalAnalyst)
[![Resume](https://img.shields.io/badge/Resume-View-green?style=flat)](https://muluwerkderebe-statisticalanalyst.github.io/Portifolio.io/)

---

## 📄 License

This project is open for **educational and portfolio purposes**. Feel free to explore the code for learning.

---

<div align="center">

Made with ❤️ to promote data-driven insights for improving road safety in Toronto.

</div>
