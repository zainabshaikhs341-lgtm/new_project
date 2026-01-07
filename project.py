import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# -------------------- TITLE --------------------
st.title("🛍️ Mall Customer Segmentation Dashboard")
st.caption("Interactive customer analysis using K-Means clustering")

# -------------------- LOAD DATA --------------------
df = pd.read_csv("Mall_Customers.csv") # Changed path for deployment

df.rename(columns={
    "Annual Income (k$)": "Income",
    "Spending Score (1-100)": "SpendingScore",
    "Genre": "Gender"
}, inplace=True)

# -------------------- LOAD MODELS --------------------
scaler = joblib.load("scaler.pkl") # Changed path for deployment
kmeans = joblib.load("kmeans_model.pkl") # Changed path for deployment

# -------------------- CLUSTER PREDICTION --------------------
features = df[["Age", "Income", "SpendingScore"]]
scaled_features = scaler.transform(features)
df["Cluster"] = kmeans.predict(scaled_features)

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.header("🔎 Filter Customers")

age_range = st.sidebar.slider(
    "Age Range",
    int(df.Age.min()),
    int(df.Age.max()),
    (18, 60)
)

income_range = st.sidebar.slider(
    "Income Range (k$)",
    int(df.Income.min()),
    int(df.Income.max()),
    (20, 120)
)

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df.Gender.unique(),
    default=df.Gender.unique()
)

filtered_df = df[
    (df.Age.between(age_range[0], age_range[1])) &
    (df.Income.between(income_range[0], income_range[1])) &
    (df.Gender.isin(gender_filter))
]

# -------------------- KPI METRICS --------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", len(filtered_df))
col2.metric("Avg Age", round(filtered_df.Age.mean(), 1))
col3.metric("Avg Income (k$)", round(filtered_df.Income.mean(), 1))
col4.metric("Avg Spending Score", round(filtered_df.SpendingScore.mean(), 1))

st.divider()

# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Cluster Analysis",
    "🧠 Business Insights",
    "🎯 Predict My Segment"
])

# ==================== TAB 1 ====================
with tab1:
    st.subheader("Customer Segmentation Visualization")

    fig = px.scatter(
        filtered_df,
        x="Income",
        y="SpendingScore",
        color="Cluster",
        hover_data=["Age", "Gender"],
        title="Income vs Spending Score by Cluster"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(filtered_df.head(10), use_container_width=True)

# ==================== TAB 2 ====================
with tab2:
    st.subheader("📌 Cluster Business Interpretation")

    cluster_profiles = {
        0: "💼 High Income, Low Spending — Careful & Value-Oriented Customers",
        1: "🔥 High Income, High Spending — Premium & Loyal Customers",
        2: "💸 Low Income, High Spending — Impulsive Buyers",
        3: "📉 Low Income, Low Spending — Price Sensitive Customers",
        4: "🎯 Average Income, Average Spending — Standard Customers"
    }

    for cluster in sorted(filtered_df.Cluster.unique()):
        cluster_data = filtered_df[filtered_df.Cluster == cluster]

        with st.expander(
            f"Cluster {cluster}: {cluster_profiles.get(cluster, 'Customer Segment')}"
        ):
            st.write(f"**Customers:** {len(cluster_data)}")
            st.write(f"**Avg Age:** {cluster_data.Age.mean():.1f}")
            st.write(f"**Avg Income:** {cluster_data.Income.mean():.1f}k$")
            st.write(f"**Avg Spending Score:** {cluster_data.SpendingScore.mean():.1f}")

# ==================== TAB 3 ====================
with tab3:
    st.subheader("🎯 Predict My Customer Segment")

    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
    spending = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

    if st.button("Predict Segment"):
        user_data = pd.DataFrame(
            [[age, income, spending]],
            columns=["Age", "Income", "SpendingScore"]
        )

        user_scaled = scaler.transform(user_data)
        prediction = kmeans.predict(user_scaled)[0]

        st.success(f"✅ You belong to **Cluster {prediction}**")
        st.info(cluster_profiles.get(prediction, "Customer Segment"))

# -------------------- FOOTER --------------------
st.divider()
st.caption("🚀 Built with Streamlit | K-Means Clustering | Business Analytics")
