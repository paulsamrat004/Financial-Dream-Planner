import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Financial Dream Planner", layout="wide")

INFLATION_RATE = 0.06


@st.cache_resource
def load_artifacts():
    model = joblib.load("best_salary_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    metrics = joblib.load("model_metrics.pkl")
    cost_df = pd.read_csv("city_goal_costs.csv")
    return model, preprocessor, metrics, cost_df


try:
    model, preprocessor, metrics, cost_df = load_artifacts()
except Exception:
    st.error(
        "Trained model files not found! Please run `python train_model.py` first."
    )
    st.stop()


def get_city_base_costs(df, city_name):
    """Calculates average base cost for marriage in a given city."""
    city_records = df[df["City"].str.lower() == city_name.lower()]
    if city_records.empty:
        city_records = df

    avg_costs = city_records.mean(numeric_only=True)
    return {"marriage": float(avg_costs["Marriage_Cost_Current"])}


def calculate_future_cost(base_cost, years, rate=INFLATION_RATE):
    """Calculates inflation-adjusted cost using compound rate."""
    return float(base_cost * ((1 + rate) ** years))


def calculate_monthly_investment(future_cost, years):
    """Calculates required monthly savings needed to reach future goal cost."""
    total_months = years * 12
    return (
        float(future_cost / total_months)
        if total_months > 0
        else float(future_cost)
    )


def assess_feasibility(required_monthly, capacity):
    """Determines feasibility status of a goal."""
    if capacity <= 0:
        return "Highly Challenging", "red"

    ratio = required_monthly / capacity
    if ratio <= 1.0:
        return "Achievable", "green"
    elif ratio <= 1.5:
        return "Challenging", "orange"
    else:
        return "Highly Challenging", "red"


def calculate_health_score(total_required, monthly_capacity):
    """Computes a Financial Health Score from 0 to 100 based on shortfall."""
    if monthly_capacity <= 0:
        return 10, "Critical Financial Gap"

    if total_required <= monthly_capacity:
        return 100, "Excellent Financial Health"

    shortfall = total_required - monthly_capacity
    shortfall_ratio = shortfall / total_required

    score = max(10, int(100 - (shortfall_ratio * 80)))

    if score >= 75:
        status = "Good Financial Health"
    elif score >= 50:
        status = "Moderate Financial Risk"
    else:
        status = "High Financial Risk"

    return score, status


# Application Header
st.title("AI-Powered Financial Dream & Goal Planner")
st.caption("Freshers Goal Planner — Salary Based Feasibility Analysis")

st.sidebar.header("Profile Details")
user_name = st.sidebar.text_input("Name")
user_age = st.sidebar.number_input("Age", min_value=18, max_value=60)
available_cities = sorted(cost_df["City"].unique().tolist())
selected_city = st.sidebar.selectbox("City", available_cities)
educations = ["B.Tech", "B.E.", "BCA", "B.Sc", "MCA", "M.Tech", "M.Sc", "MBA"]
selected_edu = st.sidebar.selectbox("Education", educations)
job_roles = [
    "Software Engineer",
    "Web Developer",
    "Data Analyst",
    "Data Scientist",
    "DevOps Engineer",
    "QA Engineer",
    "Business Analyst",
    "UI UX Designer",
    "Technical Support Engineer",
    "Project Coordinator",
]
selected_role = st.sidebar.selectbox("Job Role", job_roles)

st.sidebar.divider()
st.sidebar.header("Expected / Monthly Salary")
input_df = pd.DataFrame(
    [
        {
            "Age": user_age,
            "City": selected_city,
            "Education": selected_edu,
            "Job_Role": selected_role,
        }
    ]
)
X_input = preprocessor.transform(input_df)
ml_suggested_salary = float(model.predict(X_input)[0])

monthly_salary = st.sidebar.number_input(
    "Enter Monthly Salary (₹)",
    min_value=10000,
    max_value=500000,
    step=1000,
)

st.sidebar.divider()
st.sidebar.header("Goal Timelines (Years)")
marriage_yrs = st.sidebar.slider("Marriage Goal", 1, 15)

st.sidebar.divider()
st.sidebar.header("Savings Percentage")
savings_pct = st.sidebar.slider("Target Savings Rate (% of Salary)", 5, 80, 20)
monthly_capacity = monthly_salary * (savings_pct / 100.0)

# Calculations for Marriage Goal Only
base_costs = get_city_base_costs(cost_df, selected_city)
future_m_cost = calculate_future_cost(base_costs["marriage"], marriage_yrs)
req_m_monthly = calculate_monthly_investment(future_m_cost, marriage_yrs)

total_req_monthly = req_m_monthly
monthly_shortfall = total_req_monthly - monthly_capacity

# Metrics Display
col1, col2, col3, col4 = st.columns(4)
col1.metric("User Monthly Salary", f"₹{monthly_salary:,.2f}")
col2.metric("Monthly Savings Capacity", f"₹{monthly_capacity:,.2f}")
col3.metric("Total Goal Requirement", f"₹{total_req_monthly:,.2f}")

if monthly_shortfall > 0:
    col4.metric(
        "Monthly Gap",
        f"-₹{monthly_shortfall:,.2f}",
        delta_color="inverse",
    )
else:
    col4.metric(
        "Monthly Surplus",
        f"+₹{abs(monthly_shortfall):,.2f}",
        delta_color="normal",
    )

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Goal Progress & Feasibility",
        "6% Inflation Comparison",
        "Financial Health Score",
        "Model Comparison (LR vs DT)",
    ]
)

with tab1:
    display_name = user_name if user_name else "User"
    st.subheader(f"Goal Plan for {display_name} ({selected_city})")

    # Full monthly capacity dedicated to Marriage Goal (100% capacity)
    m_status, m_color = assess_feasibility(req_m_monthly, monthly_capacity)

    st.markdown(
        f"**Marriage Goal** ({marriage_yrs} Yrs) — Required: **₹{req_m_monthly:,.2f}/mo** | Status: :{m_color}[**{m_status}**]"
    )
    
    # Progress ratio capped between 0.0 and 1.0
    progress_val = (
        min(1.0, max(0.0, float(monthly_capacity / req_m_monthly)))
        if req_m_monthly > 0
        else 1.0
    )
    st.progress(progress_val)

    st.subheader("Feasibility Recommendation")
    if monthly_shortfall > 0:
        st.warning(
            f"You have a monthly shortfall of **₹{monthly_shortfall:,.2f}**. "
            "To bridge this gap: increase savings percentage slider, extend goal timelines, or adjust salary target."
        )
    else:
        st.success(
            f"Great job! Your budget has a monthly surplus of **₹{abs(monthly_shortfall):,.2f}**. All planned goals are achievable."
        )

with tab2:
    st.subheader("Impact of 6% Annual Inflation: Today vs Future Cost")
    comparison_data = pd.DataFrame(
        {
            "Goal": ["Marriage"],
            "Timeline": [f"{marriage_yrs} Years"],
            "Today's Base Cost": [base_costs["marriage"]],
            "Future Inflation Cost (6%)": [future_m_cost],
        }
    )
    st.table(
        comparison_data.style.format(
            {
                "Today's Base Cost": "₹{:,.2f}",
                "Future Inflation Cost (6%)": "₹{:,.2f}",
            }
        )
    )
    chart_df = comparison_data.set_index("Goal")[
        ["Today's Base Cost", "Future Inflation Cost (6%)"]
    ]
    st.bar_chart(chart_df)

with tab3:
    st.subheader("Financial Health Score Assessment")
    health_score, score_status = calculate_health_score(
        total_req_monthly, monthly_capacity
    )
    col_s1, col_s2 = st.columns([1, 3])
    col_s1.metric("Financial Health Score", f"{health_score} / 100")
    col_s2.subheader(f"Status: {score_status}")
    st.progress(health_score / 100.0)
    st.divider()
    st.markdown("### Score Breakdown Guidance:")
    st.write(
        "• **80 - 100**: Low risk. Savings capacity comfortably covers goal targets."
    )
    st.write(
        "• **50 - 79**: Moderate risk. Shortfall exists; adjustments recommended."
    )
    st.write(
        "• **0 - 49**: High risk. High financial pressure; timeline extension needed."
    )

with tab4:
    st.subheader("Machine Learning Model Comparison")
    comparison_metrics = pd.DataFrame(
        {
            "Metric": ["$R^2$ Score (Accuracy)", "Mean Absolute Error (MAE)"],
            "Linear Regression": [
                f"{metrics['lr_r2']:.4f}",
                f"₹{metrics['lr_mae']:,.2f}",
            ],
            "Decision Tree Regressor": [
                f"{metrics['dt_r2']:.4f}",
                f"₹{metrics['dt_mae']:,.2f}",
            ],
        }
    )
    st.table(comparison_metrics)
    st.info(
        f"Selected ML Model: **{metrics['selected']}** (Automated reference suggestion: ₹{ml_suggested_salary:,.2f})"
    )