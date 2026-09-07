import pandas as pd
INFLATION_RATE = 0.06
def get_city_base_costs(cost_df, city_name):
    """Calculates average base cost for goals in a given city."""
    city_records = cost_df[cost_df["City"].str.lower() == city_name.lower()]
    if city_records.empty:
        city_records = cost_df
    avg_costs = city_records.mean(numeric_only=True)
    return {
        "marriage": float(avg_costs["Marriage_Cost_Current"]),
        "car": float(avg_costs["Car_Cost_Current"]),
        "home": float(avg_costs["Home_Cost_Current"]),
    }
def calculate_future_cost(base_cost, years, rate=INFLATION_RATE):
    """Calculates inflation-adjusted cost using compound rate."""
    return base_cost * ((1 + rate) ** years)
def calculate_monthly_investment(future_cost, years):
    """Calculates required monthly savings needed to reach future goal cost."""
    total_months = years * 12
    return future_cost / total_months if total_months > 0 else future_cost
def assess_feasibility(required_monthly, capacity_per_goal):
    """Determines feasibility status of a goal."""
    if capacity_per_goal <= 0:
        return "Highly Challenging", "red"
    ratio = required_monthly / capacity_per_goal
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