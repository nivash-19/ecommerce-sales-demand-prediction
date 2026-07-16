"""
E-Commerce Sales Analysis & Demand Prediction
===============================================
Author: Nivash
Role   : Data Science Intern
Purpose: End-to-end analysis of historical e-commerce sales data and
         development of machine learning models (Linear Regression &
         Random Forest Regressor) to forecast product demand (Quantity)
         for inventory optimization.

Dataset: "Sample Superstore" dataset (9,994 orders, US retail superstore)
         Contains Order Date, Product Name, Category, Sales, Quantity,
         Region, Profit, Discount, Segment, etc.

Run:    python ecommerce_sales_analysis.py
Output: All charts saved to ../visuals/, metrics saved to ../reports/metrics.json
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# 0. CONFIGURATION
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "superstore.csv")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"

RANDOM_STATE = 42


def save_fig(fig, name):
    """Utility to save a matplotlib figure to the visuals directory."""
    path = os.path.join(VISUALS_DIR, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [saved] {name}.png")


# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH, encoding="latin1")
print(f"Raw shape: {df.shape}")
print(f"Columns  : {list(df.columns)}")

# --------------------------------------------------------------------------
# 2. DATA CLEANING
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: DATA CLEANING")
print("=" * 70)

# 2.1 Standardise column names (strip spaces)
df.columns = [c.strip() for c in df.columns]

# 2.2 Missing values -------------------------------------------------------
missing_before = df.isnull().sum().sum()
print(f"Missing values before cleaning: {missing_before}")

# Numeric columns -> fill with median; Categorical -> fill with mode
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

for c in num_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median())

for c in cat_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].mode()[0])

print(f"Missing values after cleaning : {df.isnull().sum().sum()}")

# 2.3 Duplicates ------------------------------------------------------------
dupes = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Duplicate rows removed: {dupes}")

# 2.4 Date conversion --------------------------------------------------------
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y", errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y", errors="coerce")
df = df.dropna(subset=["Order Date"])  # cannot use rows without a valid order date
print(f"Rows after date validation: {df.shape[0]}")

# 2.5 Outlier detection & treatment (IQR method on Sales & Quantity) --------
def cap_outliers_iqr(series, factor=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return series.clip(lower=lower, upper=upper), lower, upper

before_sales_max = df["Sales"].max()
df["Sales_capped"], s_low, s_high = cap_outliers_iqr(df["Sales"])
n_capped_sales = (df["Sales"] != df["Sales_capped"]).sum()
print(f"Sales outliers capped (IQR method): {n_capped_sales} rows "
      f"(bounds: {s_low:.2f} - {s_high:.2f}); original max={before_sales_max:.2f}")
# We retain the ORIGINAL Sales column for business reporting (true revenue),
# and use the capped version only where extreme leverage would distort EDA/ML.
df["Sales"] = df["Sales"]  # keep original for real revenue figures

# 2.6 Date-based feature engineering ---------------------------------------
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Day"] = df["Order Date"].dt.day
df["Quarter"] = df["Order Date"].dt.quarter
df["DayOfWeek"] = df["Order Date"].dt.day_name()
df["MonthName"] = df["Order Date"].dt.month_name()
df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)

print("Feature engineering complete: Year, Month, Day, Quarter, DayOfWeek added.")
print(df[["Order Date", "Year", "Month", "Day", "Quarter", "DayOfWeek"]].head())

# Save the cleaned dataset for reproducibility
clean_path = os.path.join(BASE_DIR, "data", "superstore_cleaned.csv")
df.to_csv(clean_path, index=False)
print(f"Cleaned dataset saved to: {clean_path}")

# --------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# 3.1 Overall sales trend over time (daily, resampled monthly for clarity)
ts = df.set_index("Order Date").resample("MS")["Sales"].sum()
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(ts.index, ts.values, color="#2E86AB", linewidth=2)
ax.fill_between(ts.index, ts.values, alpha=0.15, color="#2E86AB")
ax.set_title("Overall Sales Trend Over Time (Monthly)", fontsize=14, weight="bold")
ax.set_xlabel("Order Date")
ax.set_ylabel("Total Sales ($)")
save_fig(fig, "01_sales_trend_over_time")

# 3.2 Monthly sales trend (seasonality across calendar months, all years combined)
monthly = df.groupby("Month")["Sales"].sum().reindex(range(1, 13))
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=monthly.index, y=monthly.values, ax=ax, palette="Blues_d")
ax.set_title("Monthly Sales Trend (Aggregated Across All Years)", fontsize=14, weight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Total Sales ($)")
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
save_fig(fig, "02_monthly_sales_trend")

# 3.3 Top-selling products (by revenue)
top_products = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_products.values, y=[p[:40] + "..." if len(p) > 40 else p for p in top_products.index],
            ax=ax, palette="viridis")
ax.set_title("Top 10 Best-Selling Products by Revenue", fontsize=14, weight="bold")
ax.set_xlabel("Total Sales ($)")
ax.set_ylabel("")
save_fig(fig, "03_top_selling_products")

# 3.4 Top-selling categories
cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.barplot(x=cat_sales.index, y=cat_sales.values, ax=ax[0], palette="crest")
ax[0].set_title("Total Sales by Category", weight="bold")
ax[0].set_ylabel("Total Sales ($)")
ax[1].pie(cat_sales.values, labels=cat_sales.index, autopct="%1.1f%%",
          colors=sns.color_palette("crest", len(cat_sales)), startangle=90)
ax[1].set_title("Category Share of Total Sales", weight="bold")
fig.suptitle("Top-Selling Categories", fontsize=14, weight="bold")
save_fig(fig, "04_top_selling_categories")

# 3.5 Region-wise sales
region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=region_sales.index, y=region_sales.values, ax=ax, palette="magma")
ax.set_title("Region-wise Total Sales", fontsize=14, weight="bold")
ax.set_ylabel("Total Sales ($)")
save_fig(fig, "05_region_wise_sales")

# 3.6 Region-wise quantity sold
region_qty = df.groupby("Region")["Quantity"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=region_qty.index, y=region_qty.values, ax=ax, palette="rocket")
ax.set_title("Region-wise Total Quantity Sold", fontsize=14, weight="bold")
ax.set_ylabel("Units Sold")
save_fig(fig, "06_region_wise_quantity")

# 3.7 Seasonal demand pattern (quantity by month & quarter)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
qty_month = df.groupby("Month")["Quantity"].sum().reindex(range(1, 13))
sns.lineplot(x=qty_month.index, y=qty_month.values, marker="o", ax=ax[0], color="#D62246")
ax[0].set_title("Quantity Demand by Month", weight="bold")
ax[0].set_xticks(range(1, 13))
ax[0].set_xlabel("Month")
ax[0].set_ylabel("Units Sold")

qty_quarter = df.groupby("Quarter")["Quantity"].sum()
sns.barplot(x=qty_quarter.index, y=qty_quarter.values, ax=ax[1], palette="flare")
ax[1].set_title("Quantity Demand by Quarter", weight="bold")
ax[1].set_xlabel("Quarter")
ax[1].set_ylabel("Units Sold")
fig.suptitle("Seasonal Demand Patterns", fontsize=14, weight="bold")
save_fig(fig, "07_seasonal_demand_patterns")

# 3.8 Correlation heatmap
num_df = df[["Sales", "Quantity", "Discount", "Profit", "Year", "Month", "Quarter"]]
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Heatmap of Numeric Features", fontsize=14, weight="bold")
save_fig(fig, "08_correlation_heatmap")

# 3.9 Sales distribution
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(df["Sales"], bins=60, kde=True, ax=ax, color="#3A86FF")
ax.set_xlim(0, df["Sales"].quantile(0.99))
ax.set_title("Distribution of Sales (clipped at 99th percentile for readability)", fontsize=13, weight="bold")
ax.set_xlabel("Sales ($)")
save_fig(fig, "09_sales_distribution")

# 3.10 Quantity distribution
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(df["Quantity"], bins=14, kde=False, ax=ax, color="#FB5607", discrete=True)
ax.set_title("Distribution of Order Quantity", fontsize=13, weight="bold")
ax.set_xlabel("Quantity per Order Line")
save_fig(fig, "10_quantity_distribution")

# 3.11 Additional insight: Sub-category performance (Sales vs Profit)
subcat = df.groupby("Sub-Category").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).sort_values("Sales", ascending=False)
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(subcat))
width = 0.4
ax.bar(x - width/2, subcat["Sales"], width, label="Sales", color="#457B9D")
ax.bar(x + width/2, subcat["Profit"], width, label="Profit", color="#E63946")
ax.set_xticks(x)
ax.set_xticklabels(subcat.index, rotation=45, ha="right")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Sales vs Profit by Sub-Category", fontsize=14, weight="bold")
ax.legend()
save_fig(fig, "11_subcategory_sales_vs_profit")

# 3.12 Additional insight: Discount vs Profit relationship
fig, ax = plt.subplots(figsize=(9, 5))
sns.scatterplot(data=df.sample(min(2000, len(df)), random_state=RANDOM_STATE),
                 x="Discount", y="Profit", hue="Category", alpha=0.6, ax=ax)
ax.set_title("Discount vs Profit (sampled orders)", fontsize=14, weight="bold")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
save_fig(fig, "12_discount_vs_profit")

# 3.13 Additional insight: Segment-wise sales
seg_sales = df.groupby("Segment")["Sales"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=seg_sales.index, y=seg_sales.values, ax=ax, palette="Set2")
ax.set_title("Sales by Customer Segment", fontsize=14, weight="bold")
ax.set_ylabel("Total Sales ($)")
save_fig(fig, "13_segment_wise_sales")

print(f"\nAll {13} EDA visualizations saved to: {VISUALS_DIR}")

# --------------------------------------------------------------------------
# 4. MACHINE LEARNING — DEMAND (QUANTITY) PREDICTION
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: MACHINE LEARNING - DEMAND PREDICTION")
print("=" * 70)

# 4.1 Build a daily/product-category level demand table.
# Business framing: inventory planners need to forecast UNIT DEMAND
# (Quantity) for a given Category/Region/time period — not raw invoice
# lines. We aggregate to (Category, Sub-Category, Region, Year, Month)
# granularity, which mirrors how a replenishment planner would consume
# this model's output.
agg = df.groupby(
    ["Category", "Sub-Category", "Region", "Segment", "Year", "Month", "Quarter"]
).agg(
    Quantity=("Quantity", "sum"),
    Sales=("Sales", "sum"),
    Discount=("Discount", "mean"),
    Profit=("Profit", "sum"),
    Orders=("Order ID", "nunique"),
).reset_index()

print(f"Aggregated modeling table shape: {agg.shape}")

# 4.2 Feature engineering for ML
agg["AvgSalesPerOrder"] = agg["Sales"] / agg["Orders"]
agg["ProfitMargin"] = agg["Profit"] / agg["Sales"].replace(0, np.nan)
agg["ProfitMargin"] = agg["ProfitMargin"].fillna(0)
agg["MonthSin"] = np.sin(2 * np.pi * agg["Month"] / 12)
agg["MonthCos"] = np.cos(2 * np.pi * agg["Month"] / 12)

# Encode categoricals
le_dict = {}
for col in ["Category", "Sub-Category", "Region", "Segment"]:
    le = LabelEncoder()
    agg[col + "_enc"] = le.fit_transform(agg[col])
    le_dict[col] = le

feature_cols = [
    "Category_enc", "Sub-Category_enc", "Region_enc", "Segment_enc",
    "Year", "Quarter", "MonthSin", "MonthCos",
    "Sales", "Discount", "Profit", "Orders", "AvgSalesPerOrder", "ProfitMargin",
]
target_col = "Quantity"

X = agg[feature_cols]
y = agg[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# 4.3 Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# 4.4 Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=300, max_depth=12, min_samples_leaf=2,
    random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 4.5 Evaluation
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} Performance:")
    print(f"  MAE  : {mae:.3f}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  R2   : {r2:.3f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

results = {}
results["Linear Regression"] = evaluate(y_test, lr_pred, "Linear Regression")
results["Random Forest"] = evaluate(y_test, rf_pred, "Random Forest Regressor")

# 4.6 Model comparison chart
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
metrics_names = ["MAE", "RMSE", "R2"]
for i, m in enumerate(metrics_names):
    vals = [results["Linear Regression"][m], results["Random Forest"][m]]
    sns.barplot(x=["Linear Regression", "Random Forest"], y=vals, ax=ax[i],
                palette=["#8D99AE", "#EF233C"])
    ax[i].set_title(m, weight="bold")
fig.suptitle("Model Comparison: Linear Regression vs Random Forest", fontsize=14, weight="bold")
save_fig(fig, "14_model_comparison")

# 4.7 Predicted vs Actual (Random Forest — best model)
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, rf_pred, alpha=0.4, color="#EF233C", s=25)
lims = [0, max(y_test.max(), rf_pred.max()) * 1.05]
ax.plot(lims, lims, "k--", linewidth=1.5, label="Perfect Prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual Quantity Demand")
ax.set_ylabel("Predicted Quantity Demand")
ax.set_title("Random Forest: Predicted vs Actual Demand", fontsize=14, weight="bold")
ax.legend()
save_fig(fig, "15_predicted_vs_actual_rf")

# 4.8 Feature importance (Random Forest)
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(x=importances.values, y=importances.index, ax=ax, palette="crest")
ax.set_title("Random Forest — Feature Importance for Demand Prediction", fontsize=13, weight="bold")
ax.set_xlabel("Importance")
save_fig(fig, "16_feature_importance_rf")

print("\nTop 5 most important features for demand prediction:")
print(importances.head(5))

# --------------------------------------------------------------------------
# 5. SAVE METRICS & KEY BUSINESS AGGREGATES FOR REPORT/README GENERATION
# --------------------------------------------------------------------------
summary = {
    "dataset_rows": int(df.shape[0]),
    "date_range": [str(df["Order Date"].min().date()), str(df["Order Date"].max().date())],
    "total_sales": float(df["Sales"].sum()),
    "total_profit": float(df["Profit"].sum()),
    "total_orders": int(df["Order ID"].nunique()),
    "model_results": results,
    "top_5_products_by_sales": top_products.head(5).round(2).to_dict(),
    "category_sales": cat_sales.round(2).to_dict(),
    "region_sales": region_sales.round(2).to_dict(),
    "region_quantity": region_qty.to_dict(),
    "top_5_feature_importance": importances.head(5).round(4).to_dict(),
    "worst_5_products_by_sales": df.groupby("Product Name")["Sales"].sum().sort_values().head(5).round(2).to_dict(),
    "best_profit_subcategory": subcat["Profit"].idxmax(),
    "worst_profit_subcategory": subcat["Profit"].idxmin(),
}

with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nMetrics & summary saved to: {os.path.join(REPORTS_DIR, 'metrics.json')}")
print("\n" + "=" * 70)
print("PROJECT EXECUTION COMPLETE")
print("=" * 70)
