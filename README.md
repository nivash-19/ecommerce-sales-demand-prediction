# E-Commerce Sales Analysis & Demand Prediction

**Internship Project — Data Science**
Author: Nivash | B.Tech CSE (AI & ML), VIT Chennai

## 1. Objective

Analyze historical e-commerce sales data and build machine learning models to
predict future product demand for **inventory optimization**. The project
covers the full data science lifecycle: data acquisition, cleaning,
exploratory data analysis (EDA), predictive modeling, evaluation, and
business recommendation.

## 2. Dataset

**Source:** [Sample Superstore dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
(a widely-used public retail dataset, originally distributed by Tableau /
mirrored on Kaggle).

| Property | Value |
|---|---|
| Rows | 9,994 order line items |
| Time span | Jan 2014 – Dec 2017 |
| Columns | 21 (Order Date, Ship Date, Category, Sub-Category, Product Name, Sales, Quantity, Discount, Profit, Region, Segment, etc.) |
| Missing values | 0 (verified) |
| Duplicates | 0 (verified) |

This dataset was selected because it contains every field required by the
project brief (Order Date, Product Name, Category, Sales, Quantity, Region)
plus additional fields (Profit, Discount, Segment) that enable richer,
business-relevant feature engineering for the demand model.

## 3. Project Structure

```
ecommerce_project/
├── data/
│   ├── superstore.csv              # Raw dataset
│   └── superstore_cleaned.csv      # Cleaned dataset (generated on run)
├── notebooks/
│   └── ecommerce_sales_analysis.ipynb   # Fully commented notebook (main deliverable)
├── scripts/
│   └── ecommerce_sales_analysis.py      # Equivalent standalone script
├── visuals/
│   └── 01_....png – 16_....png     # All generated charts (16 total)
├── reports/
│   ├── metrics.json                # Machine-readable model & business metrics
│   ├── Project_Report.docx         # 15–20 page written report
│   └── Presentation.pptx           # 10–12 slide summary deck
├── requirements.txt
└── README.md
```

## 4. How to Run

```bash
# 1. Create environment & install dependencies
pip install -r requirements.txt

# 2. Run the standalone script (generates all charts + metrics.json)
python scripts/ecommerce_sales_analysis.py

# 3. Or open the notebook for the fully annotated, cell-by-cell walkthrough
jupyter notebook notebooks/ecommerce_sales_analysis.ipynb
```

Outputs (charts, cleaned CSV, metrics.json) are written to `visuals/`,
`data/`, and `reports/` respectively.

## 5. Methodology

1. **Data Cleaning** — missing-value imputation (median/mode), duplicate
   removal, date parsing, IQR-based outlier capping on `Sales`, and creation
   of `Year`, `Month`, `Day`, `Quarter`, `DayOfWeek` features.
2. **EDA** — 13 visualizations covering time trends, product/category/region
   performance, seasonality, correlations, and distributions.
3. **Feature Engineering for ML** — the transactional data is aggregated to
   a `(Category, Sub-Category, Region, Segment, Year, Month)` grain, which
   mirrors how an inventory planner actually consumes a demand forecast
   (per product line, per region, per month) rather than predicting a
   single invoice line's quantity in isolation.
4. **Modeling** — Linear Regression (baseline, interpretable) and Random
   Forest Regressor (non-linear, handles interactions) are trained to
   predict aggregated unit demand (`Quantity`).
5. **Evaluation** — MAE, RMSE, and R² on a held-out 20% test split.

## 6. Key Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 2.61 | 3.50 | 0.762 |
| **Random Forest Regressor** | **2.47** | **3.44** | **0.770** |

Random Forest outperforms Linear Regression on every metric, confirming that
demand has non-linear, interaction-driven patterns (e.g., discount level ×
category × season) that a linear model cannot fully capture.

The most influential predictor of aggregated unit demand is the number of
distinct orders placed in that Category/Region/month, followed by profit,
profit margin, and average sales per order — confirming that demand
forecasting should be grounded in order-volume trends rather than revenue
alone.

## 7. Business Insights (Summary)

- **Best category by revenue:** Technology ($836K), closely trailed by
  Furniture and Office Supplies.
- **Best region:** West ($725K in sales, 12,266 units) — outperforms every
  other region in both revenue and volume.
- **Weakest region:** South, both in revenue and unit volume — a candidate
  for a focused growth/marketing initiative.
- **Most profitable sub-category:** Copiers. **Least profitable:** Tables
  (frequently sold at a loss after discounting).
- **Seasonality:** Demand consistently peaks in Q4 (Nov–Dec), suggesting a
  need for pre-holiday inventory build-up.

Full details, charts, and recommendations are in `reports/Project_Report.docx`.

## 8. Future Improvements

- Incorporate external variables (holidays, marketing spend, promotions,
  competitor pricing) to improve forecast accuracy.
- Move from month-level aggregation to weekly granularity for tighter
  inventory cycles.
- Experiment with gradient boosting (XGBoost/LightGBM) and time-series
  models (SARIMA, Prophet) for comparison against Random Forest.
- Deploy the trained model behind a lightweight API for integration into
  an inventory management system.

## 9. Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Jupyter
