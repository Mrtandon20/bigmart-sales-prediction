# BigMart Sales Prediction — Multiple Linear Regression

Predicting `Item_Outlet_Sales` for product–store combinations from the BigMart retail
dataset, using a Multiple Linear Regression model built end to end: data cleaning,
exploratory analysis, feature engineering, modelling, and diagnostics.

## Problem statement

BigMart wants to understand which product and outlet properties most influence sales,
so they can predict sales for any given product–store combination. This project frames
that as a supervised regression task and uses linear regression as an interpretable baseline.

## Dataset

`Sales_Data.csv` — 8,523 rows × 12 columns.

| Column | Description |
| --- | --- |
| Item_Identifier | Unique product ID |
| Item_Weight | Product weight |
| Item_Fat_Content | Low Fat / Regular |
| Item_Visibility | Share of display area given to the product |
| Item_Type | Product category |
| Item_MRP | Maximum retail price |
| Outlet_Identifier | Unique store ID |
| Outlet_Establishment_Year | Year the store opened |
| Outlet_Size | Small / Medium / High |
| Outlet_Location_Type | Tier 1 / 2 / 3 |
| Outlet_Type | Grocery Store / Supermarket Type 1–3 |
| **Item_Outlet_Sales** | **Target** — sales of the product at the store |

## Workflow

- **A. Data inspection** — shape, types, summary stats, missing values, duplicates, cardinality.
- **B. Cleaning** — item-level imputation for `Item_Weight`; group-mode imputation for `Outlet_Size`; standardised `Item_Fat_Content` labels; treated zero `Item_Visibility` as disguised-missing.
- **C. EDA** — target distribution and skew, numeric scatter plots, categorical boxplots, correlation heatmap (each with a written conclusion).
- **D. Feature engineering** — created `Outlet_Age`, dropped ID columns, one-hot encoded categoricals (`drop_first=True`).
- **E. Model training** — 80/20 split (`random_state=42`), `LinearRegression`.
- **F. Evaluation** — R², MAE, MSE, RMSE, plus bonus diagnostics: scaled coefficients, residual analysis, log-transform test, 5-fold cross-validation.
- **G. Summary & next steps.**

## Key findings

- `Outlet_Type` is the strongest predictor — Supermarket Type3 outlets sell far more than Grocery Stores.
- `Item_MRP` is the second-strongest predictor once coefficients are scaled.
- `Item_Weight` and `Item_Visibility` add almost no predictive value.

## Results

| Metric | Value |
| --- | --- |
| R² (test) | ≈ 0.58 |
| MAE | ≈ ₹793 |
| 5-fold CV R² | ≈ 0.56 (std ≈ 0.01) |

The result is stable across folds. A log-transform of the target improved MAE but not R²,
and residuals show heteroscedasticity (errors widen for high-sales predictions) — consistent
with real-world drivers (promotions, competition, seasonality) not present in the data.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook BigMart_Sales_Prediction.ipynb
```

Then run **Kernel → Restart & Run All**. Keep `Sales_Data.csv` in the same folder as the notebook.

## Next steps

- Wrap preprocessing in a scikit-learn `Pipeline` / `ColumnTransformer` to eliminate the
  data leakage noted in the notebook (imputation statistics should be learned from the
  training split only).
- Try tree-based models (Random Forest, Gradient Boosting), which typically outperform
  linear regression on this dataset.

## Tech stack

Python · pandas · NumPy · Matplotlib · seaborn · scikit-learn
