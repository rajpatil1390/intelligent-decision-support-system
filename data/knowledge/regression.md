# Regression on tabular data

Regression predicts a number such as a price, a demand quantity or a delivery time. Start with a
regularised linear model (ridge regression) as a baseline, then try tree ensembles: random forests are
robust with little tuning, while gradient boosting (LightGBM, XGBoost) is usually the most accurate on
mixed numeric and categorical data. Always check that the model beats the simplest baseline, which
predicts the mean of the training target.

Pick the error metric that matches the business loss. RMSE penalises large errors heavily and is in the
same unit as the target. MAE is more robust to outliers and easy to explain ("on average we are off by
X"). MAPE expresses errors as percentages but breaks when the target can be zero. RMSLE suits targets
that span several orders of magnitude, such as prices of very different products, and is only defined
for non-negative values. R² is useful for reporting how much variation the model explains.

Inspect the target before modelling. Strongly skewed targets often benefit from a log transform. Check
residuals across segments: a model can look good overall while being systematically biased for one
region or product category, which matters for business decisions.

Tree models cannot extrapolate beyond the range of target values seen in training. If the future may
contain larger values than the past, for example because of inflation or growth, linear models or
models trained on ratios and differences tend to generalise better.
