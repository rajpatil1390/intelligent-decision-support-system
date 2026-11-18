# Handling class imbalance

Many business targets are imbalanced: fraud, churn, equipment failure and loan default are all
minority classes. With a 5% positive rate, a model that always predicts "no" is 95% accurate and
completely useless, so accuracy should not be the headline metric. Prefer F1, recall, precision,
balanced accuracy or the area under the precision-recall curve for the minority class.

The simplest effective remedy is class weighting: give errors on the minority class a larger penalty
during training. Most libraries support it directly (class_weight="balanced" in scikit-learn and
LightGBM, scale_pos_weight in XGBoost for binary problems). Weighting usually works as well as
resampling and does not create synthetic rows, which keeps the data honest.

Threshold tuning is the second lever. A weighted or unweighted model still produces probabilities;
moving the cut-off below 0.5 trades precision for recall. Pick the threshold on the validation split
using the metric that reflects the business cost, then apply it unchanged to the test split.

Always use stratified splitting so that rare classes appear in every split, and check that the test
split contains enough minority examples to make the metric meaningful. With fewer than a few dozen
positive cases, results are noisy and should be reported with caution.

Finally, translate errors into money: a missed churner may cost the lifetime value of a customer, while
a false alarm costs only the price of a retention offer. That ratio tells you where the threshold should
sit far better than any default.
