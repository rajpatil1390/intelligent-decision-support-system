# Tabular classification

Classification predicts a category for each row, such as whether a customer will churn or whether a loan
will default. On tabular business data, gradient-boosted trees (LightGBM, XGBoost) are usually the
strongest models, because they capture non-linear effects and interactions between columns without
heavy feature engineering. A regularised logistic regression should always be trained as well: it is a
fast, honest baseline, and when it performs almost as well as boosting it is often the better business
choice because its coefficients are easy to explain.

Choose the evaluation metric from the decision the model supports. Accuracy is misleading when one class
is rare. F1 balances precision and recall for the class of interest; ROC AUC measures how well the model
ranks cases independent of any cut-off; precision matters when acting on a case is expensive, recall
when missing a case is expensive.

Most classifiers output probabilities. The default cut-off of 0.5 is rarely optimal: tune the decision
threshold on a validation split for the chosen metric, never on the test split. Report results at both
the default and the tuned threshold so the trade-off is visible.

Always compare against a trivial baseline that predicts the majority class. A model that cannot beat it
on the business metric adds no value, however sophisticated it is. Keep a final test split untouched
until the very end, and use stratified splits so every class appears in train, validation and test.
