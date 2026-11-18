# Feature leakage

Leakage happens when a model is trained with information that will not be available at prediction time.
It produces excellent validation scores and disappointing real-world results, and it is one of the most
common reasons machine-learning projects fail after deployment.

Typical sources of leakage: columns recorded after the outcome (a "cancellation reason" column when
predicting churn, or "days overdue" when predicting default); identifiers or timestamps that encode the
outcome by accident; aggregates computed over the whole dataset, including the test period; and
preprocessing, such as scaling or imputation, fitted on all data before splitting.

Warning signs: a single feature that is almost perfectly correlated with the target, a model that is far
better than expected for the problem, or a validation score much higher than results on genuinely new
data. Treat any feature with a correlation above about 0.98 with the target as suspicious and ask how
and when it is recorded.

Prevention: split the data first and fit every preprocessing step on the training split only, ideally
inside a single pipeline. For time-ordered data, split chronologically and build lag and rolling
features only from earlier rows. Keep a test split untouched until the final evaluation and evaluate on
it once. Drop identifier columns from the features. When in doubt, remove the suspicious column and
retrain: if performance collapses to a realistic level, the column was probably leaking.
