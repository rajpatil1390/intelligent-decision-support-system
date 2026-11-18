# Choosing models by data size

The amount of data should shape both the choice of models and how much tuning is worthwhile.

Small datasets (a few hundred rows): variance is the main risk. Prefer simple, regularised models such
as logistic or ridge regression and shallow tree ensembles. Use cross-validation or repeated splits
rather than a single small validation split, keep the number of features low, and expect test metrics to
fluctuate by several points between splits. Large hyperparameter searches mostly fit noise.

Medium datasets (thousands to hundreds of thousands of rows): gradient-boosted trees usually lead,
with random forests as a robust alternative. A modest Optuna search of 15 to 40 trials per model is
typically enough; most of the gain comes from learning rate, tree size and regularisation. Always keep
a linear baseline for comparison and explanation.

Large datasets (millions of rows): training time and memory dominate. LightGBM and histogram-based
gradient boosting scale well; algorithms that need pairwise distances, such as agglomerative clustering,
must be run on samples. Tune on a subsample first, prune unpromising settings early, and refit the best
configuration on all data.

Whatever the size, the number of rows in the rarest class matters as much as the total. A dataset of
100,000 rows with only 150 positives behaves like a small dataset for that class. Respect the time
budget: a slightly less accurate model delivered on time is usually worth more than a perfect model that
arrives too late.
