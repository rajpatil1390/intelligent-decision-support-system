# Time-series forecasting

Forecasting predicts future values of a series such as daily sales or weekly demand. The most important
rule is to respect time: never shuffle the rows. Split chronologically so that the model is always
evaluated on a period that comes after everything it was trained on.

A practical approach for business series is to turn forecasting into regression with engineered
features: lagged values of the target (yesterday, one week ago, one season ago), rolling means and
standard deviations computed only from past values, and calendar features such as day of week, month,
quarter and holiday flags. Every feature for a given day must be computable using information available
before that day; otherwise the model leaks the future and looks far better than it will be in practice.

Always compare against a seasonal-naive baseline, which predicts the value from one season ago (for
daily retail data, the same weekday last week). It is surprisingly strong. A model that cannot beat it
is not worth deploying.

Tree-based models are good at seasonality and promotions but cannot extrapolate a trend beyond the
values seen in training. When a series is clearly growing, linear models on lag features, or modelling
the change rather than the level, often forecast better. Evaluate with RMSE or MAE in the units of the
series, and state the forecast horizon clearly: one-step-ahead accuracy is usually better than accuracy
several weeks ahead.
