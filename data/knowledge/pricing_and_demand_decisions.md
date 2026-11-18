# Pricing and demand decisions

Price and demand models support decisions such as how much stock to order, which price to set and when to
run a promotion. They differ from pure prediction because the decision itself changes the outcome: a
lower price raises demand.

For demand forecasting, include drivers that are known in advance: calendar effects, holidays, planned
promotions, price, weather forecasts and events. Lagged demand captures momentum and seasonality. Measure
error in units that matter for operations, for example MAE in units of stock, and translate it into the
cost of overstock and stockouts, which are rarely symmetric.

For price prediction, such as crop or product prices, account for seasonality, regional differences,
supply indicators (yield, rainfall) and long-term inflation. Tree models capture interactions between
these factors well but cannot extrapolate beyond historical price ranges, so check performance on the
most recent period separately.

Elasticity, the percentage change in demand for a one percent change in price, is the key quantity for
pricing decisions. Observational data can mislead because prices are not set at random; promotions often
coincide with high-demand periods. Where possible, use controlled price tests, and treat model-based
what-if results as estimates with uncertainty.

Present decisions as scenarios: expected demand and revenue under two or three candidate prices or order
quantities, with a range rather than a single number. Decision-makers trust ranges that are honest about
uncertainty more than precise-looking point forecasts.
