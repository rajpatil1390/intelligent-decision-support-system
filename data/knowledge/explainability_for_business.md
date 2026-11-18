# Explainability for business users

Business users act on a model only when they understand and trust it. Explanations should answer three
questions: which factors matter overall, why a particular case received its prediction, and what could
change the outcome.

Global importance shows the main drivers across all records. SHAP values are a reliable way to compute it
for any model, including gradient boosting, because they distribute each prediction fairly among the
input features. Present the top five to ten drivers in plain language ("customers on month-to-month
contracts are much more likely to leave") rather than as raw feature names.

Local explanations show why one customer, loan or product got its score. They are essential when a
person must justify an action, such as offering a discount or declining an application.

What-if analysis lets users change one input, for example contract type or price, and see how the
prediction moves. It turns a model into a decision tool, but it shows association, not guaranteed cause
and effect, and that limitation should be stated.

If a simple model such as logistic regression performs nearly as well as a complex one, it may be the
better choice because its behaviour is transparent. When a complex model is used, validate that its main
drivers make domain sense: a model that relies on an identifier or a data-entry artefact should not be
trusted, however accurate it looks. Always report uncertainty and the size of the data behind each claim.
