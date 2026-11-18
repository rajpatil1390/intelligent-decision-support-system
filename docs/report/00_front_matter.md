<!-- Front matter, in the page sequence of docs/format/Project_Report_Format_new.docx:
     First Page -> Certificate -> Acknowledgement -> Page Index -> List of Figures -> List of Tables
     -> Abbreviations / Nomenclature -> Abstract. Page Index and the two lists are generated in Step 10B from
     figures_tables_index.json. Roman page numbers up to the lists; Arabic numbering starts at the Abstract. -->

# First Page

A
**Project Report**
on

**INTELLIGENT DECISION SUPPORT SYSTEM USING GENERATIVE AI AND MACHINE LEARNING**

submitted for Partial Fulfillment of
the Requirements of the Degree
of
**Bachelor of Technology**
in
**CSE (Data Science)**
to
**G H Raisoni College of Engineering and Management, Jalgaon**

Submitted by

**Raj Vishwas Patil**
**Bhavesh Vinod Patil**
**Chaitany Rajesh Naval**
**Varun Darshan Patil**

Under the Guidance of
**Dr. Arjun Jadhaw**

**Department of Data Science**
**G H Raisoni College of Engineering and Management,**
**Jalgaon - 425002 (MS)**
**2026 – 2027**

<!-- page-break -->

# Certificate

**G H Raisoni College of Engineering and Management, Jalgaon**
**Department of Data Science**

**CERTIFICATE**

*This is to certify that the project entitled “Intelligent Decision Support System Using Generative AI and Machine
Learning”, submitted by*

**Raj Vishwas Patil**
**Bhavesh Vinod Patil**
**Chaitany Rajesh Naval**
**Varun Darshan Patil**

*in partial fulfillment of the degree of Bachelor of Technology in CSE (Data Science) has been satisfactorily carried
out under my guidance as per the requirement of G H Raisoni College of Engineering and Management, Jalgaon.*

**Date:**
**Place:** Jalgaon

| | |
|---|---|
| **Dipak Patil** | **Dr. Arjun Jadhaw** |
| HoD, Department of Data Science | Guide |

<!-- page-break -->

# Acknowledgement

We would like to express our sincere gratitude to all those who supported and guided us throughout the successful
completion of this group project titled “Intelligent Decision Support System Using Generative AI and Machine
Learning”.

First and foremost, we are deeply thankful to our project guide, Dr. Arjun Jadhaw, for the constant support, valuable
feedback and encouragement we received during every stage of the project. Dr. Jadhaw's guidance greatly enhanced the
quality of our work.

We are also grateful to Dipak Patil, Head of the Department of Data Science of G H Raisoni College of Engineering and
Management, Jalgaon, for providing us with the necessary resources and a conducive learning environment. We thank
the faculty and staff of the department for their help during the project weeks.

Finally, we would like to thank our families and friends for their constant encouragement and support.
Thank you all.

| | |
|---|---|
| | Raj Vishwas Patil |
| | Bhavesh Vinod Patil |
| | Chaitany Rajesh Naval |
| | Varun Darshan Patil |

<!-- page-break -->

# Page Index

<!-- generated in Step 10B: chapters and sections with page numbers -->

# List of Figures

<!-- generated in Step 10B from figures_tables_index.json (figures) -->

# List of Tables

<!-- generated in Step 10B from figures_tables_index.json (tables) -->

<!-- page-break -->

# Abbreviations

| Abbreviation | Full form |
|---|---|
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| ARI | Adjusted Rand Index |
| AST | Abstract Syntax Tree |
| AUC | Area Under the (ROC) Curve |
| AutoML | Automated Machine Learning |
| COCOMO | Constructive Cost Model |
| CPU | Central Processing Unit |
| CS | Comprehensive Score |
| CSV | Comma-Separated Values |
| DFD | Data Flow Diagram |
| DSS | Decision Support System |
| EM | Expectation–Maximisation |
| F1 | F1 score (harmonic mean of precision and recall) |
| FR | Functional Requirement |
| GenAI | Generative Artificial Intelligence |
| HoD | Head of Department |
| HPO | Hyperparameter Optimisation |
| HTTP | Hypertext Transfer Protocol |
| IDSS | Intelligent Decision Support System |
| JSON | JavaScript Object Notation |
| KLOC | Thousand (Kilo) Lines of Code |
| KPI | Key Performance Indicator |
| LLM | Large Language Model |
| ML | Machine Learning |
| NFR | Non-Functional Requirement |
| NPS | Normalised Performance Score |
| PDF | Portable Document Format |
| RAG | Retrieval-Augmented Generation |
| REST | Representational State Transfer |
| RI | Rand Index |
| RMSE | Root Mean Squared Error |
| RMSLE | Root Mean Squared Logarithmic Error |
| ROC | Receiver Operating Characteristic |
| SDLC | Software Development Life Cycle |
| SHAP | SHapley Additive exPlanations |
| SR | Success Rate |
| SRS | Software Requirements Specification |
| TF-IDF | Term Frequency–Inverse Document Frequency |
| TPE | Tree-structured Parzen Estimator |
| UI | User Interface |
| UML | Unified Modeling Language |

# Nomenclature

| Symbol | Meaning |
|---|---|
| *t* | Decision threshold on the predicted probability of the positive class (default 0.5) |
| *p* | Predicted probability of the positive class |
| *φ<sub>j</sub>* | SHAP value (contribution) of feature *j* to one prediction |
| *E* | Development effort in person-months (COCOMO) |
| *D* | Development time in months (COCOMO) |
| *P* | Average number of persons (*E* / *D*) |
| *k* | Number of knowledge passages retrieved for a query |
| *n* | Number of rows (records) |
| *y*, *ŷ* | True and predicted value of the target |
| *g* | Train–test gap of a 0–1 metric used by the overfitting check |

<!-- page-break -->

# Abstract

*Many organisations hold data but lack the skills to turn it into decisions. This project
presents an Intelligent Decision Support System (IDSS) that accepts a business problem in plain language together
with a tabular dataset and returns a verified machine-learning model with explained, actionable decisions. Inspired
by the AutoML-Agent framework, a resumable orchestrator coordinates seven LLM agents that interpret the request, plan,
verify and explain, while deterministic code profiles the data, tunes scikit-learn, XGBoost and LightGBM models with
Optuna, and computes SHAP explanations, what-if simulations, risk tiers and a PDF report. Every agent has an offline
fallback. A FastAPI backend and a Streamlit interface make the system usable by non-experts. On five synthetic
datasets with three seeds, every run completed (SR = 1.00), with NPS = 0.800 and CS = 0.900; the system beat a
trivial baseline on all five datasets and passed verification in 14 of 15 runs.*

**Keywords:** Decision Support System, Generative AI, Large Language Model Agents, Automated Machine Learning,
Explainable AI, SHAP, What-if Analysis
