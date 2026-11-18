# Chapter 1: Introduction

Organisations make many decisions that depend on data: which customers to contact before they leave, how much stock
to order, what price to expect for a crop, or which shipments will arrive late. The data is often already available in
spreadsheets and databases; what is missing is the expertise to turn it into a reliable model, and the model into a
clear recommendation. This project, the **Intelligent Decision Support System (IDSS)**, addresses that gap. A user
describes a business problem in plain language and uploads a table of data; a team of Large Language Model (LLM)
agents turns the request into a validated machine-learning pipeline, deterministic code trains and verifies the models,
and the system returns explained, actionable decisions together with a PDF report.

## 1.1 Background and Motivation

### 1.1.1 Background

**Decision support systems.** A decision support system (DSS) is a computer-based system that helps people use data,
models and knowledge to solve semi-structured decision problems [1]. Classical model-driven DSS needed an
analyst to build each model by hand, and most dashboards describe what *has* happened rather than predict what *will*
happen or suggest what to do.

**Machine learning and AutoML.** Libraries such as scikit-learn [2], XGBoost [3] and LightGBM
[4] make strong predictive models available to everyone, but a trustworthy model still requires many expert
decisions: the target and metric, cleaning and splitting, leakage prevention, model choice and tuning, and a fair test
on unseen data. AutoML systems such as Auto-sklearn [5] automate model selection and hyperparameter
optimisation, often with the Tree-structured Parzen Estimator (TPE) [6] as implemented in Optuna
[7]. They still expect a precisely specified problem and stop at a trained model.

**LLM agents.** Large language models can read a request in everyday language, ask clarifying questions, plan
multi-step work and explain results; agent frameworks combine an LLM with tools in a loop of reasoning and acting
[8]. AutoML-Agent by Trirat, Jeong and Hwang [9] showed that a team of specialised LLM agents can take
a natural-language request all the way to a deployable model, using retrieval-augmented planning and multi-stage
verification. IDSS is inspired by this work. We did not invent the idea of a multi-agent LLM framework for AutoML; our
contribution is a different division of labour between the LLM and deterministic code, and a decision layer on top of
the model (Section 2.7).

### 1.1.2 Motivation

1. **The skills gap.** Small businesses, retailers and logistics teams can state their question ("Which customers are
   likely to leave, and what should we do?") but rarely employ data scientists to answer it.
2. **Accuracy is not a decision.** A good F1 score does not tell a manager what to do. Decision makers need the key
   drivers, the effect of possible actions, a list of whom to act on first and an honest statement of how far to trust
   the result. Explanation methods such as SHAP [10] and counterfactual explanations [11] provide
   these answers but are seldom packaged for non-experts.
3. **LLMs are good with language and weak with numbers.** An LLM that writes its own training code can produce scripts
   that fail, leak test data or report numbers that were never computed — unacceptable in a decision support tool.

These points led to the guiding rule of IDSS: **the LLM plans and explains, deterministic code computes.** Agents
interpret the request, rank models from a fixed list, review results and write the brief; profiling, training,
evaluation, explanation and simulation are done by tested Python code, and every agent has a deterministic fallback, so
the whole pipeline also runs without an LLM.

## 1.2 Problem Definition

Given a tabular dataset *D* (CSV, Excel, JSON or Parquet, up to 50 MB) and a business question *q* in plain language,
build a system that:

1. decides whether *q* can be answered from *D* and asks short clarifying questions when information is missing;
2. converts *q* into a validated requirement — task type, target column, metrics and constraints — consistent with *D*;
3. prepares the data, searches and tunes candidate models within a time budget and evaluates the chosen model exactly
   once on a held-out test split;
4. verifies the result against the requirement and simple baselines, re-planning once if a check fails;
5. explains the model, simulates what-if scenarios and derives risk tiers, actions or a forecast;
6. writes a decision brief whose numbers are all grounded in computed facts, and a PDF / Markdown report; and
7. offers all of this through a web interface with resumable runs, never sending raw rows, file paths or keys to the
   LLM.

## 1.3 Scope and Objectives

### 1.3.1 Scope

**In scope:** one table per analysis (≤ 50 MB); binary and multi-class classification, regression, single-series
forecasting and clustering; a fixed model zoo built on scikit-learn, XGBoost and LightGBM tuned with Optuna; SHAP
explanations, what-if simulation, risk tiers and counterfactual actions; an optional LLM through any OpenAI-compatible
provider or a local Ollama model, and a fully offline mode; a FastAPI backend, an SQLite index and an eight-page
Streamlit interface on one machine, listening on localhost only.

**Out of scope:** images, audio or free text as the main signal; joins across tables; multi-series forecasting; deep
learning; multi-user accounts; deploying models as production services. Chapter 8 returns to these limits.

### 1.3.2 Objectives

1. To design a multi-agent architecture in which LLM agents interpret, plan, review and explain while deterministic
   engines perform every computation, with an offline fallback for each agent.
2. To turn a plain-language question into a structured requirement, repaired against the data, with clarifying
   questions when information is missing.
3. To build an AutoML engine that searches a fixed model zoo within budgets and constraints, tunes the decision
   threshold for binary classification and reads the test split exactly once.
4. To verify every model with rule-based checks and re-plan once when a check fails.
5. To provide decision support: key drivers, what-if simulation, risk tiers, counterfactual actions, forecasts, a
   grounded decision brief and a PDF report.
6. To make the system usable by non-experts through a REST API and a web interface with resumable runs.
7. To keep data private and the system safe: aggregates only to the LLM, keys only in `.env`, LLM-written code only in a
   sandbox, bounded loops and retries.
8. To evaluate the system honestly with the SR, NPS and CS scores of AutoML-Agent [9], an ablation and a
   robustness study, and to test it with an automated test suite.

## 1.4 Selection of Life Cycle Model for Development

Table 1.1 compares four software development life cycle (SDLC) models against the needs of the project.

**Table 1.1:** Comparison of life cycle models for IDSS

| Model | Main idea | Fit for IDSS |
|---|---|---|
| Waterfall | Each phase is finished before the next begins | Poor: model quality and LLM behaviour can only be judged on running code, which would come late |
| Spiral | Repeated cycles driven by risk analysis | Partial: useful risk focus, but heavy for an eight-week student project |
| Agile / Scrum | Short sprints, backlog, customer reviews | Partial: we used weekly cycles but had no external product owner or daily stand-ups |
| **Iterative and incremental** | Build the system in increments, each adding a working, tested layer | **Selected**: matches the layered architecture and gives working software every week |

We selected the **iterative and incremental model with one increment per week**:

- **Layered architecture.** Foundation → data → requirements and orchestration → AutoML → verification → decision
  layer → API and UI → evaluation; each layer depends only on the layers below it.
- **Working software every week.** Each increment ended with a runnable result (a demo script, a benchmark, an offline
  end-to-end run or UI pages), automated tests and a Git commit, so integration problems surfaced in the same week.
- **Learning from measurements.** For example, in a single demo run (seed 42) the week-4 benchmark recorded an F1 score
  of 0.558 on the churn data at the default 0.5 cut-off, and the decision-threshold tuning added in week 5 raised it to
  0.623 on the same split; over the three evaluation seeds churn F1 rises from 0.561 to 0.649 (Section 7.6).
- **Fixed duration.** Eight weeks map directly to eight increments (Table 3.3).

Each increment followed the same cycle: plan the layer, design its interfaces as Pydantic schemas, implement, test, run
the full suite, demonstrate and commit.

## 1.5 Organization of Report

Chapter 2 reviews decision support systems, AutoML, LLM agents, retrieval-augmented generation and explainable AI, and
identifies the research gap. Chapter 3 covers feasibility, risks, scheduling, effort allocation and a COCOMO cost
estimate. Chapter 4 specifies the requirements and presents the architecture, data flow diagrams, UML diagrams and the
database. Chapter 5 gives the algorithms, the modules, code extracts and the user interface. Chapter 6 describes testing,
Chapter 7 the results and their discussion, and Chapter 8 concludes with future scope. References, publications and two
appendices (a short user manual and the weekly progress) follow.

## 1.6 Summary

This chapter introduced IDSS, which turns a plain-language business question and a data table into a verified model
and explained decisions. The motivation — a skills gap, the need for decisions rather than only accuracy, and the
weakness of LLMs with numbers — led to the rule "the LLM plans and explains, deterministic code computes". We defined
the problem, the scope and eight objectives, and justified the iterative and incremental life cycle with one tested
increment per week.
