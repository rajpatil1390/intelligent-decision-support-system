# Chapter 2: Literature Review

This chapter reviews the work on which IDSS builds: the evolution of decision support systems, automated machine
learning, LLM agents, retrieval-augmented generation, explainable AI and counterfactual explanations. It then describes
AutoML-Agent, the framework that inspired IDSS, compares related work and identifies the research gap.

## 2.1 Evolution of Decision Support Systems

Power defines a decision support system as an interactive computer-based system that helps decision makers use
communication technologies, data, documents, knowledge and models to identify and solve problems [1]. He
groups such systems by their dominant component into **data-driven**, **model-driven**, **knowledge-driven**,
**document-driven** and **communication-driven** systems [1]. Table 2.1 summarises, in broad terms, how
decision support has evolved. Each stage made more data and stronger models available, but each still kept an expert in
the loop to build the model, write the query or configure the dashboard.

**Table 2.1:** Broad stages in the evolution of decision support

| Stage | Typical tools | What the user receives | Who builds the analysis |
|---|---|---|---|
| Model-driven DSS | Spreadsheet, optimisation and simulation models | Answers from one hand-built model | Analyst |
| Data-driven DSS and BI | Data warehouses, OLAP, reports, dashboards | Descriptions of what has happened | BI developers |
| Knowledge-driven DSS | Rule-based expert systems | Rule-based recommendations | Knowledge engineers |
| Predictive analytics | Machine-learning models [2], [3], [4] | Predictions with accuracy figures | Data scientists |
| Automated ML | AutoML systems [5] | A tuned model for a specified task | Users who can specify target, metric and data |
| LLM-assisted DSS (this project) | LLM agents + deterministic AutoML + explanations | Verified model, explanations, actions and a report from a plain-language question | The system, with the user answering questions |

IDSS belongs to the last stage. It is model-driven at its core — every recommendation comes from a trained and verified
model — but it uses an LLM for the parts that used to need an expert to talk to the user: understanding the question,
asking for missing information and explaining the result.

## 2.2 Automated Machine Learning and Hyperparameter Optimisation

**Auto-sklearn** framed AutoML as combined algorithm selection and hyperparameter optimisation over scikit-learn
pipelines, using Bayesian optimisation, meta-learning to warm-start the search and an ensemble of the models found
[5]. It matches or beats manual model building on many tabular datasets, but the user must still specify the
features, target, task and metric in code, and the output is a model, not a decision.

**TPE** models the hyperparameter densities of good and bad trials separately and proposes the configuration that
maximises their ratio; it handles conditional, mixed search spaces cheaply [6]. **Optuna** offers a
*define-by-run* API, efficient samplers such as TPE and pruners that stop unpromising trials early [7]. IDSS
uses Optuna's TPE sampler with a median pruner.

The IDSS model zoo is built from **scikit-learn**, with its consistent `fit` / `predict` interface and pipelines
[2], **XGBoost**, which made gradient tree boosting scalable through a regularised objective and
sparsity-aware split finding [3], and **LightGBM**, which made it faster with histogram-based, leaf-wise
growth, gradient-based one-side sampling and exclusive feature bundling [4].

**Gap:** AutoML automates the *search*, not the *framing* of the problem or the *use* of the result.

## 2.3 Large Language Model Agents

An LLM agent uses a language model to decide the next step, calls tools and reads their results. **ReAct** interleaves
reasoning traces with actions, so the model can plan, act, observe and correct itself, which reduces hallucination
compared with reasoning alone [8]. **AutoGen** lets several conversable agents cooperate through messages, with
a human in the loop and execution of the code they write [12]. **Data Interpreter** decomposes data-science tasks
into a graph of sub-tasks and writes and executes code for each step, revising the plan when execution fails
[13].

These systems show that LLMs can drive multi-step analyses, but also the weakness that matters most for decision
support: the reported numbers come from code written during the run, which may be wrong in ways a non-expert cannot
detect. IDSS therefore lets agents *choose and explain* but not *compute*: they pick models only from a fixed, tested
zoo, and the one place where LLM-written code runs (*Ask Your Data*) is isolated in a sandbox.

## 2.4 Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) combines a generator with a non-parametric memory: a retriever fetches relevant
passages from an indexed corpus and the generator conditions on them, which improves factual accuracy and lets
knowledge be updated without retraining [14]. IDSS uses a lightweight, offline form of RAG for *planning*: ten
best-practice notes written by the team are split into chunks of about 500 characters, indexed with TF-IDF and cached;
the Knowledge Agent retrieves the four most relevant passages (at most two per note) and condenses them into a brief for
the Model Agent. Web search is used only when a key is configured, so retrieval stays deterministic, fast and
auditable.

## 2.5 Explainable AI

**LIME** explains one prediction by fitting a simple model to the black-box model's behaviour on perturbed samples
around that record [15]. **SHAP** unifies additive feature attributions under Shapley values and is the only
additive attribution with local accuracy, missingness and consistency [10]. A SHAP value *φ<sub>j</sub>* is
the contribution of feature *j* to moving one prediction away from the average prediction; the mean absolute values give
a global ranking. IDSS uses SHAP's tree explainer for tree ensembles and the linear explainer for logistic and ridge
regression (permutation importance otherwise), sums the values of derived features back to the user's original columns
— `contract_type` instead of `cat__contract_type_Month-to-month` — and orients binary explanations to the positive
class, so a positive value always means "raises the risk".

## 2.6 Counterfactual Explanations and What-if Analysis

Wachter, Mittelstadt and Russell proposed counterfactual explanations: the smallest change to the input that would
change the model's decision [11]. Unlike feature importances, counterfactuals directly suggest an action.
IDSS implements the idea twice: a **what-if simulator** that changes inputs for one record or a segment and shows the
prediction before and after, and a **counterfactual search** that finds at most three changes to actionable columns,
using values observed in the data, that bring a high-risk record below the decision threshold. As in the literature,
these statements describe the *model*, not causal effects in the real world.

## 2.7 AutoML-Agent and What IDSS Changes

**AutoML-Agent** [9] is a multi-agent LLM framework for full-pipeline AutoML. An agent manager receives the
request, a prompt agent turns it into a structured specification, retrieval-augmented planning produces candidate plans,
data and model agents carry them out, an operation agent writes and runs the code, and verification is applied to the
request, the execution and the implementation. The framework is evaluated with a success rate (SR), a normalised
performance score (NPS) and a comprehensive score (CS = 0.5 · SR + 0.5 · NPS). We read the reference implementation for
ideas only; no code was copied. Tables 2.2 and 2.3 list what we kept and what we changed.

**Table 2.2:** Ideas kept from AutoML-Agent

| Idea | In AutoML-Agent | In IDSS |
|---|---|---|
| Multi-agent decomposition | Manager with prompt, data, model and operation agents | Orchestrator with Requirement, Data, Knowledge, Model, Verifier, Analysis and Insight agents, each with typed (Pydantic) inputs and outputs |
| Structured requirement parsing | Prompt agent produces a JSON specification | Requirement Agent produces a `UserRequirement` and repairs it against the data |
| Retrieval-augmented planning | Plans informed by retrieved knowledge | Model Agent ranks candidates with a brief retrieved from best-practice notes |
| Multi-stage verification | Request, execution and implementation verified | Relevance and sufficiency checks; audited single use of the test split; seven result checks with one re-plan |
| Evaluation metrics | SR, NPS, CS | Reproduced offline; adjusted Rand index instead of Rand index for clustering (Section 7.7) |

**Table 2.3:** Design decisions changed in IDSS

| Change | Reason |
|---|---|
| The LLM only ranks names from a fixed model zoo; tested code trains, tunes, selects and evaluates | Every run that starts can finish; results are reproducible; the test split provably stays untouched; works without an LLM |
| Offline RAG over ten curated notes (TF-IDF, cached); web search only with a key | No network dependency; no data leaves the machine; auditable retrieval |
| Sandbox for the only LLM-written code (*Ask Your Data*) | LLM-generated code is untrusted input |
| Decision layer: drivers on original columns, what-if, risk tiers, counterfactual actions, grounded brief, PDF report | Decision makers need to know *what to do*, not only how accurate a model is |
| Persistent, resumable runs with a REST API and a web UI | Non-experts use a browser; runs survive restarts and can wait for answers |
| Fast / full budgets and parsed constraints (latency, training time, explainability) | Predictable runtime on a laptop; constraints are enforced, not only mentioned |

## 2.8 Comparison of Related Work

**Table 2.4:** Comparison of related work with IDSS

| Work | Approach | Strengths | Limitations | How IDSS differs |
|---|---|---|---|---|
| Model- and data-driven DSS [1] | Hand-built models, reports, dashboards | Trusted, domain-specific | Each new question needs an analyst; mostly descriptive | Builds a predictive model per plain-language question |
| Auto-sklearn [5] | Bayesian optimisation over pipelines, meta-learning, ensembles | Strong automatic search on tabular data | Problem specified in code; output is a model | Parses the question; adds verification and decisions |
| TPE / Optuna [6], [7] | Model-based hyperparameter search with pruning | Efficient, flexible | A library: objective and models must be written | Used as the tuning engine of a tested model zoo |
| ReAct [8] | Interleaved reasoning and tool use | General; grounds reasoning in tool results | No notion of splits, baselines or model quality | Fixed state machine with bounded loops; tools are tested engines |
| AutoGen [12] | Conversing agents that run code | Flexible workflows, human in the loop | Results depend on generated code; hard to bound | Typed hand-offs, ≤ 2 clarification rounds, 1 re-plan |
| Data Interpreter [13] | Task graph with code per step | End-to-end data analyses | Numbers come from code generated during the run | Numbers only from deterministic code; LLM code sandboxed |
| AutoML-Agent [9] | Multi-agent LLM framework, request to deployable model | Full pipeline, systematic verification | Generated training code can fail; stops at the model | Deterministic engine, offline fallbacks, decision layer, UI |
| SHAP / LIME [10], [15] | Post-hoc feature attributions | Well-founded explanations | Expert libraries; engineered feature names | Built in, on original columns, in plain language |

## 2.9 Research Gap

AutoML systems automate model search but need a precisely framed problem and return only a model. LLM agents understand
plain language but compute their numbers with code generated during the run. Explanation and counterfactual methods
exist but are rarely combined with model building in a tool for non-experts. We found few systems that (1) accept a
plain-language question and repair it against the data, (2) compute every number with deterministic, tested code and
work without an LLM, (3) audit their own evaluation and say when a result is not verified, (4) turn the model into
decisions with a brief whose numbers are checked against computed facts, and (5) run as a resumable web application on a
single laptop. IDSS is designed to fill this gap; its contribution is the combination and the division of labour, not a
new learning algorithm.

## 2.10 Summary

This chapter traced decision support from hand-built models to AutoML and LLM agents and reviewed RAG, SHAP, LIME and
counterfactual explanations. We described AutoML-Agent, listed the ideas we kept and the decisions we changed, compared
related work and identified the gap that IDSS addresses: a decision support system that is conversational at the front,
deterministic in its computation, self-verifying and decision-oriented.
