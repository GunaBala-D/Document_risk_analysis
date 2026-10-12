# Interview Preparation — Document Risk Engine

Everything here refers to what is actually in this repository and what was actually measured. All numbers
come from synthetic data — **say so every time you quote one.** Interviewers trust a candidate who states
the limits of their evidence far more than one who quotes big numbers.

---

## The 60-second explanation

> "Procurement and compliance teams review a lot of vendor contracts by hand, and it's slow and inconsistent.
> I built a system that takes a contract — PDF, Word or a scan — extracts the commercial terms like value,
> payment days and termination notice, and tells the analyst whether it's risky, why, and what to do.
>
> The risk signal comes from three independent parts: a LightGBM classifier I picked over logistic
> regression, random forest and XGBoost through cross-validation; an anomaly detector that compares the
> contract to the vendor's own history; and a deterministic rule engine for policy requirements like
> 'every contract needs a termination clause'. Those combine into a 0-to-100 score.
>
> For explainability, SHAP shows which factors drove the model, and a local LLM through Ollama turns that
> structured evidence into a readable summary — it never decides risk, and the system works without it.
>
> Analysts confirm or correct the result, a second person validates that feedback, and only validated
> labels go into retraining. Monitoring separates data drift, prediction drift and performance drift, and a
> new model only reaches production through MLflow if it passes a quality gate. In my demo the gate rejected
> seven of nine candidates — including one that looked great on recall but was actually useless.
>
> The data is synthetic, so the numbers are about the engineering, not real-world accuracy — on the
> synthetic test set the production model reached 0.87 macro-F1 and 95% recall on high-risk contracts."

Structure: Problem → Solution → ML → Risk engine → Explainability → LLM → Human feedback → MLOps. Practise it
until it takes 60 seconds without sounding read.

---

## Resume bullets (only implemented technologies, only measured numbers)

- Built an end-to-end contract risk engine (FastAPI, PostgreSQL, React/TypeScript) that extracts terms from
  PDF/DOCX/scanned contracts (PyMuPDF, Tesseract OCR), combines a LightGBM classifier, Isolation-Forest
  anomaly detection and a 15-rule YAML compliance engine into an explainable 0–100 risk score with SHAP and a
  grounded local-LLM (Ollama) summary.
- Ran a cross-validated comparison of 4 model families × 3 class-imbalance strategies with threshold tuning on
  out-of-fold predictions; the selected model reached 0.867 macro-F1 and 0.946 high-risk recall on a held-out
  synthetic test set (threshold tuning raised high-risk recall from 0.873 to 0.946 at equal F1).
- Implemented MLOps with MLflow tracking and a model registry (staging/production aliases) behind a
  multi-metric quality gate, feedback-driven retraining with four-eyes label validation, and PSI/KS drift
  monitoring; the gate rejected 7 of 9 candidate models, including corrupted ones that a recall-only check
  would have passed.
- Delivered 136 automated tests (unit, ML-contract, API, PostgreSQL integration), Docker Compose packaging and
  a GitHub Actions pipeline (lint, tests, frontend build, image builds).

If asked about the numbers: "synthetic corpus of 3,000 generated contracts with a known labeling policy."

---

## Questions and answers

### 1. Why XGBoost? / Why LightGBM?
I didn't pick a model up front. I compared logistic regression, random forest, XGBoost and LightGBM under
the same 5-fold stratified CV, each with no rebalancing, class weights and SMOTE, then tuned each family.
The selection rule was written down before the results: among models that meet the recall constraints, take
the best CV macro-F1, and if two are within 0.01 prefer the simpler one. LightGBM (SMOTE) won at 0.852
CV macro-F1 versus 0.835 for XGBoost, so it was selected. Honest detail: on the test set XGBoost scored
slightly higher (0.873 vs 0.867) — I didn't switch, because choosing on the test set would make the reported
numbers optimistic. Earlier, Random Forest beat XGBoost via the simplicity tie-break, which is also a good
answer to "why not XGBoost": it wasn't measurably better.

### 2. Why not logistic regression?
It was my baseline and it's competitive on recall — it actually has the highest CRITICAL recall in CV
(0.848) — but its macro-F1 was about 6 points lower (0.788). The label is a sum of threshold-based policy
factors (payment > 60 days, penalty > 10%...), and a linear model can't represent thresholds without me
hand-crafting indicator features. Adding those indicators (feature set fv2) closed part of the gap, which I
measured. Trees learn thresholds and interactions directly.

### 3. Why wasn't accuracy the primary metric?
CRITICAL is 7.4% of contracts. A model that never predicts CRITICAL can still be ~90% accurate. The business
cost is asymmetric — missing a high-risk contract is far worse than an extra review — so selection used
macro-F1 with recall constraints, and the gate checks macro-F1, macro PR-AUC, HIGH+CRITICAL recall and
CRITICAL recall. Accuracy is reported for reference only.

### 4. How did you handle class imbalance?
Experimentally, not by default. Three strategies per family: none, class weights (sample weights for XGBoost),
and SMOTE inside the CV pipeline (so synthetic samples never leak into validation folds). SMOTE helped the
tree models most. The bigger lever turned out to be **decision thresholds**: I tune per-class decision
weights — predict `argmax(p × w)` — on out-of-fold probabilities to maximise macro-F1 subject to the recall
constraints. On the test set that raised high-risk recall from 0.873 to 0.946 at the same macro-F1. There's
also a separate escalation threshold on P(HIGH)+P(CRITICAL), tuned for 95% recall, that routes documents to
mandatory review.

### 5. How did you create features from unstructured documents?
Pipeline: validated file → text (PyMuPDF; Tesseract for pages without a text layer) → cleaning → section
detection from headings with a clause-language fallback → entity extraction with regex scoped to the right
section (so a 30-day termination notice is never read as payment terms) → 29 features in a versioned
registry: contract terms, policy indicators, text signals (missing clauses, risky keywords), and
**point-in-time vendor history** — only contracts dated before the analysis date, to avoid leakage. The same
code builds features for training and serving. I measured extraction against generated ground truth:
99.5–100% per field, lower on OCR (customer name 96.2%).

### 6. Why use Isolation Forest?
Classification asks "does this look like contracts labelled risky?"; anomaly detection asks "is this unusual?"
without labels, so it can catch patterns the labeling policy never anticipated. Isolation Forest is fast,
works on mixed-scale tabular data without distribution assumptions, and SHAP can explain it. I combined it
with simple vendor-specific checks (robust z-score of value vs that vendor's history, payment terms longer than
ever before, contract bursts), because those give precise statements like "6× this vendor's median".

### 7. How does Isolation Forest work?
It builds random trees by picking a random feature and a random split value. Anomalies are few and different,
so they get isolated in fewer splits — a shorter average path length across trees means more anomalous. I
convert the raw score into a percentile of the training distribution (0.98 = more isolated than 98% of
training contracts) and flag above the 97th percentile.

### 8. Why separate compliance rules from ML?
A policy is a hard requirement: "no termination clause → violation" must fire 100% of the time, not "usually".
Policies also change by decision — moving a limit from 60 to 45 days is a YAML edit, not a retraining cycle —
and every violation must trace to a written rule for audit. My retraining demo proves the point: when I
simulated a policy change through reviewer labels, retrained candidates couldn't learn it from 39 examples and
the gate rejected them. The right fix for a policy change is to change the rule.

### 9. Why not use an LLM as the risk classifier?
Not reproducible enough for an audit trail, hard to evaluate and calibrate, can hallucinate facts, slow
(30–45 s per explanation on my laptop vs ~200 ms for the whole ML pipeline), and gives no clean feature
attributions. The structured approach gives measured metrics, SHAP, deterministic rules and a quality gate.
The LLM does what it's good at — explaining the evidence in plain language.

### 10. How does SHAP work?
It comes from Shapley values in game theory: a feature's contribution is its average marginal effect on the
prediction over all orders in which features could be added. The contributions plus a base value add up
exactly to the model's output for that prediction. TreeExplainer computes this exactly and efficiently for
tree ensembles. I have a test that checks this additivity against the real model output.

### 11. How does local SHAP differ from global SHAP?
Local explains one prediction: which features pushed *this* contract towards CRITICAL (e.g. "Payment terms
above hard limit: yes +1.17"). Global aggregates mean |SHAP| over many contracts to show what the model relies
on overall (top: vendor violation rate, missing mandatory clauses, termination clause). In a multi-class model
I explain the predicted class; LightGBM's SHAP values are in log-odds.

### 12. How did you select the risk-score weights?
I started with the brief's 40/25/25/10 as an assumption and then validated them: on the 600 test contracts I
computed each component and evaluated 1,771 weight combinations. The defaults beat ~75% of combinations on
ranking (Spearman 0.906, AUC 0.974), but the level macro-F1 was only 0.577, and the ML component alone matched
the labels best. I explain that as circular — the labels come from the policy the ML model learned — while the
composite intentionally adds anomaly and exposure signals the labels don't measure. I kept the weights
(changing them is a policy decision, and fitting them on the test set would overfit) and recommended
recalibrating the level thresholds on reviewer-validated data.

### 13. How do you prevent hallucinations from Ollama?
Five layers: (1) it only receives structured evidence, never contract text; (2) a system instruction forbidding
invented facts; (3) JSON-schema-constrained output at temperature 0.1; (4) a post-check that drops
explanations of factors not in the evidence, flags numbers that don't appear in the evidence, and flags
clause-presence contradictions; (5) recommended actions are deterministic, so the LLM only rephrases. Real
story: the first version said "the contract includes a termination clause" for a contract without one. The
root cause was my evidence format (`"Termination clause present"` with value `0`), so I made every factor
self-describing (`Termination clause: missing`) and added the contradiction check plus a regression test.

### 14. What happens if Ollama goes down?
Nothing important. The explanation runs as a background task after analysis; if Ollama times out or fails,
the status becomes `unavailable` and the risk score, rules, anomalies, SHAP and actions are all already
there. There's an API test for exactly this, and `/health` reports the LLM as down while the API stays healthy.

### 15. How does human feedback improve the model?
Reviewers confirm or correct the level with a comment. Feedback isn't trusted automatically: consistency flags
(e.g. labelled LOW despite a CRITICAL violation) are raised, and a different person with the risk-manager role
must validate it — you can't validate your own. Only validated feedback, with the features stored at analysis
time, is added to the training data. Then the candidate still has to pass the gate.

### 16. How do you decide when to retrain?
Not on drift alone — retraining on unlabeled drifted data teaches nothing. Rule: at least 50 new validated
labels AND (dataset-level drift OR measured degradation), or degradation with at least 30 labels. In my demo,
drifted traffic produced feature and prediction drift but no retraining; only after validated feedback showed
macro-F1 dropping to 0.39 did it trigger.

### 17. What is data drift?
A change in the input distribution P(X) — e.g. contracts with longer payment terms or bigger values than at
training time. I measure it per feature with PSI against the training reference plus a KS test (numeric) or
chi-square (binary), and call drift only when PSI ≥ 0.2 *and* p < 0.01 — requiring both avoided false alarms on
small samples in my baseline batch. Data drift doesn't necessarily mean the model is wrong.

### 18. What is concept drift?
A change in P(y|X): the same contract should now get a different answer — e.g. a new treasury policy makes 50
payment days unacceptable. You can't see it in the inputs; you need labels. I detect it as degradation of the
model on validated reviewer labels (macro-F1 0.39 vs 0.87 in the demo).

### 19. How do you monitor model performance?
Four separate views: data drift, prediction drift (PSI of predicted classes vs the training reference),
performance on validated feedback against the model's test baseline, and service health (p50/p95 latency,
error rate and volume from request middleware). Monitoring reports are persisted so trends can be compared.

### 20. How do you decide whether a new model replaces production?
A quality gate in config: absolute minimums (macro-F1 0.75, PR-AUC 0.80, HIGH+CRITICAL recall 0.90, CRITICAL
recall 0.70, escalation recall 0.95), no regression beyond a tolerance vs production on the same frozen test
set, and an identical feature schema. Passing moves the MLflow `@production` alias; the old version becomes
`@previous` for rollback. Two stories: a corrupted model with 0.98 CRITICAL recall failed because macro-F1 and
PR-AUC caught it; and when a bug fix made v6 miss the CRITICAL minimum by one contract, I didn't override the
gate — I tried a sanctioned improvement and disclosed that it came after seeing the result.

### 21. How would you handle 1 million documents?
Measured locally: extraction averages ~0.2 s per contract across the corpus (an OCR'd scan takes ~1.5 s end
to end), and everything after extraction takes tens of milliseconds. A million documents is therefore roughly
2–3 CPU-days of extraction, so: an async queue with stateless workers scaling
horizontally, OCR workers in their own pool, files in object storage, batch inference for backfills,
partitioned tables and indexes on vendor/date, and LLM explanations on demand rather than for every document.

### 22. How would you scale document processing?
Split the pipeline into stages connected by a queue: validate/store → extract (CPU, OCR-heavy) → analyse
(cheap) → explain (GPU/LLM, optional). Each scales independently; failed stages retry idempotently by document
ID. The code is already structured this way: `process_document` and `analyze_document` are separate pure
functions.

### 23. How would you handle concurrent uploads?
The API is stateless; uploads are deduplicated by SHA-256; a document already "processing" returns 409. With
the queue design, the upload request returns immediately and the client polls status — the dashboard already
polls for the LLM explanation the same way. Database writes are per-document transactions.

### 24. How would you secure uploaded documents?
Already in place: content-based validation (magic bytes, parser open, zip-bomb and page limits), sanitised
filenames, per-user API keys stored as SHA-256 hashes, role permissions, four-eyes validation, audit logs
without document text, no text sent to the LLM, secrets only via environment. For production: SSO/OIDC,
encryption at rest (KMS) for object storage, malware scanning, TLS everywhere, retention/deletion policies,
row-level access by business unit.

### 25. How would you improve OCR accuracy?
Measure first — I already compare OCR extraction to ground truth per field. Then: higher DPI and deskew /
binarisation pre-processing, Tesseract page-segmentation tuned per layout, confidence-based routing to review
(already done: OCR documents are flagged for human verification), fuzzy matching of vendor names against the
vendor master, and for real scans a layout-aware document AI model with these rules as fallback.

### 26. How would you deploy this on AWS?
ECS Fargate (or EKS) for API and workers from the same image; RDS PostgreSQL; S3 for uploads and MLflow
artifacts; MLflow on ECS with RDS as backend store; SQS between pipeline stages; an optional GPU instance for
the LLM (or Bedrock if policy allows); Secrets Manager; CloudWatch for the JSON logs; ALB in front of the API
and the static dashboard on S3 + CloudFront. The registry alias approach doesn't change.

### 27. How would you reduce inference latency?
The ML part is already cheap (SHAP ~18 ms, LightGBM inference is milliseconds once warm; the first cold call
was ~0.5 s). Latency is dominated by extraction/OCR and the LLM. So: keep models loaded (already done at
startup), warm-up calls on deploy, OCR only on pages without text (already done), run the LLM asynchronously
(already done), cache vendor history lookups, and batch documents for bulk analysis.

### 28. What happens if the database fails?
Requests fail fast with a controlled 503 `database_unavailable` (5-second connect timeout — I added this after
seeing a request hang when my local database was down). `/health` returns 503. Request metrics are written
best-effort so metrics failures never break requests. In production: managed PostgreSQL with multi-AZ failover,
connection pooling, retries with backoff in workers, and backups with tested restores.

### 29. How would you implement asynchronous document processing?
Upload returns 202 with a document ID; a worker (Celery/RQ/SQS consumer) runs `process_document` and
`analyze_document`; status lives on the `documents.processing_status` column the API already exposes; the
client polls or subscribes. Idempotency by document ID, dead-letter queue for failures, visibility in
monitoring. The LLM step already works this way via FastAPI background tasks.

### 30. What would you change for a production enterprise deployment?
Real labelled data and a proper annotation process; layout-aware extraction; vendor identity from the vendor
master / tax IDs instead of names; an async pipeline; SSO and encryption at rest; shadow and canary deployment
before promotion; confidence intervals in the gate (with 45 CRITICAL test cases a single contract moves recall
by 2 points); recalibrated score thresholds; Evidently-style reports for stakeholders; and an on-call runbook.

---

## Stories worth telling (they show judgement, not just tools)

1. **Diagnosing a weak model instead of tuning blindly.** v1 failed the gate. I wrote a label-observability
   check: 94% of labels could be reconstructed from the extracted features, so the information was there and
   the models were under-fitting. A learning curve was still rising → more data plus policy-indicator
   features → big improvement.
2. **A selection rule that ignored the business constraint.** v2 had good F1 but 0.62 CRITICAL recall. I made
   selection constraint-aware and added threshold tuning on out-of-fold predictions — v3 passed.
3. **An entity-resolution bug found through monitoring.** The dashboard showed 78% of contracts flagged as
   anomalous. Root cause: vendor names were normalised too aggressively, so "X LLP" and "X Pvt Ltd" shared
   history. I fixed it, versioned the features (fv3) and made the server refuse a model trained on a different
   feature version.
4. **Not overriding the gate.** v6 missed one threshold by a single contract. I kept the gate, tried a
   sanctioned change (LightGBM) and disclosed it in the report.
5. **LLM hallucination traced to my own input format**, fixed with self-describing evidence and a contradiction
   check plus a regression test.

## Weak spots to own before they're found

- Synthetic data and rule-derived labels: the model learns my policy, not expert judgement.
- Small CRITICAL test class → noisy recall estimates.
- v7 came after seeing v6's test result (disclosed).
- Docker stack not run end-to-end locally (no Docker on the dev machine); CI builds the images.
- Rule-based extraction won't survive real-world layout variety without a layout-aware model.
