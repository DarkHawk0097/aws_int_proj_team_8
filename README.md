# P3 — Demand Forecasting & Inventory Optimization System

**AWS Academy ML Engineering Internship — Capstone Project P3**
**Author:** Arshit Samkria (arshit.24bce10444@vitbhopal.ac.in), Simran Kumari(simran.kumari2024a@vitstudent.ac.in)
**Environment:** AWS Academy Learner Lab (LabRole, us-east-1)
**S3 Bucket:** `arshit-p3-demand-forecast-aws`

---

## 1. Project Overview

This project builds an end-to-end system that forecasts product-level demand across multiple stores and uses that forecast — including its uncertainty range — to inform inventory and safety-stock decisions. The system is built entirely on Amazon SageMaker's built-in **DeepAR** algorithm, deployed via **Batch Transform**, monitored for data drift with **SageMaker Model Monitor**, and orchestrated on a recurring schedule using **AWS Step Functions** and **Amazon EventBridge**.

The core design goal was to demonstrate a full ML lifecycle — not just a trained model in a notebook, but a system that can retrain, re-forecast, and re-evaluate itself on a schedule without manual intervention.

---

## 2. Architecture

```
                    ┌─────────────────────┐
   EventBridge      │  Weekly Cron Trigger │
   (weekly cron) ───►  cron(0 6 ? * MON *) │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Step Functions     │
                    │   State Machine      │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
 ┌───────────────┐    ┌────────────────┐    ┌───────────────────┐
 │ TrainDeepAR    │───►│ CreateModel     │───►│ BatchTransform     │
 │ (SageMaker)    │    │ (SageMaker)     │    │ (SageMaker)        │
 └───────────────┘    └────────────────┘    └─────────┬──────────┘
                                                        │
                                                        ▼
                                              ┌────────────────────┐
                                              │ ComputeMAPE (Lambda)│
                                              │ → CloudWatch Metric │
                                              └────────────────────┘

 Parallel: SageMaker Model Monitor (daily schedule)
 reads captured Batch Transform input → checks for data drift
 against the training baseline → CloudWatch + violation reports
```

**Data flow through S3:**

```
s3://arshit-p3-demand-forecast-aws/
├── raw/               → synthetic sales_data.csv
├── processed/         → train.json, test.json, transform.json
├── models/            → DeepAR model artifacts
├── forecasts/         → batch transform output (forecast quantiles)
├── captured-data/      → BatchDataCaptureConfig output (input to Model Monitor)
└── monitoring/
    ├── baseline/       → suggest_baseline() output
    └── reports/        → scheduled monitoring job reports
```

---

## 3. AWS Services Used

| Service | Purpose |
|---|---|
| Amazon S3 | Data lake for raw, processed, model, forecast, and monitoring artifacts |
| SageMaker (DeepAR) | Probabilistic time-series forecasting model |
| SageMaker Batch Transform | Cost-efficient, schedule-friendly batch inference (no always-on endpoint) |
| SageMaker Model Monitor | Data-quality drift detection against a training baseline |
| AWS Lambda | Computes MAPE from forecast vs. actuals, pushes to CloudWatch |
| AWS Step Functions | Orchestrates train → deploy → forecast → evaluate as one workflow |
| Amazon EventBridge | Triggers the Step Functions pipeline on a weekly schedule |
| Amazon CloudWatch | Custom MAPE metric, alarm on accuracy degradation, Model Monitor metrics |
| AWS IAM (LabRole) | Single scoped execution role for all services, per Learner Lab constraints |

---

## 4. Environment Notes (AWS Academy Learner Lab)

This project runs inside an AWS Academy Learner Lab sandbox, not a personal AWS account. This has a few consequences documented here for anyone reviewing or re-running the notebook:

- **IAM is locked down.** No custom roles were created. Every service (SageMaker, Lambda, Step Functions, EventBridge) uses the pre-provisioned **LabRole**, which has broad-but-scoped permissions typical of the Learner Lab environment.
- **Region:** us-east-1 (required — Learner Lab restricts to us-east-1 / us-west-2).
- **SageMaker instance types** are restricted to the Learner Lab-supported set (`ml.t3.medium`, `ml.c5.xlarge`, `ml.m5.xlarge`, etc.). No GPU instances were needed or used — DeepAR trains efficiently on CPU at this data scale.
- **Budget constraint:** the lab operates under a fixed budget cap. This directly influenced two design decisions documented in the Production Roadmap (Section 8): AWS Glue and SageMaker Hyperparameter Tuning were designed and documented but not executed against real infrastructure, to preserve budget for the core deliverables.
- **Session timers:** the Learner Lab session is time-boxed. S3 data and created resources persist across sessions; the SageMaker notebook instance stops between sessions and must be manually restarted.

---

## 5. Data

All data used in this project is **synthetically generated** by the accompanying notebook — it is not sourced from any public dataset, Kaggle notebook, or third-party tutorial. This is stated explicitly for academic-integrity purposes.

The synthetic dataset simulates:
- 20 products (`PROD-01` – `PROD-20`) across 3 stores (`STORE-01` – `STORE-03`)
- 730 days (2 years) of daily sales history
- Seasonality (annual sine-wave pattern), weekend demand boosts, and randomized promotional periods (~5% of days, +40% demand boost)
- Per-product baseline demand levels, which are also used to construct the cold-start category feature (Section 6)

---

## 6. Cold-Start Strategy

**Requirement addressed:** *"Include a cold-start strategy for new products with little or no sales history."*

DeepAR supports a categorical feature (`cat`) that lets the model share a learned embedding across series in the same category. This project uses each product's baseline demand level to assign it into one of **4 demand tiers** (quartiles: low, medium-low, medium-high, high), computed once from `product_base_demand` and attached to every training, test, and transform record via `"cat": [category_index]`.

In practice, this means a brand-new product with zero sales history can be assigned to the demand tier it's expected to belong to (based on category, price point, or business judgment) and immediately inherit that tier's learned demand pattern, rather than the model having no prior information about it at all. `cardinality="4"` was set in the DeepAR hyperparameters to match the four tiers.

---

## 7. Model Monitoring

**Requirement addressed:** *"SageMaker Model Monitor: to track forecast accuracy... and flag when accuracy degrades."*

Two complementary monitoring mechanisms are used, since they answer different questions:

1. **Accuracy tracking (CloudWatch custom metric):** After each batch transform run, forecasted medians are compared against held-out actuals to compute MAPE, which is pushed to CloudWatch under the `P3-DemandForecasting` namespace. A CloudWatch Alarm (`P3-MAPE-High-Alarm`) fires if MAPE exceeds 15%. This answers: *"is the model's accuracy degrading?"*

2. **Data drift detection (SageMaker Model Monitor):** A `DefaultModelMonitor` was configured with a baseline built from `train.json` (via `suggest_baseline`), and a daily monitoring schedule (`p3-forecast-datadrift-monitor`) checks incoming batch-transform input (captured via `BatchDataCaptureConfig`) against that baseline for statistical drift. This answers: *"has the shape of incoming data changed in a way that could silently break the model?"*

Both metrics report to CloudWatch, giving both an accuracy signal and a data-integrity signal — covering the monitoring requirement from two angles rather than relying on a single manual check.

---

## 8. Automation: Step Functions + EventBridge

**Requirement addressed:** *"Automate the end-to-end pipeline to run on a recurring schedule without manual intervention."*

A Step Functions state machine (`P3-Demand-Forecasting-Pipeline`) chains four states:

1. `TrainDeepAR` (`sagemaker:createTrainingJob.sync`) — retrains DeepAR on the latest processed data
2. `CreateModel` (`sagemaker:createModel`) — registers the new model artifact
3. `BatchTransform` (`sagemaker:createTransformJob.sync`) — generates fresh forecasts for the full catalog
4. `ComputeMAPE` (`lambda:invoke`) — computes accuracy against held-out actuals and pushes to CloudWatch

An EventBridge rule (`P3-Weekly-Retrain-Trigger`, `cron(0 6 ? * MON *)`) triggers this state machine every Monday at 06:00 UTC, targeting the state machine's ARN directly — no manual notebook execution required for the pipeline to keep running.

This was deployed and executed live (not just designed) — a full end-to-end execution was triggered manually and observed completing successfully through all four states in the Step Functions console, confirming the pipeline works before relying on the schedule to run it automatically going forward.

> **Budget note:** the EventBridge rule and Model Monitor schedule were disabled/deleted after confirming a successful run, to avoid indefinite background cost against the Learner Lab budget cap. In a production AWS account, both would be left running continuously.

---

## 9. Using Forecast Uncertainty for Safety-Stock Decisions

DeepAR doesn't just give us a single predicted number for each product's demand — it gives us a full probability range (captured here as the 0.1, 0.5, and 0.9 quantiles). The gap between the 0.1 and 0.9 quantiles tells us how confident the model is about that specific product-store series. A narrow gap means the model is fairly certain of the demand, so that product can safely run with lean inventory close to the median forecast. A wide gap means genuine uncertainty — the actual demand could swing well above or below the median — and holding stock only to the median forecast risks frequent stockouts whenever real demand lands near the upper end of that range.

In practice, this means safety stock shouldn't be a flat buffer applied equally across the catalog. Instead, it should scale with each product's own prediction interval: for a given service-level target, safety stock can be set using the gap between the median (0.5) and a chosen upper quantile (e.g., 0.9) rather than a fixed percentage of average demand. Products with wide intervals — typically newer items, low-history products relying on the cold-start category embedding, or those with irregular promo-driven spikes — should carry proportionally more buffer stock, while consistently-selling products with narrow intervals can operate closer to just-in-time levels. This directly ties the model's uncertainty output to a concrete inventory action, rather than treating the forecast as a single deterministic number the business has to blindly trust.

---

## 10. Production Roadmap (Designed, Not Executed in Sandbox)

The following components were architected and documented, but intentionally not run against live infrastructure in this Learner Lab, to stay within the sandbox's budget cap. Code is included in the notebook appendix for reference.

- **AWS Glue ETL migration:** the current pandas-based transformation logic is designed to migrate to a PySpark Glue job reading from a Glue Data Catalog table, for datasets too large for a single notebook instance to process in memory.
- **SageMaker Hyperparameter Tuning:** a `HyperparameterTuner` sweep over `learning_rate`, `num_cells`, and `context_length` is defined, minimizing `test:mean_wQuantileLoss`, to replace the manually-chosen hyperparameters used in this submission.

---

## 11. Notebook Structure (Execution Order)

1. Synthetic data generation (`sales_data.csv` → S3 `raw/`)
2. Demand-tier categorization for cold-start (`product_category`)
3. JSON-lines preparation for DeepAR (`train.json`, `test.json`, `transform.json`, all including `cat` and `dynamic_feat`)
4. DeepAR training (`estimator.fit(...)`)
5. Batch Transform inference with data capture enabled
6. MAPE computation against held-out actuals
7. CloudWatch metric push + alarm configuration
8. Forecast vs. actual visualization (matplotlib, quantile bands)
9. SageMaker Model Monitor: baseline + monitoring schedule
10. Lambda deployment (MAPE calculator)
11. Step Functions state machine creation
12. EventBridge weekly trigger + manual proof-of-execution run

---

## 12. Honest Limitations

- Ground-truth actuals for freshly forecasted future periods aren't available in a synthetic-data sandbox project, so accuracy evaluation uses a held-out historical window (last 14 days of the 2-year synthetic history) rather than true future actuals.
- Model Monitor here tracks *data drift*, not *ground-truth prediction quality drift* — the latter would require a live feedback loop with real actuals arriving after each forecast period, which is a production concern beyond this project's scope.
- Glue and HPO are designed, not executed, per the budget constraints documented in Section 10.
