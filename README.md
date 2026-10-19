# Flex travel allocation

Allocate SNCB **Flex subscription sales** to individual **travels** (pax), each with a
travel date and travel hour, using observed activation behaviour and survey
probability distributions — while respecting each product's validity period and
activation limits, and distinguishing **there-travels** from **back-travels**.

The model is written in PySpark and follows the Advanced Analytics team's technical
conventions (Spark-native, functional, notebooks orchestrate while functions implement
logic, centralised data access and configuration).

## Business context

Flex subscriptions target commuters who do not travel every working day. A subscription
is valid on one route in both directions for a number of travel days within a validity
window:

| Product  | Article code | Activation limit | Validity   |
| -------- | ------------ | ---------------- | ---------- |
| Flex 6   | S0256 (256)  | 6 travel days    | 1 month    |
| Flex 10  | S0257 (257)  | 10 travel days   | 1 month    |
| Flex 80  | S0258 (258)  | 80 travel days   | 12 months  |
| Flex 120 | S0259 (259)  | 120 travel days  | 12 months  |

Before travelling, a commuter *activates* the subscription for a date. Each activation is
assumed to produce **two travels**: a there-travel and a back-travel (each one pax).

## Approach

1. **Reconcile tickets** — the sales `ticket_number` and the activation
   `ticket_number_activations` use different formats. The activation ticket is the
   four-digit sale year followed by the sales ticket number left-padded with zeros to 19
   digits (23 characters total). Verified on the provided data: 37 675 / 37 675 sales
   match 1:1. See `ticket_matching.py`.
2. **Join and constrain** — join each sale to its activations, keep only activations that
   fall **inside the validity window** and **within the product activation limit**
   (`activation_sequence <= maximum_activations_allowed`). See `allocation.py`.
3. **Estimate travel times** — for each retained activation:
   - **there-travel**: sample `minutes_to_add` from the survey distribution and add it to
     the activation timestamp;
   - **back-travel**: sample `hours_between_there_and_back` and add it to the there-travel.
   Sampling is weighted by the survey probabilities, deterministic for a fixed `SEED`, and
   implemented with a native Spark cumulative-probability range join — **no UDFs**. See
   `travel_timing.py`.
4. **Emit travels** — every activation becomes one there-travel row and one back-travel
   row (one pax each), in a single long table.

Timestamps are stored in UTC and reported in local railway time (`Europe/Brussels`) for
the travel date and hour.

## Output schema (`flex_travels`)

| Column                   | Type      | Description                                  |
| ------------------------ | --------- | -------------------------------------------- |
| `ticket_number`          | bigint    | Sales ticket number                          |
| `article_code_int`       | smallint  | Flex product code (256–259)                  |
| `article_description_nl` | string    | Product description                          |
| `travel_direction`       | tinyint   | `1` = there, `2` = back (`TravelDirection`)  |
| `travel_date`            | date      | Local travel date                            |
| `travel_hour`            | tinyint   | Local travel hour (0–23)                     |
| `travel_timestamp`       | timestamp | Estimated travel instant                     |
| `pax`                    | tinyint   | Passengers for the travel (1)                |

Key invariant (smoke test): `sum(pax) == 2 × number of valid activations`.

## Assumptions, exceptions and special handling

- **Ticket format**: activation ticket = `year(sales_date)` + `ticket_number` zero-padded
  to 19 digits. Derived from `sales_date` rather than hardcoding the year.
- **Two travels per activation**: each activation yields exactly one there-travel and one
  back-travel (one pax each), per the assignment.
- **Validity**: activations with `activation_date` outside `[travel_start_date,
  travel_end_date]` are dropped.
- **Activation limit**: activations beyond `maximum_activations_allowed` (by
  `activation_sequence`) are dropped, so allocation never exceeds a product's limit.
- **Sampling vs. expansion**: travel times are **sampled** (one concrete time per travel)
  rather than fractionally expanded, giving a realistic, storage-efficient travels table.
  Determinism is guaranteed by `SEED` in `conf/config.py`.
- **Timezone**: activation timestamps are UTC; travel date/hour are reported in
  `Europe/Brussels`.

## Project structure

```
flex-travel-allocation/
├── conf/
│   ├── config.py            # environments, table names, SEED, timezone, IntEnums
│   ├── params.yml           # tunable analysis parameters (no magic numbers in scripts)
│   └── params.py            # loads params.yml for the simulation/demo scripts
├── src/flex_travel_allocation/
│   ├── data_handler.py      # governed read_table / write_table (DataHandler.from_config)
│   ├── ingestion.py         # clean/cast raw input tables
│   ├── ticket_matching.py   # reconcile sales <-> activation ticket formats
│   ├── travel_timing.py     # weighted, seeded sampling of travel times (no UDF)
│   ├── allocation.py        # join, validity/limit filtering, there/back travels
│   ├── analysis.py          # distribution aggregations for the presentation
│   ├── expected_distribution.py  # expected (probability-weighted) travel-hour distribution
│   ├── monte_carlo.py       # Monte-Carlo confidence bands for the sampled distribution
│   ├── revenue_risk.py      # sales vs activation revenue-risk analysis
│   ├── mdp_what_if.py       # MDP what-if (value iteration) for the sale->activation transition
│   ├── capacity_analysis.py # train-filling / capacity (sales vs allocation) by hour
│   ├── train_scheduling.py  # trains scheduled per hour (sales vs activation) + savings
│   ├── tradeoff_analysis.py # capacity-vs-revenue-risk trade-off + per-product recommendation
│   ├── scenario_tables.py   # pandas tables for the revenue-risk, MDP and capacity analyses
│   └── models/              # production model estimation + Spark scoring
│       ├── data_prep.py     # Spark unit table (per subscription): activation lag, exposure
│       ├── estimators.py    # Stage 1 cure model + Stage 2 negative-binomial GLM
│       ├── scoring.py       # apply fitted parameters as Spark columns (no UDF)
│       └── parameters.py    # MODEL_PARAMETERS schema + fit_version
├── notebooks/
│   ├── flex_travel_allocation_pipeline.py           # production pipeline (allocation + scoring)
│   ├── flex_travel_allocation_modeling_pipeline.py  # model estimation job (writes MODEL_PARAMETERS)
│   ├── flex_travel_allocation_input_tests.py        # input data-quality checks
│   ├── flex_travel_allocation_input_audit.py        # input data audit (sample, types, statistics)
│   └── flex_travel_allocation_analysis.py           # visualisations (manual, non-production)
├── streamlit_app/           # interactive demo (app.py, artifacts/, assets/)
├── tests/                   # unit, contract (schema) and smoke tests
├── precompute_demo.py       # build the demo artifacts from the dummy data
├── revenue_simulation.py    # revenue-risk report (sales vs activation)
├── mdp_simulation.py        # per-product MDP what-if report
├── capacity_simulation.py   # train-filling capacity report
├── train_scheduling_simulation.py  # trains scheduled (sales vs activation) report
├── score_units_simulation.py  # materialise flex_model_parameters + flex_unit_scores (CSV, for the demo)
├── modeling/                # offline sale->activation->travel value model (analysis + artifacts)
│   ├── run_modeling.py      # fit Stage 1/2, chain, bootstrap, diagnostics -> modeling/output/
│   ├── stage1_cure.py       # mixture cure model + saturating curve (activation)
│   ├── stage2_travel.py     # logistic + negative-binomial (travel intensity)
│   ├── chaining.py          # expected value per sale, bootstrap, funnel leak, recommendation
│   ├── diagnostics.py       # activation-lag bimodality check
│   └── output/              # CSVs, key_metrics.json, report, figures (mirrored to the app)
├── pyproject.toml           # package + tooling config (ruff, pytest)
├── .env.example             # template for local JAVA_HOME / HADOOP_HOME (.env is gitignored)
├── requirements.txt         # runtime dependency pins
└── requirements-dev.txt     # development/CI dependencies
```

## Local development

PySpark 3.5.x requires **Python >=3.10,<3.13** and a JDK (Java 17 recommended).

Using [uv](https://docs.astral.sh/uv/) (the team's environment manager):

```powershell
uv venv --python 3.11 .venv
uv pip install -r requirements-dev.txt
```

Or with plain `pip`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
```

### Configuration

- Machine paths (`JAVA_HOME`, `HADOOP_HOME`) live in a local `.env` — copy `.env.example`
  to `.env` and adjust. `.env` is gitignored; only the template is committed.
- Tunable analysis parameters (non-payment rate, MDP uplift/cost, Monte-Carlo replicates,
  seats per train, target load factor, Spark settings) live in `conf/params.yml`. The
  simulation/demo scripts read them via `conf/params.py`, so no magic numbers are hardcoded.
  Structural constants (table names, article codes, `SEED`, timezone) stay in `conf/config.py`.

### Run the tests

```powershell
.\.venv\Scripts\python -m pytest
```

### Lint / pre-commit

```powershell
.\.venv\Scripts\python -m ruff check .
pre-commit run --all-files
```

## Demo app (Streamlit)

An interactive demo in `streamlit_app/` shows a sample subscription (input → output)
and the global travel distributions. It reads small precomputed artifacts, so it is
fast and needs no Spark.

```powershell
# 1. (re)generate the artifacts from the dummy data via the model
.\.venv\Scripts\python precompute_demo.py

# 2. run the app
.\.venv\Scripts\python -m streamlit run streamlit_app/app.py
```

Then open <http://localhost:8501>. To use the official SNCB logo, drop
`sncb_logo.png` into `streamlit_app/assets/` (the app auto-detects it and otherwise
falls back to a branded banner). The demo also includes the revenue-risk and MDP
what-if analyses described below, plus a **travel-allocation value-model** section
(sale → activation → travel) driven by the modeling outputs.

## Analyses: capacity, revenue risk, trade-off and recommendation

Before moving from a **sales** model (revenue at purchase) to an **activation** model
(revenue per travel), the model runs four analyses per product — in the order
**capacity → revenue risk / what-if → trade-off → recommendation**. Figures below come
from the provided dummy data; the non-payment rate, promotion uplift and incentive cost
are assumptions to be calibrated (e.g. with an A/B test).

### 1. Train filling: capacity (sales vs allocation)

- The allocation gives the **actual** pax per hour; capacity planned from **sales** assumes
  every paid day is travelled, so potential demand per hour = allocated demand / product
  utilisation.
- Function: `train_filling_by_hour(travels, product_utilisation)` in `capacity_analysis.py`.
- Headline: **2.19M actual vs 3.15M sales-potential** travels (**69.3% overall fill**); the
  07h peak is **333k actual vs 481k** sales-planned, so sizing the fleet to sales
  over-provisions the peak by **+44%** (actual trains run at ~69% peak / ~19% average load
  against a sales-sized fleet).

```powershell
# prints a per-hour capacity table; writes capacity_simulation.txt and
# streamlit_app/artifacts/output_capacity_by_hour.csv
.\.venv\Scripts\python capacity_simulation.py
```

#### 1a. Train scheduling: optimising the number of trains

- Turns the per-hour capacity into an **operational fleet plan**: trains per hour =
  `ceil(pax / (seats_per_train × target_load_factor))`, computed once for the
  **sales-planned** demand and once for the **actual activation** demand.
- Function: `trains_by_hour(capacity, seats_per_train, target_load_factor)` and
  `train_optimisation_summary(trains)` in `train_scheduling.py`.
- Illustrative fleet (500 seats/train, planned to 80% max load): a sales-based plan
  schedules **7,894 trains/day** versus **5,478** when planning to activation — **2,416
  fewer trains/day (−30.6%)**. At the 07h peak that is **1,203 vs 834 trains (−369)**.
- Fewer scheduled trains at equal service means lower crew, energy and rolling-stock cost;
  the saving tracks the under-use, so it is largest on the long passes (Flex 80/120).

```powershell
# prints a per-hour trains table + optimisation summary; writes
# train_scheduling_simulation.txt and streamlit_app/artifacts/output_trains_by_hour.csv
.\.venv\Scripts\python train_scheduling_simulation.py
```

### 2. Revenue risk: sales vs activation

- **Sales (today):** the full subscription price is collected upfront, guaranteed
  regardless of use.
- **Activation (proposed):** revenue is recognised per activation at
  `price / max_activations`.
- Because Flex holders under-use their passes, the activation model earns less; charging
  at activation also exposes revenue to **non-payment** (evasion / uncaptured taps).
- Function: `revenue_risk_by_product(sales, articles, activations, non_payment_rate)` in
  `revenue_risk.py`.
- Headline: sales **€20.9M** vs activation **€14.7M** → **~€6.2M (29.8%) at risk**,
  concentrated in Flex 80/120 (Flex 120 uses ~50% of its days); non-payment adds ~€0.15M
  per 1% evasion.

```powershell
# prints a per-product table + non-payment sensitivity; writes revenue_simulation.txt
# and streamlit_app/artifacts/output_revenue_risk.csv
.\.venv\Scripts\python revenue_simulation.py
```

### 3. What-if: sale -> activation transition (MDP)

Each subscription is modelled as a **Markov Decision Process**:

- **State** `j` = paid days already activated (`0..N`).
- **Hazard** `h[j]` = `P(activate day j+1 | used j)`, estimated from the data (a
  per-product survival curve).
- **Reward** = the per-day price on each activation; lapsing is absorbing.
- **Action** = *promote* an extra activation (hazard `+uplift` at an `incentive_cost`).
- **Value iteration** returns the optimal promotion policy and the recoverable revenue
  (the acyclic state space makes one backward sweep exact).
- Functions: `estimate_hazards` + `evaluate_transition` in `mdp_what_if.py` (numpy solver);
  `mdp_what_if_table` in `scenario_tables.py` estimates the per-product hazards in Spark.
- Headline: targeted promotion recovers **~€2.1M/yr net** (about a third of the gap);
  the optimal policy promotes Flex 80/120 while cumulative use is below ~88-89%, and
  leaves Flex 6/10 on sales pricing (already ~98% used).

```powershell
# prints a per-product what-if table; writes mdp_simulation.txt and
# streamlit_app/artifacts/output_mdp_what_if.csv
.\.venv\Scripts\python mdp_simulation.py
```

### 4. Trade-off: capacity vs revenue risk

- Both effects grow with **under-use**, so they rank the products the same way: the
  products that expose the most **revenue** if switched to activation pricing free the
  most **train capacity** if planned to actual demand, and offer the most MDP-recoverable
  revenue.
- Function: `product_tradeoff(mdp_table)` in `tradeoff_analysis.py` (writes
  `streamlit_app/artifacts/output_tradeoff.csv` from the precompute).
- Priority quadrant: **Flex 120** (~50% risk / ~50% capacity freed, *High*), **Flex 80**
  (~23%, *Medium*), **Flex 6/10** (~1-2%, *Low*).

### Recommendation per product

- **Flex 6 & 10** (~98% used, *Low*): keep the sales model; capacity already matched; no action.
- **Flex 80** (77% used, *Medium*): keep sales pricing; plan capacity to actual (-23% fleet);
  pilot targeted promotion (+€0.77M).
- **Flex 120** (50% used, *High*): keep sales pricing (activation would halve revenue);
  plan capacity to actual (-50% fleet); prioritise promotion (+€1.32M).
- Rule of thumb: **do not** switch pricing to activation where under-use is high (it only
  destroys revenue); **always** plan capacity to actual demand (saving scales with under-use);
  focus MDP promotion on Flex 80/120 (~€2.1M/yr net).

All four analyses are shown interactively in the Streamlit demo (in the order capacity →
revenue risk / what-if → trade-off → recommendation) and in the modeling section.

## Modeling: sale → activation → travel value model

A separate, unit-level model reframes revenue from *cash collected at sale* to **expected
value per sale**, by chaining two stages fitted on the individual subscriptions.

- **Stage 1 — sale → activation (mixture cure model).** `P(activate ever) = logit⁻¹(Xβ)`
  with Weibull timing among activators; a saturating curve `p(t)=p∞(1−e^(−λt))` recovers the
  same headline numbers, censoring-corrected. Yields the **activation ceiling p∞**
  (breakage = 1 − p∞), the **speed** (median lag) and the lag **shape** (Weibull k).
- **Stage 2 — travel after activation.** Logistic regression for *travelled at least once*
  and a **negative-binomial GLM** with a `log(exposure)` offset for the trip **count** →
  trips per active-day (Poisson overdispersion ⇒ negative binomial is required).
- **Chaining (per unit sold):**
  `E[value] = p∞·(P(travel)·m_used + (1−P(travel))·m_idle) + (1−p∞)·m_breakage`, with margins
  as fractions of price and a **1000-draw joint bootstrap** for the confidence interval.

**Headline (dummy data):** p∞ ≈ **100%** (breakage ≈ 0), median activation lag ~2–3 days;
**travel-allocation E[value] €399/sale vs only-sale €344 → +€55 (95% CI 55–56)**. The gap
sits in usage (pricing), not activation; the €20.9M collected runs off within ~3 weeks; the
activation lag is **bimodal** (same-day spike + ~3-day hump — possible gifting/stockpiling).

**Recommendation per product (from the chained model):** migrate to travel-allocation
recognition everywhere (all value-accretive). **Flex 120 & 80 (High: +€144 / +€114 per sale)**
are the priorities (light per-day use, high idle margin); **Flex 10 & 6 (Low: +€12 / +€9)**
still add value, with Flex 10 the heaviest per-day user.

```powershell
# offline analysis: fit + chain + bootstrap + diagnostics -> modeling/output/ (+ app artifacts)
.\.venv\Scripts\python -m modeling.run_modeling
```

### Productionising the models

The estimators live in the installable package under `src/flex_travel_allocation/models/`,
following a **fit-offline → persist parameters → score in-pipeline** split so the daily job
stays Spark-only:

- **Estimation job** (weekly) — `notebooks/flex_travel_allocation_modeling_pipeline.py`
  builds the Spark unit table, collects the small frame to the driver, fits Stage 1 + Stage 2
  and writes the tiny **`flex_model_parameters`** table (`p∞`, `λ`, Weibull `k/η`, NB rate/day,
  dispersion, `fit_version`, `as_of`).
- **Scoring** (daily) — the allocation pipeline reads `flex_model_parameters` and applies it as
  pure Spark columns (`models.scoring.score_units`): `expected_trips`, `p_activated_by_now`,
  `expected_breakage_value` → written to **`flex_unit_scores`** (skipped gracefully until the
  model is first fitted). No scipy/statsmodels on the hot path.

```powershell
# materialise flex_model_parameters + flex_unit_scores locally as CSVs (shown in the demo app)
.\.venv\Scripts\python score_units_simulation.py
```

## Running on Databricks

Notebooks are stored as `.py` (Databricks source format) for version control. The pipeline
notebook reads inputs, allocates travels and writes the `flex_travels` table via the
`DataHandler`. Set the `environment` widget (`NPD` / `PRD`); it defaults to `NPD`.

The estimation notebook (`flex_travel_allocation_modeling_pipeline.py`) fits the Stage 1/2
models and writes `flex_model_parameters`; the pipeline then scores units into
`flex_unit_scores`. `scipy` / `statsmodels` must be available on the estimation cluster
(they are pinned in `requirements.txt`); the daily allocation job needs neither.

- `flex_travel_allocation_input_tests.py` — run before the pipeline to validate inputs;
  it raises if `errors_stopping_pipeline` is non-empty.
- `flex_travel_allocation_pipeline.py` — production, non-interactive, deterministic.
- `flex_travel_allocation_analysis.py` — manual exploration and presentation plots.

## Data inputs

Provided as dummy parquet files under `../data` (resolved by `DataHandler` in `NPD`):

- `article_code_flex` — products and activation limits.
- `sales_flex` — three months of purchased subscriptions with validity periods.
- `activations_article` — product per activation ticket.
- `activations_flex` — activation timestamps and validity.
- `minutes_to_add_probabilities` — there-travel minute offsets.
- `hours_between_there_and_back_probabilities` — back-travel hour offsets.
