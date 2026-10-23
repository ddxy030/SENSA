# TimeMMD-MMTSFlib-Corrected v1

This is the **frozen dataset version** used for subsequent unified reruns of the proposed model and baselines. Freeze date: **2026-07-22**.

The numerical sequences are based on the MM-TSFlib snapshot. The corresponding `fact` and `preds` text fields are taken from the SpecTF `data/tats` snapshot and aligned row by row with the numerical records. This release does **not** use the numerical files from the current original Time-MMD repository.

All CSV files are UTF-8 encoded with a BOM and use the following schema:

```text
date,start_date,end_date,OT,fact,preds
```

- `OT`: the only forecasting target; the main protocol uses univariate input and a single output target.
- `fact`: factual text provided by the upstream dataset.
- `preds`: prediction/outlook text provided by the upstream dataset. It is retained in the files but is not enabled in the main experiments.
- The CSV files retain the original numerical values; **no global normalization has been applied in advance**. Normalization must be fitted at runtime using the training partition only.

## 1. Climate target definition: do not overlook this

The Climate domain in this version uses the monthly `US_precipitation_month` numerical series from MM-TSFlib / SpecTF. The actual meaning of `OT` is:

> **Contiguous U.S. Monthly Precipitation (inches)**

It must not be described in the paper as “Drought Level.” Precipitation is related to drought, but drought levels also depend on soil moisture, temperature, evapotranspiration, runoff, duration, and other factors. These concepts are not equivalent. The target may be described as “monthly precipitation” or, with an explicit qualification, “a precipitation proxy related to drought,” but it must not be renamed “drought level.”

The `Drought Level / Monthly / 496` label in the Time-MMD paper's table does not match the actual target in the public MM-TSFlib file. This release follows the meaning of the actual data field.

## 2. Data sources and versions

- **MM-TSFlib snapshot:** commit `e789ce78c9bafd8e3ba0d8850f9ad2becbe83548`. The numerical fields `date`, `start_date`, `end_date`, and `OT` come from this version. Repository: https://github.com/AdityaLab/MM-TSFlib
- **SpecTF snapshot:** commit `85185c7b883fed7de40098d76ce6782ba1eba016`. Across all nine domains, `date` and `OT` in `data/tats` match the MM-TSFlib records row by row. This release takes the unified `fact` and `preds` fields from that snapshot. Repository: https://github.com/hiepnh137/SpecTF
- **Time-MMD paper:** used only for task definitions, frequencies, and the official lookback/label/horizon protocol. The numerical files from the current Time-MMD repository are not used directly.
- **KairosAgent paper:** identifies an error in the original Economy chronological ordering and reports correcting it before evaluation. This release explicitly parses dates and sorts them in ascending order rather than simply reversing the rows.

## 3. Changes relative to MM-TSFlib

1. **Unified univariate protocol.** Extra covariates, categorical metadata, `his_avg_*`, `his_std_*`, `prior_history_*`, `Final_Search_*`, and `Final_Output` are removed to prevent models from using inconsistent input information.
2. **Unified text fields.** Row-level `fact` and `preds` fields are used. The pre-fixed `Final_Search_2/4/6` fields from MM-TSFlib are not used, avoiding hard-coded retrieval windows in the data files.
3. **Economy chronological ordering.** The original file was ordered by year from newest to oldest, while months within each year were ordered from oldest to newest. A simple `reverse()` would not correctly fix this. This release parses `date` and sorts strictly in ascending order, producing a continuous monthly series from 1989-01 through 2024-03.
4. **Health weekly-date correction.** The original file contained five groups of duplicate weekly dates and five 14-day gaps, caused by converting cross-year week numbers to dates incorrectly. This release retains all 1,389 observations and their paired text, reconstructs a continuous seven-day weekly grid in original row order from 1997-09-29 through 2024-05-06, and recalculates each week's `start_date` and `end_date`.
5. **SocialGood missing-target cleanup.** Eight rows with empty `OT` placeholders from 2024-05 through 2024-12 are removed. The final valid sample count is 892, ending in 2024-04. No interpolation or fabricated target values are used.
6. **Climate retained at monthly frequency.** The public-paper-compatible monthly version with 496 records is used, but the target is interpreted according to the actual field meaning: monthly precipitation.
7. **Environment extraction and standardization.** The numerical series from the MM-TSFlib RAR archive is paired with row-level text from SpecTF. The actual number of valid rows is 15,248, covering 1982-01-01 through 2023-09-30. This differs from the 11,102 samples reported in the paper or some tables; the data are not arbitrarily truncated to match the table.
8. **Date-interval standardization.** Monthly intervals use the first and last day of each month; weekly intervals use Monday through Sunday; for daily data, `start_date = end_date = date`.
9. **Text-format standardization.** Newlines and excess whitespace are consolidated, and exact null-value markers are converted to empty strings. Text semantics have not been manually rewritten.

## 4. Differences from the current original Time-MMD repository

- This release uses the MM-TSFlib paper-compatible numerical snapshot, not the numerical version currently present in the original repository.
- Climate is monthly precipitation in this release. Climate in the current Time-MMD repository is a different weekly drought-related series; these are not the same forecasting task.
- This release retains only `OT + fact + preds` for the main task. It does not include all original covariates or the original, separately split report/search files.
- The text fields have been aligned with the numerical rows for downstream model input. This does not mean the text has been manually reviewed and semantically corrected item by item from the original report/search files.

## 5. Frozen experimental protocol

### 5.1 Task

```text
features = S
target = OT
enc_in = dec_in = c_out = 1
```

For all nine domains, the task is: historical target-variable values plus available historical text → future target-variable values. Results obtained by changing some domains to multivariate forecasting must not be mixed into the main results table.

### 5.2 Official Time-MMD Appendix K.4 window settings

This protocol uses the frequency-dependent input lengths in Appendix K.4 of the Time-MMD paper, rather than the uniform `seq_len=24` used in later SpecTF experiments.

| Frequency | `seq_len` | `label_len` | `pred_len` |
|---|---:|---:|---|
| Monthly | 8 | 4 | 6, 8, 10, 12 |
| Weekly | 36 | 18 | 12, 24, 36, 48 |
| Daily | 96 | 48 | 48, 96, 192, 336 |

`label_len` is used only by models that require decoder warm-up. Models without a decoder must still use the same `seq_len` and `pred_len`.

### 5.3 Chronological split

Apply the split to the full sequence after cleaning and sorting:

```python
train_n = int(N * 0.7)
test_n  = int(N * 0.2)
val_n   = N - train_n - test_n
```

- Train targets: `[0, train_n)`
- Validation targets: `[train_n, train_n + val_n)`
- Test targets: `[train_n + val_n, N)`
- Validation/test input windows may read historical context from the end of the preceding partition.
- Prediction targets must lie entirely within their own partition. No prediction window may cross a validation/test target boundary.

The TSlib-style boundaries should be implemented as:

```python
train_end = train_n
test_start = train_n + val_n
border1s = [0, train_end - seq_len, test_start - seq_len]
border2s = [train_end, test_start, N]
```

### 5.4 Normalization and metrics

1. CSV files retain raw numerical values; normalized values must not be written back to the data files.
2. Fit a separate `StandardScaler` for each domain using only the training partition's `OT` values (`sklearn` default population variance, `ddof=0`).
3. All models and prediction lengths for the same domain must use the same split rule. The scaler can be deterministically recomputed from the same training interval.
4. Validation/test data may only use the scaler fitted on the training partition. Fitting on the full dataset is prohibited.
5. The main results table should report MSE/MAE in standardized space to match common MM-TSFlib implementations. Metrics in original units may be reported in the appendix, but the two metric types must not be mixed.
6. The test set must not be used for hyperparameter tuning, epoch selection, text-configuration selection, or model selection.

### 5.5 Causal text protocol

- The main experiments use only `fact`. `preds` is retained for separate ablations and must not be mixed with `fact` and then described as “factual text.”
- For an input window `[s, s + seq_len - 1]`, only row-level text within that window may be read. The latest permitted row is index `s + seq_len - 1`.
- **Do not read text at index `s + seq_len`, which is the first prediction-horizon row.** The original MM-TSFlib `get_text()` implementation may read text from the prediction period when using `self.text[s_end]`; this protocol instead requires aggregation over the historical window only.
- Empty text remains empty; future text must not be used to fill it.
- If text is later filtered by its own internal dates, the text's availability time must be no later than the forecast origin.
- The `fact`/`preds` fields were automatically cleaned upstream and may contain irrelevant content, discussion of future events in historical documents, or generation noise. This release audits structure, the time axis, and target values; it does not claim that every text item has been manually verified semantically.

## 6. Domain inventory and exact splits

| Domain | Actual target | Frequency | N | `seq_len`/`label_len` | `pred_len` | Train / Val / Test | Date range |
|---|---|---|---:|---|---|---|---|
| Agriculture | U.S. Retail Broiler Composite | Monthly | 496 | 8/4 | 6, 8, 10, 12 | 347 / 50 / 99 | 1983-01-01 – 2024-04-01 |
| Climate | Contiguous U.S. Monthly Precipitation (inches) | Monthly | 496 | 8/4 | 6, 8, 10, 12 | 347 / 50 / 99 | 1983-01-01 – 2024-04-01 |
| Economy | U.S. International Trade Balance | Monthly | 423 | 8/4 | 6, 8, 10, 12 | 296 / 43 / 84 | 1989-01-01 – 2024-03-01 |
| Energy | U.S. Gasoline Price | Weekly | 1,479 | 36/18 | 12, 24, 36, 48 | 1,035 / 149 / 295 | 1996-01-01 – 2024-04-29 |
| Environment | New York Metropolitan Area Air Quality Index | Daily | 15,248 | 96/48 | 48, 96, 192, 336 | 10,673 / 1,526 / 3,049 | 1982-01-01 – 2023-09-30 |
| Health | U.S. Influenza-like Illness Patient Proportion | Weekly | 1,389 | 36/18 | 12, 24, 36, 48 | 972 / 140 / 277 | 1997-09-29 – 2024-05-06 |
| Security | U.S. FEMA Disaster and Emergency Grants | Monthly | 297 | 8/4 | 6, 8, 10, 12 | 207 / 31 / 59 | 1999-09-01 – 2024-05-01 |
| SocialGood | U.S. Unadjusted Unemployment Rate | Monthly | 892 | 8/4 | 6, 8, 10, 12 | 624 / 90 / 178 | 1950-01-01 – 2024-04-01 |
| Traffic | U.S. Vehicle Miles Traveled | Monthly | 531 | 8/4 | 6, 8, 10, 12 | 371 / 54 / 106 | 1980-01-01 – 2024-03-01 |

Exact target-date boundaries:

| Domain | Train targets | Validation targets | Test targets |
|---|---|---|---|
| Agriculture | 1983-01-01 – 2011-11-01 | 2011-12-01 – 2016-01-01 | 2016-02-01 – 2024-04-01 |
| Climate | 1983-01-01 – 2011-11-01 | 2011-12-01 – 2016-01-01 | 2016-02-01 – 2024-04-01 |
| Economy | 1989-01-01 – 2013-08-01 | 2013-09-01 – 2017-03-01 | 2017-04-01 – 2024-03-01 |
| Energy | 1996-01-01 – 2015-10-26 | 2015-11-02 – 2018-09-03 | 2018-09-10 – 2024-04-29 |
| Environment | 1982-01-01 – 2011-03-22 | 2011-03-23 – 2015-05-26 | 2015-05-27 – 2023-09-30 |
| Health | 1997-09-29 – 2016-05-09 | 2016-05-16 – 2019-01-14 | 2019-01-21 – 2024-05-06 |
| Security | 1999-09-01 – 2016-11-01 | 2016-12-01 – 2019-06-01 | 2019-07-01 – 2024-05-01 |
| SocialGood | 1950-01-01 – 2001-12-01 | 2002-01-01 – 2009-06-01 | 2009-07-01 – 2024-04-01 |
| Traffic | 1980-01-01 – 2010-11-01 | 2010-12-01 – 2015-05-01 | 2015-06-01 – 2024-03-01 |

## 7. Assertions required before training

```python
assert list(df.columns) == ['date', 'start_date', 'end_date', 'OT', 'fact', 'preds']
assert pd.to_datetime(df['date']).is_monotonic_increasing
assert not pd.to_datetime(df['date']).duplicated().any()
assert pd.to_numeric(df['OT'], errors='coerce').notna().all()
assert (pd.to_datetime(df['start_date']) <= pd.to_datetime(df['end_date'])).all()
```

Also verify that every horizon has a positive number of training, validation, and test windows, and that no prediction target crosses a partition boundary.

## 8. Completed audits

- All nine output files have the same schema.
- `date` is strictly ascending and has no duplicates.
- All `OT` values can be parsed as finite numeric values.
- Monthly, weekly, and daily frequencies are continuous, with no structural gaps.
- Before cleaning, `date`/`OT` match row by row between MM-TSFlib and SpecTF.
- Dates with ISO prefixes in the text do not occur later than the corresponding row's `end_date`.
- All splits and window protocols generate valid samples. Every horizon in every domain has a positive number of training, validation, and test windows, and prediction targets do not cross partition boundaries.
- An independent audit script was run on the final outputs; the count of structural errors was zero.

## 9. Known limitations that mechanical processing cannot fully eliminate

1. `fact`/`preds` are upstream automatically cleaned text and contain semantic noise. Not every text item should be treated as a manually verified fact. Mechanical scanning also found some `fact` rows that mention years later than the current row. These may describe plans or outlooks available at the time, or may be upstream misclassification or generation noise; they cannot be removed based on year alone. The paper should therefore call them “upstream-labeled factual text” and must not claim that every item has undergone manual factual verification.
2. The Climate label in the paper differs from the actual target in the public file. This release uses the actual meaning of the file: monthly precipitation.
3. The actual number of Environment rows differs from the sample count in the paper's table. This release retains the actual continuous data and does not truncate it arbitrarily to match the table.
4. Because Economy, Health, and SocialGood have been corrected, results from this release must not be mixed directly with published numbers obtained from uncorrected upstream data. All baselines and the proposed model must be rerun on this version.

## 10. SHA-256 hashes of the released files

| File | SHA-256 |
|---|---|
| `data/Agriculture/Agriculture.csv` | `3e3051391f457c480c368f327e87cae1d100b521171c6a980e275e53991d1d9c` |
| `data/Climate/Climate.csv` | `bc6e792f5490f5919431f560036fd13a17d9db55d6bcb2f84c647df7f5395e21` |
| `data/Economy/Economy.csv` | `444da8dc328a4d695259e47d664186370fd9a24d09dcb18cc8931f81e33d3ab4` |
| `data/Energy/Energy.csv` | `3d55a4df628cd378c3d985cf2a49a560dd02109a136c326c65efd272dd9d4daf` |
| `data/Environment/Environment.csv` | `da99bc4e5f3540b81e4610d641c1b54da6b17b1b018acc6991e7f1ff1e030b5c` |
| `data/Health/Health.csv` | `c708a79668d6a2759bb7d465109477f15cdaf750650ba06f76eecb9ef029dad1` |
| `data/Security/Security.csv` | `b9c5c343b15e3f89c3d76bc71db5f0d6d863c0407e61b3fad7a6d6dbb10a2758` |
| `data/SocialGood/SocialGood.csv` | `f1e5d8a9527633f32e639463b05955ab248b82907a400ec340ae9813b5fbd311` |
| `data/Traffic/Traffic.csv` | `b0eff9606cd5aea729fa60650a73aa0208e742543986a55862884e86c57132d2` |

## 11. SHA-256 hashes of the upstream source files

<details>
<summary>Expand to view upstream source hashes</summary>

- **Agriculture**
  - MM-TSFlib numerical file: `2a6d4d7ebfc280cc97a8797c2191423cb24e610bc3565a9ed66fbea490cdc3ec`
  - SpecTF `fact`/`preds`: `99135c58a8a67099970a990584c895530e3d56db309702271081981f976a71b6`
- **Climate**
  - MM-TSFlib numerical file: `6abc821276bb9a16d401b8797415e341c25a84709bdf23df6934b34017dc8dae`
  - SpecTF `fact`/`preds`: `d5c0cffd3bfb27296e5031d89e0630e585fc0f221fb9c9b85775b1b2e2fdb50c`
- **Economy**
  - MM-TSFlib numerical file: `1129dc381e56cd814cb61ca2e04bb9fc079c273c102f4b317768bb40099372a6`
  - SpecTF `fact`/`preds`: `024520a09e421b951e27e77e17b0e54eec9ae98288fdd8dd87ab82db93173c4a`
- **Energy**
  - MM-TSFlib numerical file: `99dd5214dd8cf73bf24fdfd3d5a1b4cbae3b6a49d5198ecaa2687af34ecd8284`
  - SpecTF `fact`/`preds`: `d32f6dca6818cf8cb415c15af77a9d8e63904a04bf2cd8243fde92ffb67e547f`
- **Environment**
  - MM-TSFlib numerical file: `e566c9182d06680de3751afd8a60f3f7ac8fe3c44e3d1df50b69de16cc2852df`
  - SpecTF `fact`/`preds`: `f8fce652d148ecddf251925023874933aa52c9bbc9041b35045340fa7ffa7f08`
- **Health**
  - MM-TSFlib numerical file: `899502b83b9891f071f31509f2b88f7062d21bb36809479193c1f89420881d04`
  - SpecTF `fact`/`preds`: `63b7bcd452c021edf241206262a239d063c908ba1115e3d23f0de70055f6aff4`
- **Security**
  - MM-TSFlib numerical file: `c708f54ca5a53cde72022b0bc3b466a3f088a5514bf52b022fd354ec016148c3`
  - SpecTF `fact`/`preds`: `f01af1376e25d182b2c8adc06436d864ae0abf5978aba22be7d2d598678cb837`
- **SocialGood**
  - MM-TSFlib numerical file: `e12b953337650da3f8a698798bef29e9e03fe2705dd8431319fb7daa214f4bcf`
  - SpecTF `fact`/`preds`: `e2b2cc58e0852760c7809923a61440e96fa2010d9e2d146185f07305d2252be0`
- **Traffic**
  - MM-TSFlib numerical file: `2699d780aa30c9582cb0a429407accf971c748f89e4f1f4b70dfdc148f58101e`
  - SpecTF `fact`/`preds`: `18cf0994f6b58736387547217b9af8113d9b7da9ac240146d0a6737c900b7287`

</details>

## 12. Freeze rules

- Subsequent models and baselines must read CSV files only from this directory.
- Individual model data loaders must not independently sort, impute, or reconstruct text.
- If a data error requiring a change is discovered, increment the dataset version and rerun all experiments. Do not overwrite this version while continuing to report results from the old version.
