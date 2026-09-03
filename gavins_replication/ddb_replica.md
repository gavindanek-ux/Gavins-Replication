# Gavin's DeepLOB Replica Documention

## Data Ingestion and Normalization

### Raw Data Source
Databento give MBO data as .dbn/.dbn.zst files. It gives a dataframe of individual order-book events (adds,cancels, trades, and fills), one row per event.

### Message Parsing
- 'A' -> AddOrderMessage which shows new resting orders that enter the book
- 'C' -> OrderCancelMessage shows an existing order's reduction in size or full removal.
- TradeMessage and FillMessage are no-ops so the book reconstruction relies on the adds and cancels to track resting liquididty

**Table 1 of the TTSTL paper confirms this is right, not a shortcut. An Order Executed maps to the Databento sequence T, F, then C. The trailing C is what carries the size reduction, so ignoring T and F loses no book information.**

### Book Reconstructing Mechanics

- AddOrderMessage -> enter_quote() insterts an order at (price, volume, order_id, and side)

- OrderCancelMessage -> cancel_quote() reduces or removes volume for that order_id

- Before either mutation is applied, pre_lob_event() fires before_lob_update(timestamp), which notifies every attached handler before the book changes — this is the hook the recorder taps into.

### Snapshot Extraction (deeplob_recorder.py)

- **cadlag** is the right continuous default that stamps the snapshot with current_lob.timestamp (the last event that actually changed the book). Essentially "this state has held since that past event, up until now"

- **caglad** is left continuous and stamps the snapshot with upcoming event's timestamp. Essentially "this is the state immediately before the jump at time t," but labeled with time t itself

**When multiple events share the same exact current_lob.timestamp, cadlag mode pops the previous record and keeps on the latest so you don't get duplicate rows.**


### What gets written per snapshot

For L = 10 levels per side: ASKp{}, ASKs{}, BIDP{l}, BIDs{} for i = 1...10-40 feature columns which are ordered by level, plus a **timestamp** column. 

- Missing levels get sentinel values: ask price -> 2^32, bid price -> 0 , missing volumes -> 0. 
- The recorder internally keeps 12 levels (max_depth = L + 2) but only 10 are exposed in the output columns. 
- The column order (ask price, ask volume, bid price, bid volume, per level) matches the $l^{(L)}(t)$ row vector the TTSTL paper defines, so this schema is the standard one.

new function mask_missing_levels(df) scans all 10 levels, and wherever ASKp{l} == 2^32 or BIDp{l} == 0, replaces that price and its paired volume with NaN. Now these masked cells won't pullute statistics

**After adding a restricted trading window (9:40 - 15:50, matching data_loader.py's ts_start/ts_end) to build_dataset, it found missing-level sentinels at approx 0% inside the window vs up to 6% at deep levels outside it. This was checked across 24 sample days on all 6 tickers.**

### Driver Script/ file conventions

- Iterates raw files in data.nosync/Raw_Data/ , extracts the date from the filename by splitting on - and writes output to data.nosync/Recorded_Data/<MMDDYYYY>_deeplob_data_<mode>.csv 

### 5 day rolling z-score normalization

We can see from section 3 of Zhang, et al. (DeepLOB) that we need to prevent lookahead bias and handle non-stationarity by using the summary statistics for $\mu_{d-5:d-1}$ and ${\sigma_{d-5:d-1}}$

$$
z_{d,t} = \frac{x_{d,t} - \mu_{d-5:d-1}}{\sigma_{d-5:d-1}}
$$

compute_prev_5day_stats now takes the full day list plus current_day_idx and slices d-5..d-1 itself, so the current day can't leak into its own stats. It raises if there aren't 5 prior days, and build_dataset drops those first 5 days.

Winsorization clips the z-scores at +/- 3 sigma.

**normalize_day edited so that a missing level now reads as "neutral" instead of leaving NaNs that would have broken the DeepLOBDataset tensor conversion.** Neutral is 0.0 here, which is the z-score of the mean.

### Smoothed Label Generation equation (4)

This compares the future moving average $m_+(t)$ against the past moving average $m_-(t)$ accross $k \in{10,20,50,100}$ ticks.

- added INVALID_LABEL = -1 as an explicit sentinel which means that "this tick's label is undefined for this horizon" 
- needed because the first k-1 and last k ticks of a day have no full rolling window. NaN fails both the > alpha and < -alpha checks, so those ticks were silently coming out as Stationary.
- alpha = 0.0002. TTSTL calls the threshold and the horizon tuning parameters with no value that holds across regimes, so treat it as a knob to retune per instrument.

### Windowing
- DeepLOBDataset now computes valid_indices: for every candidate 100-tick window and it checks wheter any of the 4 horizon labels at that window's target tick is invalid and excludes it if so. 

- Ivalid target ticks are filtered from which windows get offered as samples, not by dropping rows (wich would break the 100-tick contiguity the window needs)

- Each sample comes out as (1, 100, 40), the PyTorch channel-first version of DeepLOB's (100, 40, 1) input.

### End-to-end wiring (load_daily_lob, build_dataset)

**This takes a chronological list of per-day recorded CSV paths and loads and masks each day for each day index i >= 5. It then computes the past 5 day stats, normalizes that day, and generates its labels. 

It wraps it in a DeepLOBDataset and returns all days combined through torch.utils.data.ConcatDataset**

Per day the order is: trim to trading hours -> mask sentinels -> z-score against the previous 5 days -> label -> window.

Each day gets its own DeepLOBDataset so a 100-tick window never straddles two days.

### Validation

Ran the whole pipeline on 7 synthetic days built with pre-market, trading window, and post-market rows, with sentinels injected outside the window.

- the trim kept exactly the 2000 in-window rows per day
- k = 10 flagged 19 ticks per day as INVALID_LABEL, which matches 2k - 1 (9 at the start, 10 at the end)
- 0 samples leaked an INVALID_LABEL, and every sample came out (1, 100, 40)

### Evolving-Microstructural-toolkit-data Validation

- Used the same core mechanism as Juan's validation scripts and confirmed by hashing a real file both ways and got the identical result: SHA-256 hash of every file










