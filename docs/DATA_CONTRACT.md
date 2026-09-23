# Immutable Data Contract

**Contract status:** Frozen for Phase 2B implementation  
**Governing specification:** [`EXPERIMENT_SPEC.md`](./EXPERIMENT_SPEC.md), version 1.1

This contract defines the only accepted daily market-data representation for V2. It covers acquisition and canonicalization, not predictive modeling.

## Provider and acquisition boundary

- **Provider:** `yfinance`
- **Target assets:** AAPL, MSFT, GOOGL, NVDA
- **Market reference:** SPY
- **Frequency:** Daily U.S. exchange sessions
- **Requested start:** 2015-01-01, inclusive
- **Requested end:** 2025-12-31, inclusive
- **Provider request end:** 2026-01-01, because the provider's `end` argument is exclusive
- **Acquisition mode:** An explicit command only; importing the package must never start a download

The downloader requests each locked ticker explicitly and assigns ticker identity from that request, never from row position. No other asset may enter a canonical snapshot.

## Date and timezone convention

`date` is the U.S. exchange session date in `America/New_York`. Canonical files store it as a timezone-free ISO date (`YYYY-MM-DD`) because it represents a session label, not an intraday timestamp. Provider timestamps are converted to the exchange timezone before the time component is removed.

Rows are present only for observed provider trading sessions. Weekends and exchange holidays are not synthesized. The pipeline does not assume calendar-day adjacency; the next row for the same asset defines the next observed session.

## Canonical raw schema

| Field | Logical dtype | Rule |
|---|---|---|
| `date` | date | Parsed exchange session date; within the locked range |
| `asset` | string | One of AAPL, MSFT, GOOGL, NVDA, SPY |
| `open` | float64 | Finite, strictly positive adjusted price |
| `high` | float64 | Finite, strictly positive adjusted price |
| `low` | float64 | Finite, strictly positive adjusted price |
| `close` | float64 | Finite, strictly positive adjusted price |
| `volume` | float64-compatible numeric | Finite and non-negative; fractional values are tolerated at schema level even though provider volume is normally integral |

The canonical raw primary key is:

`(asset, date)`

Rows are sorted ascending by `asset`, then `date`. Duplicate keys are fatal. A normalizer may deterministically parse types, uppercase ticker labels, select the canonical fields, and sort rows; it may not impute, clip, deduplicate, or otherwise repair invalid observations silently.

## Price adjustment policy

Acquisition explicitly uses `auto_adjust=true`. Canonical `open`, `high`, `low`, and `close` therefore follow the provider's split- and dividend-adjusted OHLC policy. The manifest records this setting and the installed `yfinance` version. An unadjusted price, `Adj Close`, or a differently adjusted refresh must not be mixed into the same dataset version.

Because a provider may revise historical adjustment factors, any later refresh that changes canonical bytes is a new dataset version even when the requested dates are unchanged.

## Validation rules

A canonical snapshot is rejected if:

- a required field is absent;
- an asset is outside the locked universe;
- a date is unparseable, duplicated, on a weekend, or outside 2015-01-01 through 2025-12-31;
- rows are not canonically ordered after the explicit normalization step;
- an OHLC or volume value is non-numeric or non-finite;
- an OHLC value is non-positive;
- volume is negative;
- `high < open`, `high < close`, `low > open`, `low > close`, or `high < low`; or
- a required asset returns no rows in a complete downloaded snapshot.

The lightweight contract does not independently reconstruct the full exchange-holiday calendar. The provider supplies observed sessions; exact calendar reconciliation remains a validation report item when the real snapshot is acquired.

## Duplicate and missing-row policy

- Duplicate `(asset, date)` rows are rejected, never averaged or silently dropped.
- Missing OHLC values in an observed row are rejected.
- Missing target dates are not imputed.
- Missing trading rows are not created or forward/backward filled in raw data.
- If a target stock has a date with no matching SPY row, the feature join must either raise a clear error or retain an explicit missing indicator and missing values. It may not join by row position or backward-fill from a future SPY date.
- Provider outages, partial histories, and unexpected asset-specific gaps must be documented in the acquisition validation report.

## Raw versus processed files

`data/raw/` contains immutable, versioned canonical provider snapshots and adjacent manifests. Raw snapshots contain only normalized provider fields; they contain no targets, rolling features, regime labels, imputations, or fitted transformations.

`data/processed/` contains reproducible derivatives such as supervised targets and causal features. Processed artifacts must cite their raw dataset version and code/configuration revision. They may be regenerated; they must never replace or edit a raw snapshot.

Raw and processed data are ignored by Git by default. Their `.gitkeep` files preserve the directory layout without committing downloaded market data.

## Immutability, hashing, and versioning

Raw data must never be silently overwritten. Each acquisition writes a new filename containing a UTC retrieval timestamp and a prefix of the canonical CSV SHA-256. Files are created with exclusive-write semantics.

Every raw snapshot has a JSON manifest containing at least:

- provider;
- provider/library version when available;
- UTC retrieval timestamp;
- requested inclusive start and end dates;
- exclusive provider end date;
- returned minimum and maximum date per asset;
- row count per asset;
- canonical CSV filename and SHA-256; and
- explicit adjustment setting.

If a later acquisition produces different bytes, it is retained as a different version. Downstream artifacts identify the exact raw filename and hash used.

## Target and supervised-sample semantics

For asset \(i\), an origin row at observed session \(t\) is paired only with the next observed session for the same asset, \(t+1\):

\[
target\_log\_return_{i,t+1}=\log\left(\frac{close_{i,t+1}}{close_{i,t}}\right)
\]

`origin_date` is date \(t\); `target_date` is the next observed date for that asset. `target_direction` is 1 only when the log return is greater than zero and 0 otherwise. The last row for each asset is omitted because it has no known next-session target.

The canonical supervised-sample identity is:

`(asset, origin_date, target_date)`

Walk-forward membership is determined exclusively by `target_date`. No target may cross an asset boundary. The F1 test target period, 2025-01-01 through 2025-12-31, remains guarded from default development utilities.
