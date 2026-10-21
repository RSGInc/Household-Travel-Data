# Household allocation

`allocate_households.py` implements the socioeconomic allocation memo as a
reproducible, config-driven workflow:

1. Load the FAST TAZ polygons and 2020 Census blocks.
2. Allocate each block's Decennial population and occupied housing units using
   the configured interior-point, land-area, or tiered parcel method.
3. Scale the allocated 2020 totals to 2023 ACS tract population and household
   controls.
4. Derive weighted regional income quartiles from the configured PUMS PUMA,
   fractionally allocate ACS B19001 income bands across those quartiles, and
   weight the resulting income shares and other ACS distributions into each
   TAZ. Workers use tract-level B08202 because the API does not publish those
   cells at block-group geography.
5. Write the TAZ result plus block-conflict, tract-scaling, and block-group
   weighting audit tables.

## Overview

It uses the API helper functions from the household validation script/module to download census block data (or retrieve previously downloaded data from cache) and then allocates the blocks to TAZs using land-share areas or representative points. The allocated 2020 data is then scaled to the 2023 ACS tract controls, after which TAZ-level household-size, worker, child-presence, and income distributions (derives as quartiles using PUMS) are created. Writes outputs to CSV and GeoPackage, with optional summary audit files.

## Allocation workflow

- Retrieves 2020 Decennial Census block population and occupied-household data using the shared Census API and caching helpers.
- Allocates blocks using configurable centroid, land-area, or tiered methods. The default tiered method:
  1. Keeps blocks contained within one TAZ whole.
  2. Allocates split blocks using qualifying residential parcel counts.
  3. Falls back to land-area shares when no qualifying residential parcels are available.
- Supports reviewed block allocations through an optional manual block-share file.
- Independently scales the allocated household and total-population counts to 2023 ACS tract controls. Total population includes both household and group-quarters population.
- Supports manual TAZ household overrides in the configuration, including optional review comments. Override differences are redistributed among non-overridden TAZs within the contributing Census tracts, preserving the ACS tract household controls. Population and the original 2020 household allocation remain unchanged.
- Creates TAZ-level household-size, worker, child-presence, and income distributions. Income groups are regional quartiles derived from weighted PUMS records, with published ACS income bands fractionally allocated across the quartiles.
- Uses allocated block land area to construct distributions for TAZs with zero allocated households. Any remaining missing distribution is copied from the nearest valid TAZ, and every distribution is normalized to sum to one so it can support future-year household growth.

## Outputs and QA

The script writes the final TAZ allocation to CSV and GeoPackage, along with:

- Block allocation results
- Block conflict diagnostics
- Tract scaling audits
- Block-group weighting audits
- Manual household override audits
- PUMS income-quantile summaries
- ACS income-band allocation diagnostics
- A JSON run summary

Census and PUMS-derived inputs are cached and reused unless a refresh is requested.

## Run

From the repository root:

```powershell
uv run python household_allocation/allocate_households.py
```

The first run downloads the FNSB TIGER/Line block archive and calls the Census
API. These inputs are cached under `local/household_allocation/census`; later
runs reuse them unless `census.refresh_cache` is set to `true`. The Census API
key is read from `local/.env` as `CENSUS_API_KEY`.

Outputs are written to `household_allocation/output`. The GeoPackage and flat
CSV include `TOTHH`, `TOTPOP`, and the configured categorical share fields.
`TOTPOP` includes both household and group-quarters population. The source TAZ
field remains configurable independently; by default, output tables rename
`TAZID_26` to `TAZID` and place it first.

## Conflict handling

The default `allocation.method: tiered` follows the memo's conflict hierarchy.
Blocks intersecting one TAZ remain whole. Split blocks are divided in proportion
to qualifying residential parcel counts, using a point guaranteed to fall inside
each parcel to associate it with a block and TAZ. A split block with no qualifying
parcels is divided by land area. The configured `Assessing1` values identify
residential parcels; one qualifying parcel contributes one unit of proxy weight.

`block_conflicts.csv` lists every positive-area block/TAZ overlap, its land-area
share, block counts, allocation method, residential parcel count, and a
configurable material-conflict flag for review.

For the original winner-take-all method, set `allocation.method: centroid`. For
a blanket area allocation, set it to `area`. For reviewed exceptions, provide
`paths.manual_shares_file`; each listed block fully replaces its automatic
allocation and its `SHARE` values must sum to 1. Expected columns are `GEOID20`,
`TAZID_26`, and `SHARE`.

If an interior block point misses every TAZ because of boundary gaps or topology
differences, the centroid method assigns the block to its largest positive-area
TAZ overlap. The area method normalizes shares over the portion of the block
covered by the configured TAZs, so its shares still sum to 1 at study-area edges.

## TAZ household overrides

Reviewed final household totals can be entered under
`allocation.taz_household_overrides`, keyed by the source TAZ identifier:

```yaml
taz_household_overrides:
  "3091":
    value: 0
    comment: University
```

The allocator sets the TAZ's final `TOTHH` to the configured value and
redistributes the difference among non-overridden TAZs in each contributing
Census tract. This preserves the ACS tract household controls. It does not
change `HH_2020_ALLOC` or population totals. `manual_household_overrides.csv`
records the automatic, requested, and final household values along with the
optional comment.

## Income quantiles

The `income_quantiles` configuration selects the PUMS household file, PUMA, and
income-group labels. PUMS `HINCP` is adjusted with `ADJINC`, and the shared
income-allocation module derives weighted cut points from occupied,
positive-weight households.

ACS B19001 publishes income ranges rather than individual household incomes. A
published range can cross a PUMS cut point, so the allocator does not assign the
whole range to one group. Instead, it uses the selected PUMS households within
that range to split the ACS count fractionally across `INC1_PROP` through
`INC4_PROP`. This makes a different study area reusable by changing the PUMS
file and `puma_code`, without editing B19001 cell lists or dollar thresholds.

The ACS and PUMS CPI-U settings translate published ACS band edges into the
PUMS dollar year. Leave them equal when the ACS table and adjusted PUMS extract
use the same dollar year. The run writes `pums_income_quantiles.csv` and
`acs_income_fractional_allocation.csv` for QA.

The PUMS classification and ACS band allocations are stored in the configured
`income_quantiles.cache_dir`, shared with the HTS and destination-choice
workflows. Inputs and settings are fingerprinted, so a relevant change creates
a new entry automatically. Set `refresh_cache: true` only to force a rebuild.

## Distribution weighting and fallbacks

The allocator processes each household dimension independently using the
following order:

1. Weight complete ACS source distributions using allocated Decennial occupied
   housing units.
2. If an entire TAZ has no allocated households, use its allocated block land
   area as the weight instead.
3. Exclude a source row from a dimension when any category in that dimension is
   missing; incomplete shares are not mixed into the result.
4. If no valid source remains and
   `allocation.fill_missing_distributions_from_nearest_taz` is `true`, copy that
   dimension from the nearest TAZ with complete shares.
5. Normalize each dimension so its categories sum to 1.

The nearest-TAZ fallback is applied separately to household size, income,
workers, and presence of children. Distance is measured from an interior point
of the missing TAZ to each valid TAZ polygon. Consequently, a TAZ with zero
totals (`TOTHH = 0` and `TOTPOP = 0`) still retains valid, generally nonzero
`*_PROP` values. Its current category counts remain zero when those shares are
multiplied by zero households, while the shares remain usable if households are
added in a future scenario.
