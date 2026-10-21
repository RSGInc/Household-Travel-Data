"""Allocate Decennial Census households/population and ACS shares to FAST TAZs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SHARED_SCRIPTS = PROJECT_ROOT / "FAST" / "SharedData" / "Scripts"
# The script is also run directly (rather than only as an installed package), so
# make both repository modules and legacy FAST shared scripts importable.
for import_path in (PROJECT_ROOT, SHARED_SCRIPTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from hh_validation_data import (  # noqa: E402
    census_api_get,
    load_or_fetch_raw,
    state_fips,
    write_json_cache,
)
from hts_processing.income_allocation import (  # noqa: E402
    IncomeBandAllocation,
    PumsIncomeClassification,
    load_or_build_income_band_allocation,
    load_or_build_pums_income_classification,
)


# Each entry maps an ACS estimate field to its label and half-open dollar range.
# B19001_001E is the table total, so only detail cells 002E through 017E appear.
PUBLISHED_INCOME_BANDS = {
    "B19001": {
        "B19001_002E": ("Less than $10,000", -np.inf, 10_000),
        "B19001_003E": ("$10,000 to $14,999", 10_000, 15_000),
        "B19001_004E": ("$15,000 to $19,999", 15_000, 20_000),
        "B19001_005E": ("$20,000 to $24,999", 20_000, 25_000),
        "B19001_006E": ("$25,000 to $29,999", 25_000, 30_000),
        "B19001_007E": ("$30,000 to $34,999", 30_000, 35_000),
        "B19001_008E": ("$35,000 to $39,999", 35_000, 40_000),
        "B19001_009E": ("$40,000 to $44,999", 40_000, 45_000),
        "B19001_010E": ("$45,000 to $49,999", 45_000, 50_000),
        "B19001_011E": ("$50,000 to $59,999", 50_000, 60_000),
        "B19001_012E": ("$60,000 to $74,999", 60_000, 75_000),
        "B19001_013E": ("$75,000 to $99,999", 75_000, 100_000),
        "B19001_014E": ("$100,000 to $124,999", 100_000, 125_000),
        "B19001_015E": ("$125,000 to $149,999", 125_000, 150_000),
        "B19001_016E": ("$150,000 to $199,999", 150_000, 200_000),
        "B19001_017E": ("$200,000 or more", 200_000, np.inf),
    }
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve configured paths relative to the config file."""
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for name in (
        "taz_file",
        "parcel_file",
        "census_cache_dir",
        "output_dir",
        "manual_shares_file",
    ):
        value = config["paths"].get(name)
        if value:
            config["paths"][name] = str((config_path.parent / value).resolve())
    block_path = config["block_geography"]["path"]
    config["block_geography"]["path"] = str((config_path.parent / block_path).resolve())
    env_file = config["census"].get("env_file")
    if env_file:
        config["census"]["env_file"] = str((config_path.parent / env_file).resolve())
    income_settings = config.get("income_quantiles", {})
    pums_file = income_settings.get("pums_household_file")
    if pums_file:
        income_settings["pums_household_file"] = str(
            (config_path.parent / pums_file).resolve()
        )
    cache_dir = income_settings.get("cache_dir", "../local/income_quantile_cache")
    income_settings["cache_dir"] = str((config_path.parent / cache_dir).resolve())
    refresh_cache = income_settings.get("refresh_cache", False)
    if not isinstance(refresh_cache, bool):
        raise ValueError("income_quantiles.refresh_cache must be true or false.")
    income_settings["refresh_cache"] = refresh_cache
    config["_config_path"] = str(config_path)
    return config


def load_local_env(path: str | Path | None) -> None:
    """Load simple KEY=VALUE entries without overwriting the process environment."""
    if not path or not Path(path).exists():
        return
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def read_vector(
    path: str | Path, columns: list[str] | None = None
) -> gpd.GeoDataFrame:
    """Read a vector file, including a shapefile nested inside a zip archive."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Vector input not found: {source}")
    read_options = {"columns": columns} if columns is not None else {}
    if source.suffix.lower() != ".zip":
        return gpd.read_file(source, **read_options)
    with ZipFile(source) as archive:
        shapefiles = [
            name for name in archive.namelist() if name.lower().endswith(".shp")
        ]
    if len(shapefiles) != 1:
        raise ValueError(f"Expected one shapefile in {source}; found {shapefiles}")
    return gpd.read_file(
        f"zip://{source.resolve()}!{shapefiles[0]}", **read_options
    )


def ensure_block_geography(settings: dict[str, Any]) -> Path:
    """Return the cached TIGER block archive, downloading it when absent."""
    path = Path(settings["path"])
    if path.exists():
        print(f"Using cached block geography: {path}")
        return path
    url = settings.get("download_url")
    if not url:
        raise FileNotFoundError(
            f"Block geography not found and no download URL set: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading block geography: {url}")
    urllib.request.urlretrieve(url, path)
    return path


def geography_query(geography: str, state: str, county: str) -> dict[str, str]:
    if geography == "block":
        return {"for": "block:*", "in": f"state:{state} county:{county} tract:*"}
    if geography == "block group":
        return {
            "for": "block group:*",
            "in": f"state:{state} county:{county} tract:*",
        }
    if geography == "tract":
        return {"for": "tract:*", "in": f"state:{state} county:{county}"}
    raise ValueError(f"Unsupported Census geography: {geography}")


def census_cache_path(
    cache_dir: Path, dataset: str, year: int, state: str, county: str, geography: str
) -> Path:
    dataset_slug = dataset.replace("/", "_")
    geo_slug = geography.replace(" ", "_")
    return cache_dir / f"{dataset_slug}_{year}_{state}_{county}_{geo_slug}.json"


def fetch_geography(
    cache_dir: Path,
    dataset: str,
    year: int,
    state: str,
    counties: list[str],
    geography: str,
    fields: list[str],
    refresh: bool,
) -> pd.DataFrame:
    """Fetch/cache one Census dataset and geography for all configured counties."""
    frames: list[pd.DataFrame] = []
    for county in counties:
        path = census_cache_path(cache_dir, dataset, year, state, county, geography)

        def fetch() -> list[dict[str, Any]]:
            return census_api_get(
                year,
                fields,
                geography_query(geography, state, county),
                dataset=dataset,
            )

        records = load_or_fetch_raw(
            path,
            fetch,
            refresh_cache=refresh,
        )
        # A cache created by an older configuration may be valid JSON but omit
        # newly requested variables. Refresh it instead of silently returning NaNs.
        missing = [field for field in fields if records and field not in records[0]]
        if missing:
            print(
                f"Cached Census response is missing {missing}; refetching {path.name}"
            )
            records = fetch()
            write_json_cache(path, records)
        frames.append(pd.DataFrame(records))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def census_geoid(frame: pd.DataFrame, geography: str) -> pd.Series:
    # Census API geography components are strings by design. Concatenating them
    # preserves leading zeroes that would be lost by numeric conversion.
    result = frame["state"].astype(str) + frame["county"].astype(str)
    if geography in {"tract", "block group", "block"}:
        result += frame["tract"].astype(str)
    if geography == "block group":
        result += frame["block group"].astype(str)
    elif geography == "block":
        result += frame["block"].astype(str)
    return result


def numeric_estimates(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert Census estimates to numbers and discard negative missing sentinels."""
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result.loc[result[column] < 0, column] = np.nan
    return result


def prepare_decennial_blocks(
    raw: pd.DataFrame, variables: dict[str, str]
) -> pd.DataFrame:
    fields = list(variables.values())
    result = numeric_estimates(raw, fields)
    result["GEOID20"] = census_geoid(result, "block")
    result["TRACT_GEOID"] = result["GEOID20"].str[:11]
    result["BLOCK_GROUP_GEOID"] = result["GEOID20"].str[:12]
    return result[["GEOID20", "TRACT_GEOID", "BLOCK_GROUP_GEOID", *fields]].rename(
        columns={
            variables["households"]: "HH_2020",
            variables["population"]: "POP_2020",
        }
    )


def prepare_tract_controls(
    raw: pd.DataFrame, variables: dict[str, str]
) -> pd.DataFrame:
    fields = list(variables.values())
    result = numeric_estimates(raw, fields)
    result["TRACT_GEOID"] = census_geoid(result, "tract")
    return result[["TRACT_GEOID", *fields]].rename(
        columns={variables["households"]: "ACS_HH", variables["population"]: "ACS_POP"}
    )


def distribution_fields(specs: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    for spec in specs.values():
        fields.append(spec["denominator"])
        if source_name := spec.get("published_income_bands"):
            fields.extend(_published_income_bands(source_name))
            continue
        for source_fields in spec["categories"].values():
            fields.extend(source_fields)
    return list(dict.fromkeys(fields))


def _published_income_bands(
    source_name: str,
) -> dict[str, tuple[str, float, float]]:
    name = str(source_name).upper()
    if name not in PUBLISHED_INCOME_BANDS:
        choices = ", ".join(sorted(PUBLISHED_INCOME_BANDS))
        raise ValueError(
            f"Unsupported published income-band source {source_name!r}; "
            f"expected one of: {choices}"
        )
    return PUBLISHED_INCOME_BANDS[name]


def build_income_quantile_allocations(
    config: dict[str, Any], specs: dict[str, Any]
) -> tuple[
    PumsIncomeClassification | None,
    dict[str, IncomeBandAllocation],
]:
    """Build PUMS-derived allocations for configured published income tables."""
    dynamic_specs = {
        name: spec
        for name, spec in specs.items()
        if spec.get("published_income_bands")
    }
    if not dynamic_specs:
        return None, {}

    settings = config.get("income_quantiles")
    if not settings:
        raise ValueError(
            "income_quantiles settings are required when a distribution uses "
            "published_income_bands."
        )
    required = {
        "pums_household_file",
        "puma_code",
        "acs_income_cpi_u",
        "pums_income_cpi_u",
    }
    if missing := required - set(settings):
        raise ValueError(
            "income_quantiles is missing settings: " + ", ".join(sorted(missing))
        )
    acs_cpi_u = float(settings["acs_income_cpi_u"])
    pums_cpi_u = float(settings["pums_income_cpi_u"])
    if acs_cpi_u <= 0 or pums_cpi_u <= 0:
        raise ValueError("income_quantiles CPI-U values must be positive.")

    groups = settings.get("income_groups", ["INC1", "INC2", "INC3", "INC4"])
    cache_dir = settings.get(
        "cache_dir", PROJECT_ROOT / "local" / "income_quantile_cache"
    )
    refresh_cache = settings.get("refresh_cache", False)
    if not isinstance(refresh_cache, bool):
        raise ValueError("income_quantiles.refresh_cache must be true or false.")
    classification = load_or_build_pums_income_classification(
        settings["pums_household_file"],
        puma_code=int(settings["puma_code"]),
        cache_dir=cache_dir,
        income_groups=groups,
        refresh_cache=refresh_cache,
    )
    # Convert ACS band edges into the dollar year used by adjusted PUMS income;
    # equal CPI-U values intentionally produce no adjustment.
    inflation_factor = pums_cpi_u / acs_cpi_u
    allocations: dict[str, IncomeBandAllocation] = {}
    for name, spec in dynamic_specs.items():
        source_name = spec["published_income_bands"]
        allocations[name] = load_or_build_income_band_allocation(
            classification,
            _published_income_bands(source_name),
            cache_dir,
            inflation_factor=inflation_factor,
            band_key_column="source_field",
            band_label_column="source_income_band",
            empty_band_context=f"{source_name} income band",
            refresh_cache=refresh_cache,
        )
        # Income category columns cannot be known until the configured PUMS
        # classification is built. Mutating the in-memory spec here lets the
        # generic distribution code treat income like every other dimension.
        spec["categories"] = {
            f"{income_group}_PROP": income_group
            for income_group in classification.income_groups
        }
    return classification, allocations


def _prepare_fractional_income_categories(
    result: pd.DataFrame,
    spec: dict[str, Any],
    allocation: IncomeBandAllocation,
) -> pd.DataFrame:
    """Convert published income-band counts to PUMS quantile-group counts."""
    bands = _published_income_bands(spec["published_income_bands"])
    source_fields = list(bands)
    income_groups = list(spec["categories"].values())
    shares = (
        allocation.allocations.pivot(
            index="source_field",
            columns="income_group",
            values="income_allocation_share",
        )
        .reindex(index=source_fields, columns=income_groups, fill_value=0.0)
        .fillna(0.0)
    )
    output = pd.DataFrame(index=result.index)
    for output_column, income_group in spec["categories"].items():
        # One published ACS band can straddle a PUMS quantile boundary. Multiply
        # its household count by the PUMS-derived shares instead of assigning the
        # entire band to a single income group.
        output[output_column] = result[source_fields].multiply(
            shares[income_group], axis=1
        ).sum(axis=1, min_count=1)
    return output


def prepare_geographic_distributions(
    raw: pd.DataFrame,
    specs: dict[str, Any],
    geography: str = "block group",
    income_allocations: dict[str, IncomeBandAllocation] | None = None,
) -> pd.DataFrame:
    fields = distribution_fields(specs)
    result = numeric_estimates(raw, fields)
    geoid_column = "BLOCK_GROUP_GEOID" if geography == "block group" else "TRACT_GEOID"
    result[geoid_column] = census_geoid(result, geography)
    output = result[[geoid_column]].copy()
    income_allocations = income_allocations or {}
    for name, spec in specs.items():
        denominator = result[spec["denominator"]]
        category_columns: list[str] = []
        if spec.get("published_income_bands"):
            if name not in income_allocations:
                raise ValueError(
                    f"Distribution {name!r} requires a PUMS income-band allocation."
                )
            fractional = _prepare_fractional_income_categories(
                result, spec, income_allocations[name]
            )
            output[fractional.columns] = fractional
            category_columns.extend(fractional.columns)
        else:
            for output_column, source_fields in spec["categories"].items():
                output[output_column] = result[source_fields].sum(
                    axis=1, min_count=1
                )
                category_columns.append(output_column)
        if remainder := spec.get("remainder"):
            output[remainder] = (
                denominator - output[category_columns].sum(axis=1)
            ).clip(lower=0)
            category_columns.append(remainder)
        output[category_columns] = output[category_columns].div(
            denominator.replace(0, np.nan), axis=0
        )
        # Re-normalize after clipping a computed remainder or encountering
        # inconsistent published cells so every usable distribution sums to one.
        # A zero denominator remains missing and can be filled later from a TAZ.
        row_sum = output[category_columns].sum(axis=1).replace(0, np.nan)
        output[category_columns] = output[category_columns].div(row_sum, axis=0)
    return output


def intersect_blocks_and_zones(
    blocks: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    zone_id: str,
    tolerance: float = 0.0,
) -> pd.DataFrame:
    """Return positive-area block/zone intersections and normalized area shares."""
    intersections = gpd.overlay(
        blocks[["GEOID20", "geometry"]],
        zones[[zone_id, "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    intersections["INTERSECTION_AREA"] = intersections.geometry.area
    intersections = intersections.loc[
        intersections["INTERSECTION_AREA"] > tolerance
    ].copy()
    # Normalize over the portion covered by configured TAZs, not the block's
    # original area. Thus area-mode shares sum to one even at study-area edges.
    covered_area = intersections.groupby("GEOID20")["INTERSECTION_AREA"].transform(
        "sum"
    )
    intersections["AREA_SHARE"] = intersections["INTERSECTION_AREA"] / covered_area
    return pd.DataFrame(intersections.drop(columns="geometry"))


def centroid_allocations(
    blocks: gpd.GeoDataFrame, zones: gpd.GeoDataFrame, zone_id: str
) -> pd.DataFrame:
    """Assign each block using an interior representative point."""
    points = blocks[["GEOID20", "geometry"]].copy()
    # representative_point() is guaranteed to lie inside the block; a geometric
    # centroid can fall outside concave or multipart polygons.
    points.geometry = points.geometry.representative_point()
    joined = gpd.sjoin(
        points, zones[[zone_id, "geometry"]], how="left", predicate="within"
    )
    joined = joined.dropna(subset=[zone_id]).sort_values(["GEOID20", zone_id])
    joined = joined.drop_duplicates("GEOID20", keep="first")
    return pd.DataFrame(
        {"GEOID20": joined["GEOID20"], zone_id: joined[zone_id], "SHARE": 1.0}
    )


def tiered_allocations(
    blocks: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
    intersections: pd.DataFrame,
    zone_id: str,
    residential_column: str,
    residential_values: list[str],
) -> pd.DataFrame:
    """Allocate split blocks by residential parcel counts, then land area."""
    if residential_column not in parcels.columns:
        raise ValueError(
            f"Parcel input is missing residential field {residential_column!r}"
        )
    if not residential_values:
        raise ValueError("allocation.residential_parcel_values cannot be empty")

    # Only blocks with multiple positive-area TAZ overlaps need a proxy. Keeping
    # single-TAZ blocks whole avoids introducing parcel-data noise unnecessarily.
    pairs = intersections[["GEOID20", zone_id, "AREA_SHARE"]].drop_duplicates(
        ["GEOID20", zone_id]
    )
    zone_counts = pairs.groupby("GEOID20")[zone_id].transform("nunique")
    single_zone = pairs.loc[zone_counts == 1, ["GEOID20", zone_id]].copy()
    single_zone["SHARE"] = 1.0
    single_zone["ALLOCATION_METHOD"] = "single_zone"
    single_zone["RESIDENTIAL_PARCEL_COUNT"] = np.nan

    conflicted_blocks = set(pairs.loc[zone_counts > 1, "GEOID20"])
    residential = parcels.loc[
        parcels[residential_column].isin(residential_values)
        & parcels.geometry.notna()
        & ~parcels.geometry.is_empty,
        ["geometry"],
    ].copy()
    residential["_PARCEL_ID"] = np.arange(len(residential))
    # Represent each parcel once so a polygon spanning a TAZ boundary cannot
    # contribute proxy weight to multiple zones.
    residential.geometry = residential.geometry.representative_point()

    parcel_allocations = pd.DataFrame()
    if not residential.empty and conflicted_blocks:
        # Use intersects rather than within so a point exactly on a boundary is
        # retained. Sorting before de-duplication gives boundary ties a stable,
        # deterministic GEOID/TAZID winner.
        block_hits = gpd.sjoin(
            residential,
            blocks.loc[
                blocks["GEOID20"].isin(conflicted_blocks),
                ["GEOID20", "geometry"],
            ],
            how="inner",
            predicate="intersects",
        )
        block_hits = (
            block_hits.sort_values(["_PARCEL_ID", "GEOID20"])
            .drop_duplicates("_PARCEL_ID")[["_PARCEL_ID", "GEOID20"]]
        )
        zone_hits = gpd.sjoin(
            residential,
            zones[[zone_id, "geometry"]],
            how="inner",
            predicate="intersects",
        )
        zone_hits = (
            zone_hits.sort_values(["_PARCEL_ID", zone_id])
            .drop_duplicates("_PARCEL_ID")[["_PARCEL_ID", zone_id]]
        )
        parcel_counts = (
            block_hits.merge(zone_hits, on="_PARCEL_ID", how="inner")
            # Exclude block/TAZ combinations removed by the positive-area
            # intersection tolerance.
            .merge(pairs[["GEOID20", zone_id]], on=["GEOID20", zone_id])
            .groupby(["GEOID20", zone_id], as_index=False)
            .size()
            .rename(columns={"size": "RESIDENTIAL_PARCEL_COUNT"})
        )
        if not parcel_counts.empty:
            # Every qualifying parcel contributes one unit of proxy weight; the
            # resulting shares preserve the Census total for the parent block.
            parcel_counts["SHARE"] = parcel_counts[
                "RESIDENTIAL_PARCEL_COUNT"
            ] / parcel_counts.groupby("GEOID20")[
                "RESIDENTIAL_PARCEL_COUNT"
            ].transform("sum")
            parcel_counts["ALLOCATION_METHOD"] = "residential_parcels"
            parcel_allocations = parcel_counts

    parcel_blocks = set(parcel_allocations.get("GEOID20", pd.Series(dtype=str)))
    # The memo's land-area option is the deterministic fallback only when a
    # split block has no qualifying residential parcel point in a valid TAZ.
    fallback_blocks = conflicted_blocks - parcel_blocks
    fallback = pairs.loc[
        pairs["GEOID20"].isin(fallback_blocks),
        ["GEOID20", zone_id, "AREA_SHARE"],
    ].rename(columns={"AREA_SHARE": "SHARE"})
    fallback["ALLOCATION_METHOD"] = "land_area_fallback"
    fallback["RESIDENTIAL_PARCEL_COUNT"] = 0

    allocations = pd.concat(
        [single_zone, parcel_allocations, fallback], ignore_index=True
    )
    # Enforce both completeness and conservation before Census counts are
    # multiplied by these shares downstream.
    share_sums = allocations.groupby("GEOID20")["SHARE"].sum()
    if set(share_sums.index) != set(pairs["GEOID20"]) or not np.allclose(
        share_sums, 1.0
    ):
        raise ValueError("Tiered block allocation shares must sum to 1")
    return allocations.sort_values(["GEOID20", zone_id]).reset_index(drop=True)


def apply_manual_shares(
    allocations: pd.DataFrame,
    manual_path: str | Path | None,
    zone_id: str,
    valid_blocks: set[str],
    valid_zones: set[str],
) -> pd.DataFrame:
    if not manual_path:
        return allocations
    manual = pd.read_csv(manual_path, dtype={"GEOID20": str, zone_id: str})
    required = {"GEOID20", zone_id, "SHARE"}
    if missing := required - set(manual.columns):
        raise ValueError(f"Manual shares file is missing columns: {sorted(missing)}")
    manual["GEOID20"] = manual["GEOID20"].str.zfill(15)
    manual[zone_id] = manual[zone_id].astype(str)
    manual["SHARE"] = pd.to_numeric(manual["SHARE"], errors="raise")
    unknown_blocks = set(manual["GEOID20"]) - valid_blocks
    unknown_zones = set(manual[zone_id]) - valid_zones
    if unknown_blocks or unknown_zones:
        raise ValueError(
            f"Manual shares contain unknown blocks {sorted(unknown_blocks)} "
            f"or zones {sorted(unknown_zones)}"
        )
    sums = manual.groupby("GEOID20")["SHARE"].sum()
    if not np.allclose(sums, 1.0):
        raise ValueError("Manual SHARE values must sum to 1 for every GEOID20")
    replaced = set(manual["GEOID20"])
    manual_rows = manual[["GEOID20", zone_id, "SHARE"]].copy()
    manual_rows["ALLOCATION_METHOD"] = "manual"
    manual_rows["RESIDENTIAL_PARCEL_COUNT"] = np.nan
    # A reviewed block replaces all of its automatic rows. This permits both a
    # new winner-takes-all assignment and an explicit multi-TAZ split.
    return pd.concat(
        [
            allocations.loc[~allocations["GEOID20"].isin(replaced)],
            manual_rows,
        ],
        ignore_index=True,
    )


def scale_to_tract_controls(
    allocations: pd.DataFrame, tract_controls: pd.DataFrame, zone_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply separate household and population tract expansion factors."""
    allocated = allocations.copy()
    for source, output in (
        ("HH_2020", "HH_2020_ALLOC"),
        ("POP_2020", "POP_2020_ALLOC"),
    ):
        allocated[output] = allocated[source].fillna(0) * allocated["SHARE"]
    dec = allocated.groupby("TRACT_GEOID", as_index=False)[
        ["HH_2020_ALLOC", "POP_2020_ALLOC"]
    ].sum()
    audit = tract_controls.merge(dec, on="TRACT_GEOID", how="outer").fillna(0)
    # Households and population get independent factors because their changes
    # between the Decennial Census and ACS control year need not be proportional.
    for control, base, factor in (
        ("ACS_HH", "HH_2020_ALLOC", "HH_FACTOR"),
        ("ACS_POP", "POP_2020_ALLOC", "POP_FACTOR"),
    ):
        impossible = (audit[base] == 0) & (audit[control] > 0)
        if impossible.any():
            tracts = audit.loc[impossible, "TRACT_GEOID"].tolist()
            raise ValueError(
                f"Cannot scale positive {control} controls from zero in tracts {tracts}"
            )
        audit[factor] = np.where(audit[base] > 0, audit[control] / audit[base], 0.0)
    allocated = allocated.merge(
        audit[["TRACT_GEOID", "HH_FACTOR", "POP_FACTOR"]], on="TRACT_GEOID", how="left"
    )
    allocated["TOTHH"] = allocated["HH_2020_ALLOC"] * allocated["HH_FACTOR"]
    # HHPOP is a historical internal name. It is total population—including
    # group quarters—not household population, and is renamed TOTPOP on output.
    allocated["HHPOP"] = allocated["POP_2020_ALLOC"] * allocated["POP_FACTOR"]
    scaled = allocated.groupby("TRACT_GEOID")[["TOTHH", "HHPOP"]].sum()
    audit = audit.merge(scaled, on="TRACT_GEOID", how="left").fillna(
        {"TOTHH": 0, "HHPOP": 0}
    )
    audit["HH_DIFFERENCE"] = audit["TOTHH"] - audit["ACS_HH"]
    audit["POP_DIFFERENCE"] = audit["HHPOP"] - audit["ACS_POP"]
    return allocated, audit


def apply_taz_household_overrides(
    allocations: pd.DataFrame,
    overrides: dict[str, Any] | None,
    zone_id: str,
    valid_zones: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Set reviewed TAZ household totals while preserving each tract control."""
    audit_columns = [
        zone_id,
        "TOTHH_AUTOMATIC",
        "TOTHH_OVERRIDE",
        "TOTHH_FINAL",
        "COMMENT",
    ]
    if not isinstance(overrides, dict):
        if overrides is None:
            return allocations, pd.DataFrame(columns=audit_columns)
        raise ValueError("allocation.taz_household_overrides must be a mapping")
    if not overrides:
        return allocations, pd.DataFrame(columns=audit_columns)

    normalized: dict[str, dict[str, Any]] = {}
    for raw_zone, settings in overrides.items():
        zone = str(raw_zone)
        if not isinstance(settings, dict) or "value" not in settings:
            raise ValueError(
                f"Household override for TAZ {zone} must contain a value"
            )
        try:
            value = float(settings["value"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Household override value for TAZ {zone} must be numeric"
            ) from error
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"Household override value for TAZ {zone} must be nonnegative"
            )
        comment = settings.get("comment", "")
        if comment is None:
            comment = ""
        if not isinstance(comment, str):
            raise ValueError(
                f"Household override comment for TAZ {zone} must be text"
            )
        normalized[zone] = {"value": value, "comment": comment}

    unknown_zones = set(normalized) - {str(zone) for zone in valid_zones}
    if unknown_zones:
        raise ValueError(
            f"Household overrides contain unknown zones: {sorted(unknown_zones)}"
        )

    result = allocations.copy()
    result[zone_id] = result[zone_id].astype(str)
    result["_TOTHH_AUTOMATIC"] = result["TOTHH"]
    automatic_totals = result.groupby(zone_id)["_TOTHH_AUTOMATIC"].sum()

    for zone, settings in normalized.items():
        zone_rows = result[zone_id] == zone
        automatic_total = float(automatic_totals.get(zone, 0.0))
        override_value = settings["value"]
        if automatic_total > 0:
            result.loc[zone_rows, "TOTHH"] *= override_value / automatic_total
        elif override_value > 0:
            raise ValueError(
                "Cannot assign a positive household override to "
                f"zero-household TAZ {zone}"
            )

    overridden_zones = set(normalized)
    for tract, tract_rows in result.groupby("TRACT_GEOID").groups.items():
        tract_rows = list(tract_rows)
        tract_zone_ids = result.loc[tract_rows, zone_id]
        direct_override = tract_zone_ids.isin(overridden_zones)
        override_rows = tract_zone_ids.index[direct_override]
        recipient_rows = tract_zone_ids.index[~direct_override]
        automatic_control = result.loc[tract_rows, "_TOTHH_AUTOMATIC"].sum()
        overridden_total = result.loc[override_rows, "TOTHH"].sum()
        remaining_control = automatic_control - overridden_total
        if remaining_control < -1e-9:
            raise ValueError(
                f"Household overrides exceed the household control for tract {tract}"
            )
        remaining_control = max(float(remaining_control), 0.0)
        recipient_total = result.loc[recipient_rows, "_TOTHH_AUTOMATIC"].sum()
        if recipient_total > 0:
            result.loc[recipient_rows, "TOTHH"] = (
                result.loc[recipient_rows, "_TOTHH_AUTOMATIC"]
                * remaining_control
                / recipient_total
            )
        elif not np.isclose(remaining_control, 0.0):
            raise ValueError(
                f"Household overrides leave no TAZ to receive households in tract {tract}"
            )

    tract_automatic = result.groupby("TRACT_GEOID")["_TOTHH_AUTOMATIC"].sum()
    tract_final = result.groupby("TRACT_GEOID")["TOTHH"].sum()
    if not np.allclose(tract_final, tract_automatic, rtol=1e-10, atol=1e-8):
        raise ValueError(
            "Household overrides must preserve each tract household control"
        )

    final_totals = result.groupby(zone_id)["TOTHH"].sum()
    audit_rows = []
    for zone, settings in normalized.items():
        final_total = float(final_totals.get(zone, 0.0))
        if not np.isclose(final_total, settings["value"]):
            raise ValueError(f"Could not apply household override for TAZ {zone}")
        audit_rows.append(
            {
                zone_id: zone,
                "TOTHH_AUTOMATIC": float(automatic_totals.get(zone, 0.0)),
                "TOTHH_OVERRIDE": settings["value"],
                "TOTHH_FINAL": final_total,
                "COMMENT": settings["comment"],
            }
        )
    return result.drop(columns="_TOTHH_AUTOMATIC"), pd.DataFrame(
        audit_rows, columns=audit_columns
    )


def inherit_distributions(
    allocations: pd.DataFrame,
    distribution_tables: dict[str, pd.DataFrame],
    specs: dict[str, Any],
    zone_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weight ACS block-group shares by allocated households, then land area."""
    share_columns = [
        column
        for spec in specs.values()
        for column in [
            *spec["categories"],
            *([spec["remainder"]] if spec.get("remainder") else []),
        ]
    ]
    joined = allocations.copy()
    for geography, table in distribution_tables.items():
        geoid_column = (
            "BLOCK_GROUP_GEOID" if geography == "block group" else "TRACT_GEOID"
        )
        joined = joined.merge(table, on=geoid_column, how="left")
    # Use allocated Decennial occupied units as the small-area blending weight.
    # If an entire TAZ has no households, allocated land area supplies a stable
    # fallback so future-development zones can still inherit a distribution.
    joined["BG_WEIGHT"] = joined["HH_2020_ALLOC"]
    hh_sums = joined.groupby(zone_id)["BG_WEIGHT"].transform("sum")
    joined.loc[hh_sums <= 0, "BG_WEIGHT"] = joined.loc[hh_sums <= 0, "LAND_AREA_ALLOC"]
    bg_weights = joined.groupby([zone_id, "BLOCK_GROUP_GEOID"], as_index=False)[
        "BG_WEIGHT"
    ].sum()
    bg_weights["WEIGHT_SHARE"] = bg_weights["BG_WEIGHT"] / bg_weights.groupby(zone_id)[
        "BG_WEIGHT"
    ].transform("sum").replace(0, np.nan)

    rows: list[dict[str, Any]] = []
    for zone, group in joined.groupby(zone_id):
        row: dict[str, Any] = {zone_id: zone}
        for spec in specs.values():
            columns = [*spec["categories"]]
            if spec.get("remainder"):
                columns.append(spec["remainder"])
            # Require a complete source distribution. Mixing partially published
            # rows would understate missing categories and bias the normalized mix.
            valid = group[columns].notna().all(axis=1) & (group["BG_WEIGHT"] > 0)
            if not valid.any():
                row.update({column: np.nan for column in columns})
                continue
            weights = group.loc[valid, "BG_WEIGHT"]
            values = (
                group.loc[valid, columns].multiply(weights, axis=0).sum()
                / weights.sum()
            )
            values = values / values.sum() if values.sum() else values
            row.update(values.to_dict())
        rows.append(row)
    return pd.DataFrame(rows, columns=[zone_id, *share_columns]), bg_weights


def fill_missing_distributions_from_nearest_taz(
    result: gpd.GeoDataFrame, specs: dict[str, Any]
) -> gpd.GeoDataFrame:
    """Fill an entirely missing dimension from the nearest TAZ with valid shares."""
    filled = result.copy()
    points = filled.geometry.representative_point()
    for spec in specs.values():
        columns = [*spec["categories"]]
        if spec.get("remainder"):
            columns.append(spec["remainder"])
        missing = filled[columns].isna().all(axis=1)
        valid = filled[columns].notna().all(axis=1)
        if not missing.any() or not valid.any():
            continue
        valid_positions = np.flatnonzero(valid.to_numpy())
        for position in np.flatnonzero(missing.to_numpy()):
            # Measure from an interior point of the empty TAZ to each valid TAZ
            # polygon. This yields zero for a valid TAZ that contains that point
            # and otherwise chooses the closest polygon boundary.
            distances = filled.geometry.iloc[valid_positions].distance(
                points.iloc[position]
            )
            nearest_position = valid_positions[int(np.argmin(distances.to_numpy()))]
            filled.loc[filled.index[position], columns] = filled.iloc[nearest_position][
                columns
            ].to_numpy()
    return filled


def format_zone_output(
    frame: pd.DataFrame,
    source_zone_id: str,
    output_zone_id: str,
) -> pd.DataFrame:
    """Rename the zone identifier for output and place it first."""
    if source_zone_id not in frame:
        raise ValueError(f"Output table is missing zone identifier: {source_zone_id}")
    if output_zone_id != source_zone_id and output_zone_id in frame:
        raise ValueError(
            f"Cannot rename {source_zone_id} to existing output column "
            f"{output_zone_id}."
        )
    output = frame.rename(columns={source_zone_id: output_zone_id})
    remaining = [column for column in output if column != output_zone_id]
    return output[[output_zone_id, *remaining]]


def format_population_output(frame: pd.DataFrame) -> pd.DataFrame:
    """Give total population its unambiguous output name."""
    if "HHPOP" not in frame:
        return frame
    if "TOTPOP" in frame:
        raise ValueError("Cannot rename HHPOP to existing output column TOTPOP.")
    return frame.rename(columns={"HHPOP": "TOTPOP"})


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    load_local_env(config["census"].get("env_file"))
    cache_dir = Path(config["paths"]["census_cache_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = state_fips(config["study_area"]["state"])
    counties = [str(value).zfill(3) for value in config["study_area"]["counties"]]
    refresh = bool(config["census"].get("refresh_cache", False))
    zone_id = config["taz"]["id_column"]

    print("Reading TAZ and Census block geography")
    zones = read_vector(config["paths"]["taz_file"])
    zones[zone_id] = zones[zone_id].astype(str)
    zones = zones.loc[
        zones[config["taz"]["county_column"]].astype(str).isin(counties)
    ].copy()
    if zones.crs is None:
        raise ValueError("TAZ input has no coordinate reference system")
    block_path = ensure_block_geography(config["block_geography"])
    blocks = read_vector(block_path)
    source_geoid = config["block_geography"]["geoid_column"]
    blocks = blocks.rename(columns={source_geoid: "GEOID20"})
    blocks["GEOID20"] = blocks["GEOID20"].astype(str).str.zfill(15)
    blocks = blocks.loc[
        blocks[config["block_geography"]["county_column"]].astype(str).isin(counties)
    ].to_crs(zones.crs)
    blocks = blocks.cx[
        zones.total_bounds[0] : zones.total_bounds[2],
        zones.total_bounds[1] : zones.total_bounds[3],
    ].copy()
    if not zones.crs.is_projected:
        # Overlay areas and nearest-TAZ distances must use linear units rather
        # than degrees. A local UTM CRS is sufficient for this borough workflow.
        projected = zones.estimate_utm_crs()
        zones = zones.to_crs(projected)
        blocks = blocks.to_crs(projected)
    blocks["BLOCK_AREA"] = blocks.geometry.area

    dec = config["census"]["decennial"]
    print("Loading 2020 Decennial block counts")
    raw_blocks = fetch_geography(
        cache_dir,
        dec["dataset"],
        int(dec["year"]),
        state,
        counties,
        "block",
        ["NAME", *dec["variables"].values()],
        refresh,
    )
    block_counts = prepare_decennial_blocks(raw_blocks, dec["variables"])
    blocks = blocks.merge(block_counts, on="GEOID20", how="inner")

    acs = config["census"]["acs"]
    specs = acs["block_group_distributions"]
    income_classification, income_allocations = build_income_quantile_allocations(
        config, specs
    )
    tract_specs = {
        name: spec
        for name, spec in specs.items()
        if spec.get("geography", "block group") == "tract"
    }
    print("Loading ACS tract controls")
    raw_tracts = fetch_geography(
        cache_dir,
        acs["dataset"],
        int(acs["year"]),
        state,
        counties,
        "tract",
        [
            "NAME",
            *acs["tract_controls"].values(),
            *distribution_fields(tract_specs),
        ],
        refresh,
    )
    tract_controls = prepare_tract_controls(raw_tracts, acs["tract_controls"])
    distribution_tables: dict[str, pd.DataFrame] = {}
    if tract_specs:
        distribution_tables["tract"] = prepare_geographic_distributions(
            raw_tracts,
            tract_specs,
            "tract",
            income_allocations=income_allocations,
        )
    block_group_specs = {
        name: spec
        for name, spec in specs.items()
        if spec.get("geography", "block group") == "block group"
    }
    if block_group_specs:
        print("Loading ACS block group distributions")
        raw_block_groups = fetch_geography(
            cache_dir,
            acs["dataset"],
            int(acs["year"]),
            state,
            counties,
            "block group",
            ["NAME", *distribution_fields(block_group_specs)],
            refresh,
        )
        distribution_tables["block group"] = prepare_geographic_distributions(
            raw_block_groups,
            block_group_specs,
            "block group",
            income_allocations=income_allocations,
        )

    print("Building block-to-TAZ allocations")
    intersections = intersect_blocks_and_zones(
        blocks,
        zones,
        zone_id,
        float(config["allocation"].get("conflict_area_tolerance", 0)),
    )
    method = str(config["allocation"].get("method", "centroid")).lower()
    if method == "centroid":
        allocations = centroid_allocations(blocks, zones, zone_id)
        allocations["ALLOCATION_METHOD"] = "centroid"
        allocations["RESIDENTIAL_PARCEL_COUNT"] = np.nan
        missing = set(intersections["GEOID20"]) - set(allocations["GEOID20"])
        if missing:
            # An interior point can miss all TAZ polygons because of boundary
            # gaps or topology differences. Assign such a block to its largest
            # positive-area overlap rather than dropping its counts.
            fallback = (
                intersections.loc[intersections["GEOID20"].isin(missing)]
                .sort_values(["GEOID20", "AREA_SHARE"], ascending=[True, False])
                .drop_duplicates("GEOID20")[["GEOID20", zone_id]]
            )
            fallback["SHARE"] = 1.0
            fallback["ALLOCATION_METHOD"] = "centroid_area_fallback"
            fallback["RESIDENTIAL_PARCEL_COUNT"] = np.nan
            allocations = pd.concat([allocations, fallback], ignore_index=True)
    elif method == "area":
        allocations = intersections[["GEOID20", zone_id, "AREA_SHARE"]].rename(
            columns={"AREA_SHARE": "SHARE"}
        )
        allocations["ALLOCATION_METHOD"] = "land_area"
        allocations["RESIDENTIAL_PARCEL_COUNT"] = np.nan
    elif method == "tiered":
        parcel_path = config["paths"].get("parcel_file")
        if not parcel_path:
            raise ValueError("paths.parcel_file is required for tiered allocation")
        residential_column = str(
            config["allocation"].get("parcel_residential_column", "Assessing1")
        )
        residential_values = [
            str(value)
            for value in config["allocation"].get(
                "residential_parcel_values", []
            )
        ]
        print("Reading residential parcel geography")
        parcels = read_vector(parcel_path, columns=[residential_column])
        if parcels.crs is None:
            raise ValueError("Parcel input has no coordinate reference system")
        parcels = parcels.to_crs(zones.crs)
        allocations = tiered_allocations(
            blocks,
            zones,
            parcels,
            intersections,
            zone_id,
            residential_column,
            residential_values,
        )
    else:
        raise ValueError("allocation.method must be 'centroid', 'area', or 'tiered'")
    allocations = apply_manual_shares(
        allocations,
        config["paths"].get("manual_shares_file"),
        zone_id,
        set(blocks["GEOID20"]),
        set(zones[zone_id]),
    )
    allocations = allocations.merge(
        blocks.drop(columns="geometry")[
            [
                "GEOID20",
                "TRACT_GEOID",
                "BLOCK_GROUP_GEOID",
                "HH_2020",
                "POP_2020",
                "BLOCK_AREA",
            ]
        ],
        on="GEOID20",
        how="left",
    )
    allocations["LAND_AREA_ALLOC"] = allocations["BLOCK_AREA"] * allocations["SHARE"]
    allocations, tract_audit = scale_to_tract_controls(
        allocations, tract_controls, zone_id
    )
    allocations, household_override_audit = apply_taz_household_overrides(
        allocations,
        config["allocation"].get("taz_household_overrides"),
        zone_id,
        set(zones[zone_id]),
    )
    distributions, bg_weights = inherit_distributions(
        allocations, distribution_tables, specs, zone_id
    )

    zone_totals = allocations.groupby(zone_id, as_index=False)[
        ["HH_2020_ALLOC", "POP_2020_ALLOC", "TOTHH", "HHPOP"]
    ].sum()
    result = zones.merge(zone_totals, on=zone_id, how="left").merge(
        distributions, on=zone_id, how="left"
    )
    total_columns = ["HH_2020_ALLOC", "POP_2020_ALLOC", "TOTHH", "HHPOP"]
    result[total_columns] = result[total_columns].fillna(0.0)
    if config["allocation"].get("fill_missing_distributions_from_nearest_taz", True):
        result = fill_missing_distributions_from_nearest_taz(result, specs)

    conflicts = intersections.merge(
        blocks[["GEOID20", "HH_2020", "POP_2020"]],
        on="GEOID20",
        how="left",
    ).merge(
        allocations[
            [
                "GEOID20",
                zone_id,
                "SHARE",
                "ALLOCATION_METHOD",
                "RESIDENTIAL_PARCEL_COUNT",
            ]
        ].rename(columns={"SHARE": "ALLOCATED_SHARE"}),
        on=["GEOID20", zone_id],
        how="left",
    )
    conflict_blocks = intersections.groupby("GEOID20")[zone_id].nunique()
    material_threshold = float(
        config["allocation"].get("material_conflict_min_area_share", 0.01)
    )
    material_counts = (
        # Ignore overlaps below the review threshold independently for each TAZ;
        # a block is material only when at least two remaining overlaps qualify.
        intersections.loc[intersections["AREA_SHARE"] >= material_threshold]
        .groupby("GEOID20")[zone_id]
        .nunique()
    )
    material_blocks = set(material_counts.loc[material_counts > 1].index)
    conflicts = conflicts.loc[
        conflicts["GEOID20"].isin(conflict_blocks.loc[conflict_blocks > 1].index)
    ].sort_values(["GEOID20", "AREA_SHARE"], ascending=[True, False])
    conflicts["MATERIAL_CONFLICT"] = conflicts["GEOID20"].isin(material_blocks)

    unallocated_geoids = set(blocks["GEOID20"]) - set(allocations["GEOID20"])
    unallocated = blocks.loc[blocks["GEOID20"].isin(unallocated_geoids)]

    outputs = config["outputs"]
    output_zone_id = str(outputs.get("zone_id_column", zone_id))
    # Keep the source identifier internally for joins, then apply the model-facing
    # identifier and population names only to serialized tables.
    result_output = format_population_output(
        format_zone_output(result, zone_id, output_zone_id)
    )
    allocations_output = format_population_output(
        format_zone_output(allocations, zone_id, output_zone_id)
    )
    conflicts_output = format_zone_output(conflicts, zone_id, output_zone_id)
    bg_weights_output = format_zone_output(bg_weights, zone_id, output_zone_id)
    household_override_audit_output = format_zone_output(
        household_override_audit, zone_id, output_zone_id
    )
    tract_audit_output = format_population_output(tract_audit)
    gpkg_path = output_dir / outputs["taz_geopackage"]
    result_output.to_file(gpkg_path, layer=outputs["taz_layer"], driver="GPKG")
    result_output.drop(columns="geometry").to_csv(
        output_dir / outputs["taz_csv"], index=False
    )
    allocations_output.to_csv(
        output_dir / outputs["block_allocation_csv"], index=False
    )
    conflicts_output.to_csv(
        output_dir / outputs["conflict_audit_csv"], index=False
    )
    tract_audit_output.to_csv(
        output_dir / outputs["tract_audit_csv"], index=False
    )
    bg_weights_output.to_csv(
        output_dir / outputs["block_group_weights_csv"], index=False
    )
    household_override_audit_output.to_csv(
        output_dir / outputs["manual_household_overrides_csv"], index=False
    )
    if income_classification is not None:
        income_classification.quantile_summary.to_csv(
            output_dir / outputs["pums_income_quantiles_csv"], index=False
        )
        income_diagnostics = pd.concat(
            [
                allocation.band_diagnostics.assign(distribution=name)
                for name, allocation in income_allocations.items()
            ],
            ignore_index=True,
        )
        income_diagnostics.to_csv(
            output_dir / outputs["income_band_allocation_csv"], index=False
        )
    summary = {
        "config": config["_config_path"],
        "allocation_method": method,
        "allocation_method_counts": {
            name: int(count)
            for name, count in allocations.groupby("ALLOCATION_METHOD")[
                "GEOID20"
            ].nunique().items()
        },
        "output_zone_id_column": output_zone_id,
        "taz_count": int(len(result)),
        "allocated_block_count": int(allocations["GEOID20"].nunique()),
        "conflict_block_count": int((conflict_blocks > 1).sum()),
        "material_conflict_block_count": int(len(material_blocks)),
        "material_conflict_min_area_share": material_threshold,
        "manual_household_override_count": int(len(household_override_audit)),
        "unallocated_block_count": int(len(unallocated_geoids)),
        "unallocated_households_2020": float(unallocated["HH_2020"].sum()),
        "unallocated_population_2020": float(unallocated["POP_2020"].sum()),
        "households_2020_allocated": float(allocations["HH_2020_ALLOC"].sum()),
        "households_acs_scaled": float(result["TOTHH"].sum()),
        "population_2020_allocated": float(allocations["POP_2020_ALLOC"].sum()),
        "population_acs_scaled": float(result["HHPOP"].sum()),
        "outputs": {"geopackage": str(gpkg_path), "directory": str(output_dir)},
    }
    if income_classification is not None:
        summary["income_quantile_breakpoints"] = list(
            income_classification.breakpoints
        )
    (output_dir / outputs["run_summary_json"]).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default=SCRIPT_DIR / "household_allocation_config.yaml",
        help="Path to the household allocation YAML file",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
