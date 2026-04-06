"""
Synthetic geospatial data generation.

Generates lat/lon coordinates for patients and care sites,
computes distance-based features, assigns urban/suburban/rural classification,
and establishes nearest-site relationships.
"""
import math
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from src.config.settings import ProjectConfig


# ---------------------------------------------------------------------------
# Regional centers for realistic clustering (synthetic US cities)
# ---------------------------------------------------------------------------
REGION_CENTERS = [
    ("Northeast", 40.71, -74.01),
    ("Southeast", 33.75, -84.39),
    ("Midwest", 41.88, -87.63),
    ("Southwest", 33.45, -112.07),
    ("West", 34.05, -118.24),
    ("Northwest", 47.61, -122.33),
    ("Central", 39.10, -94.58),
    ("MidAtlantic", 38.91, -77.04),
]


def _generate_clustered_coords(n: int, seed: int) -> list[tuple[float, float, str, str]]:
    """
    Generate (lat, lon, region, urban_rural) tuples clustered around regional centers.

    Each point is offset from a random center by Gaussian noise, with
    urban/suburban/rural classification based on distance from center.
    """
    rng = np.random.RandomState(seed)
    records = []
    for i in range(n):
        idx = rng.randint(0, len(REGION_CENTERS))
        region, center_lat, center_lon = REGION_CENTERS[idx]
        # Gaussian offset: ~0.5 degrees ≈ 35 miles std dev
        lat = center_lat + rng.normal(0, 0.5)
        lon = center_lon + rng.normal(0, 0.5)
        dist_deg = math.sqrt((lat - center_lat) ** 2 + (lon - center_lon) ** 2)
        if dist_deg < 0.2:
            urban_rural = "Urban"
        elif dist_deg < 0.5:
            urban_rural = "Suburban"
        else:
            urban_rural = "Rural"
        records.append((lat, lon, region, urban_rural))
    return records


def generate_patient_locations(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Generate lat/lon + region + urban/rural for each patient."""
    coords = _generate_clustered_coords(cfg.synthetic_scale, cfg.random_seed)
    rows = [(i + 1, c[0], c[1], c[2], c[3]) for i, c in enumerate(coords)]
    return spark.createDataFrame(rows, [
        "person_id", "patient_lat", "patient_lon", "patient_region", "patient_urban_rural"
    ])


def generate_care_site_locations(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Generate lat/lon + region for each care site."""
    n_sites = max(20, cfg.synthetic_scale // 10)
    coords = _generate_clustered_coords(n_sites, cfg.random_seed + 1000)
    rows = [(i + 1, c[0], c[1], c[2], c[3]) for i, c in enumerate(coords)]
    return spark.createDataFrame(rows, [
        "care_site_id", "site_lat", "site_lon", "site_region", "site_urban_rural"
    ])


def haversine_udf():
    """Return a PySpark UDF that computes haversine distance in miles."""
    @F.udf(FloatType())
    def _haversine(lat1, lon1, lat2, lon2):
        if any(v is None for v in [lat1, lon1, lat2, lon2]):
            return None
        R = 3959.0  # Earth radius in miles
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return float(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))
    return _haversine


def compute_patient_site_distances(
    patient_locs: DataFrame,
    site_locs: DataFrame,
) -> DataFrame:
    """
    Cross-join patients × care sites and compute haversine distance.

    Returns DataFrame with: person_id, care_site_id, distance_miles, distance_band.
    """
    hav = haversine_udf()
    crossed = patient_locs.crossJoin(site_locs)
    with_dist = crossed.withColumn(
        "distance_miles",
        hav(F.col("patient_lat"), F.col("patient_lon"),
            F.col("site_lat"), F.col("site_lon"))
    )
    with_band = with_dist.withColumn(
        "distance_band",
        F.when(F.col("distance_miles") < 5, "< 5 mi")
        .when(F.col("distance_miles") < 15, "5-15 mi")
        .when(F.col("distance_miles") < 30, "15-30 mi")
        .when(F.col("distance_miles") < 60, "30-60 mi")
        .otherwise("> 60 mi")
    )
    return with_band.select(
        "person_id", "care_site_id",
        "distance_miles", "distance_band",
        "patient_region", "site_region",
        "patient_urban_rural", "site_urban_rural",
    )


def compute_nearest_sites(patient_site_distances: DataFrame, top_k: int = 5) -> DataFrame:
    """
    For each patient, find the top-k nearest care sites.

    Returns DataFrame with: person_id, care_site_id, distance_miles, rank.
    """
    from pyspark.sql.window import Window

    w = Window.partitionBy("person_id").orderBy("distance_miles")
    ranked = patient_site_distances.withColumn("site_rank", F.row_number().over(w))
    return ranked.filter(F.col("site_rank") <= top_k).select(
        "person_id", "care_site_id", "distance_miles",
        "distance_band", "site_rank",
        "patient_region", "site_region",
    )
