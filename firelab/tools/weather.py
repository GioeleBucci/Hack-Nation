"""burn_window_days: how many days per year a burn prescription is historically met (Open-Meteo, ERA5)."""

from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np

from firelab.tools._common import ToolError, get_json, tool

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# The synthetic landscape is a stand-in for Sierra Nevada foothill terrain, so it uses that weather.
REGIONS = {
    "sierra_foothills_ca": {"lat": 38.90, "lon": -120.90, "description": "Sierra Nevada foothills, California"},
    "synthetic_v1": {"lat": 38.90, "lon": -120.90, "description": "Synthetic landscape (Sierra foothills weather)"},
    "catalonia_es": {"lat": 41.80, "lon": 2.30, "description": "Central Catalonia, Spain"},
    "victoria_au": {"lat": -37.50, "lon": 145.50, "description": "Central highlands, Victoria, Australia"},
}

DEFAULT_PRESCRIPTION = {
    "rh_min": 20, "rh_max": 50,            # relative humidity, %
    "wind_min_kmh": 5, "wind_max_kmh": 25,  # 10 m wind speed
    "temp_min_c": 5, "temp_max_c": 30,
    "hours": [10, 17],                      # local daytime window [start, end)
    "min_hours": 4,                         # hours in the window that must meet the prescription
    "months": None,                         # e.g. [3, 4, 5, 10, 11]; None = all months
}


def _region(region_id: str) -> dict:
    if region_id in REGIONS:
        return {"id": region_id, **REGIONS[region_id]}
    try:
        lat, lon = (float(v) for v in region_id.split(","))
    except ValueError:
        raise ToolError(f"Unknown region '{region_id}'. Use one of {list(REGIONS)} or 'lat,lon'.") from None
    return {"id": region_id, "lat": lat, "lon": lon, "description": "custom coordinates"}


@tool
def burn_window_days(region_id: str, prescription: dict | None = None, years: list[int] | None = None) -> dict:
    """Count historical days per year meeting a prescribed-burn weather window."""
    rx = {**DEFAULT_PRESCRIPTION, **(prescription or {})}
    unknown = set(rx) - set(DEFAULT_PRESCRIPTION)
    if unknown:
        raise ToolError(f"Unknown prescription keys {sorted(unknown)}; allowed: {sorted(DEFAULT_PRESCRIPTION)}")
    region = _region(region_id)
    this_year = date.today().year
    years = sorted(int(y) for y in (years or range(this_year - 5, this_year)))
    if not years or years[-1] >= this_year:
        raise ToolError("years must be complete past years")

    data = get_json(ARCHIVE, {
        "latitude": region["lat"], "longitude": region["lon"],
        "start_date": f"{years[0]}-01-01", "end_date": f"{years[-1]}-12-31",
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "wind_speed_unit": "kmh", "timezone": "auto",
    })
    hourly = data["hourly"]
    times = hourly["time"]
    def arr(key):
        return np.array([np.nan if v is None else v for v in hourly[key]], dtype=float)
    temp, rh, wind = arr("temperature_2m"), arr("relative_humidity_2m"), arr("wind_speed_10m")
    with np.errstate(invalid="ignore"):
        ok = ((rh >= rx["rh_min"]) & (rh <= rx["rh_max"]) & (wind >= rx["wind_min_kmh"])
              & (wind <= rx["wind_max_kmh"]) & (temp >= rx["temp_min_c"]) & (temp <= rx["temp_max_c"]))

    good_hours: dict[str, int] = defaultdict(int)
    h0, h1 = rx["hours"]
    for t, good in zip(times, ok):
        day, hour = t[:10], int(t[11:13])
        if h0 <= hour < h1 and good:
            good_hours[day] += 1

    per_year = {y: 0 for y in years}
    per_month = defaultdict(int)
    for day, n in good_hours.items():
        y, m = int(day[:4]), int(day[5:7])
        if n >= rx["min_hours"] and y in per_year and (not rx["months"] or m in rx["months"]):
            per_year[y] += 1
            per_month[m] += 1
    counts = np.array(list(per_year.values()), dtype=float)
    return {
        "region": region,
        "prescription": rx,
        "source": "Open-Meteo historical archive (ERA5 reanalysis)",
        "days_per_year": per_year,
        "mean_days_per_year": round(float(counts.mean()), 1),
        "sd_days_per_year": round(float(counts.std(ddof=1)), 1) if counts.size > 1 else 0.0,
        "mean_days_by_month": {m: round(per_month[m] / len(years), 1) for m in range(1, 13)},
    }
