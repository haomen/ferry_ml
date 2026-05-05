#!/usr/bin/env python3
"""
ETA predictor — Hoboken 14 St ↔ NYC W39 NYWaterway ferry route.

Commands:
    python3 eta_hoboken_nyc.py train    # train/retrain model on all historical data
    python3 eta_hoboken_nyc.py publish  # predict active ferries + push to GitHub Pages
    python3 eta_hoboken_nyc.py live     # real-time terminal display

Cron (weekdays):
    0 6     * * 1-5  python3 /home/hao/Desktop/ais/eta_hoboken_nyc.py train >> /home/hao/Desktop/ais/eta.log 2>&1
    * 7-10  * * 1-5  /bin/bash -c 'P=/home/hao/Desktop/ais; python3 $P/eta_hoboken_nyc.py publish >> $P/eta.log 2>&1; sleep 30; python3 $P/eta_hoboken_nyc.py publish >> $P/eta.log 2>&1'
    * 16-19 * * 1-5  /bin/bash -c 'P=/home/hao/Desktop/ais; python3 $P/eta_hoboken_nyc.py publish >> $P/eta.log 2>&1; sleep 30; python3 $P/eta_hoboken_nyc.py publish >> $P/eta.log 2>&1'
"""

import json
import logging
import math
import pickle
import subprocess
import sys
import time
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2

from terminals import (
    Terminal, HOBOKEN_14, NYC_W39, TERMINALS, BY_NAME, ROUTE_MMSI,
    haversine_nm, bearing, at_terminal,
)

# ── Config ────────────────────────────────────────────────────────────────────
DB_DSN      = "host=localhost port=5432 dbname=aisdb user=ais password=aispass"
REPO_DIR    = Path(__file__).parent
MODEL_PATH  = REPO_DIR / "model" / "eta_model.pkl"
DOCS_DIR    = REPO_DIR / "docs"
DATA_JSON   = DOCS_DIR / "data.json"

MORNING_HOURS   = range(7, 11)    # 7–10 AM  → Hoboken → NYC
AFTERNOON_HOURS = range(16, 20)   # 4–7 PM   → NYC → Hoboken

MIN_TRIP_SEC  = 240    # filter out docking glitches < 4 min
MAX_TRIP_SEC  = 1200   # filter out very long / interrupted trips > 20 min

# Online adaptation: EWA weight applied to the most-recent completed trip.
# 0.4 means recent trips contribute ~40 %, prior EWA contributes 60 %.
BIAS_ALPHA    = 0.4

ACTIVE_WINDOW = 600    # seconds — vessel considered "active" if seen within this window

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eta")

# direction encoding: 0 = Hoboken → NYC, 1 = NYC → Hoboken
DIR_HOB_NYC = 0
DIR_NYC_HOB = 1

FEATURE_COLS = [
    "dist_remaining_nm",  # dominant predictor — how far to destination
    "sog",                # current speed over ground
    "cog_alignment",      # cos(angle between course and bearing-to-dest): 1=straight, -1=away
    "elapsed_sec",        # time since departure from origin terminal
    "hour_sin",           # time-of-day (captures tidal / traffic variation)
    "hour_cos",
    "dow_sin",            # day-of-week
    "dow_cos",
    "direction",          # 0 or 1
    "route_median_sec",   # historical median crossing time for this direction
]


# ── DB helpers ─────────────────────────────────────────────────────────────────
def _conn():
    return psycopg2.connect(DB_DSN)


def _last_n_weekdays(n: int) -> list[date]:
    """Return the n most-recent weekday dates before today."""
    result, d = [], date.today() - timedelta(days=1)
    while len(result) < n:
        if d.weekday() < 5:   # Mon=0 … Fri=4
            result.append(d)
        d -= timedelta(days=1)
    return result


def load_positions(today_only: bool = False,
                   training_weekdays: int = 0) -> pd.DataFrame:
    """
    Load route-vessel positions from PostgreSQL.
    today_only=True  → last 16 hours (for online bias computation)
    training_weekdays=N → restrict to the last N weekday dates (for training)
    """
    mmsi_list = list(ROUTE_MMSI.keys())

    if today_only:
        time_filter = "AND p.ts >= NOW() - INTERVAL '16 hours'"
        params: dict = {"mmsi": mmsi_list}
    elif training_weekdays > 0:
        days    = _last_n_weekdays(training_weekdays)
        day_min = datetime.combine(min(days), datetime.min.time()) \
                          .replace(tzinfo=timezone.utc)
        day_max = datetime.combine(max(days) + timedelta(days=1),
                                   datetime.min.time()) \
                          .replace(tzinfo=timezone.utc)
        time_filter = "AND p.ts >= %(day_min)s AND p.ts < %(day_max)s"
        params = {"mmsi": mmsi_list, "day_min": day_min, "day_max": day_max}
    else:
        time_filter = ""
        params = {"mmsi": mmsi_list}

    with _conn() as conn:
        df = pd.read_sql(f"""
            SELECT p.mmsi,
                   p.lat, p.lon,
                   CASE WHEN p.sog > 40 THEN NULL ELSE p.sog END AS sog,
                   p.cog,
                   EXTRACT(EPOCH FROM p.ts)::double precision AS ts_epoch
            FROM positions p
            WHERE p.mmsi = ANY(%(mmsi)s)
              AND p.lat  IS NOT NULL
              {time_filter}
            ORDER BY p.mmsi, p.ts
        """, conn, params=params)
    return df


def load_active_positions() -> pd.DataFrame:
    """Latest position per route vessel seen within ACTIVE_WINDOW seconds."""
    mmsi_list = list(ROUTE_MMSI.keys())
    cutoff = datetime.now(tz=timezone.utc) - timedelta(seconds=ACTIVE_WINDOW)
    with _conn() as conn:
        df = pd.read_sql("""
            SELECT DISTINCT ON (p.mmsi)
                p.mmsi, v.name, p.lat, p.lon, p.sog, p.cog,
                EXTRACT(EPOCH FROM p.ts)::double precision AS ts_epoch
            FROM positions p
            JOIN vessels v ON p.mmsi = v.mmsi
            WHERE p.mmsi = ANY(%(mmsi)s)
              AND p.ts >= %(cutoff)s
              AND p.lat IS NOT NULL
            ORDER BY p.mmsi, p.ts DESC
        """, conn, params={"mmsi": mmsi_list, "cutoff": cutoff})
    return df


# ── Trip detection ─────────────────────────────────────────────────────────────
def detect_trips(pos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replay the position stream through a per-vessel state machine.
    Records completed Hoboken 14 St ↔ NYC W39 crossings only.
    Processes each vessel in isolation so inter-vessel gaps don't pollute state.
    """
    trips: list[dict] = []

    for mmsi, grp in pos_df.groupby("mmsi"):
        grp = grp.sort_values("ts_epoch")
        state = dict(terminal=None, depart_ts=None, depart_lat=None, depart_lon=None)

        for _, row in grp.iterrows():
            lat  = float(row["lat"])
            lon  = float(row["lon"])
            sog  = row.get("sog")
            ts   = float(row["ts_epoch"])
            term = at_terminal(lat, lon, sog)

            if term:
                if (state["depart_ts"] is not None
                        and state["terminal"]
                        and state["terminal"] != term.name):
                    duration = ts - state["depart_ts"]
                    if MIN_TRIP_SEC <= duration <= MAX_TRIP_SEC:
                        dist = haversine_nm(state["depart_lat"], state["depart_lon"],
                                            lat, lon)
                        direction = (DIR_HOB_NYC if state["terminal"] == HOBOKEN_14.name
                                     else DIR_NYC_HOB)
                        trips.append({
                            "mmsi":         mmsi,
                            "vessel":       ROUTE_MMSI.get(mmsi, mmsi),
                            "origin":       state["terminal"],
                            "destination":  term.name,
                            "depart_ts":    state["depart_ts"],
                            "arrive_ts":    ts,
                            "duration_sec": int(duration),
                            "distance_nm":  round(dist, 3),
                            "direction":    direction,
                        })
                state["terminal"]   = term.name
                state["depart_ts"]  = None
                state["depart_lat"] = None
                state["depart_lon"] = None
            else:
                if state["terminal"] is not None and state["depart_ts"] is None:
                    state["depart_ts"]  = ts
                    state["depart_lat"] = lat
                    state["depart_lon"] = lon

    if not trips:
        return pd.DataFrame(columns=[
            "mmsi", "vessel", "origin", "destination",
            "depart_ts", "arrive_ts", "duration_sec", "distance_nm", "direction",
        ])
    return pd.DataFrame(trips)


# ── Feature engineering ────────────────────────────────────────────────────────
def _time_features(ts_epoch: float) -> dict:
    dt  = datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
    h   = dt.hour
    dow = dt.weekday()
    return {
        "hour_sin": math.sin(2 * math.pi * h   / 24),
        "hour_cos": math.cos(2 * math.pi * h   / 24),
        "dow_sin":  math.sin(2 * math.pi * dow / 7),
        "dow_cos":  math.cos(2 * math.pi * dow / 7),
    }


def build_features(trips_df: pd.DataFrame, pos_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each trip, sample at 25 %, 50 %, and 75 % of elapsed trip time.
    Sampling at multiple points triples the training set and teaches the model
    to predict accurately at any stage of a crossing (not just from the start).
    """
    if trips_df.empty:
        return pd.DataFrame()

    route_median = trips_df.groupby("direction")["duration_sec"].median()

    # Index positions by (mmsi, ts_epoch) for fast slice lookups
    pos_indexed = (pos_df
                   .set_index(["mmsi", "ts_epoch"])
                   .sort_index()[["lat", "lon", "sog", "cog"]])

    rows: list[dict] = []
    for _, trip in trips_df.iterrows():
        dest      = BY_NAME[trip["destination"]]
        direction = int(trip["direction"])
        mmsi      = trip["mmsi"]
        d_ts, a_ts = trip["depart_ts"], trip["arrive_ts"]

        try:
            grp = pos_indexed.loc[mmsi]
        except KeyError:
            continue
        mask     = (grp.index >= d_ts) & (grp.index <= a_ts)
        trip_pos = grp[mask].reset_index()   # columns: ts_epoch, lat, lon, sog, cog

        if len(trip_pos) < 4:
            continue

        for frac in (0.25, 0.50, 0.75):
            idx = max(1, int(len(trip_pos) * frac))
            s   = trip_pos.iloc[idx]

            lat, lon = float(s["lat"]), float(s["lon"])
            sog  = float(s["sog"] or 0)
            cog  = float(s["cog"] or 0)
            ts   = float(s["ts_epoch"])

            dist_rem  = haversine_nm(lat, lon, dest.lat, dest.lon)
            brg       = bearing(lat, lon, dest.lat, dest.lon)
            alignment = math.cos(math.radians(cog - brg))
            elapsed   = ts - d_ts
            remaining = a_ts - ts

            rows.append({
                "dist_remaining_nm": dist_rem,
                "sog":               sog,
                "cog_alignment":     alignment,
                "elapsed_sec":       elapsed,
                **_time_features(ts),
                "direction":         direction,
                "route_median_sec":  route_median.get(direction, 600.0),
                "remaining_sec":     remaining,
            })

    return pd.DataFrame(rows)


# ── Training ───────────────────────────────────────────────────────────────────
def train():
    """
    Train three HistGradientBoostingRegressor models (p10 / p50 / p90)
    for quantile ETA prediction.  HistGBR is typically 10–100× faster
    than GradientBoostingRegressor on tabular data.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    days = _last_n_weekdays(3)
    log.info("Training on last 3 weekdays: %s", [str(d) for d in sorted(days)])
    pos_df = load_positions(training_weekdays=3)
    log.info("  %d positions, %d vessels", len(pos_df), pos_df["mmsi"].nunique())

    log.info("Detecting trips...")
    trips_df = detect_trips(pos_df)
    log.info("  %d complete crossings", len(trips_df))

    if len(trips_df) < 10:
        log.error("Need at least 10 trips to train. Collect more data first.")
        sys.exit(1)

    log.info("\nCrossing stats by direction:")
    stats = trips_df.groupby(["origin", "destination"])["duration_sec"].agg(
        count="count", median="median", std="std"
    )
    stats["median_min"] = (stats["median"] / 60).round(1)
    stats["std_min"]    = (stats["std"]    / 60).round(1)
    log.info(stats[["count", "median_min", "std_min"]].to_string())

    log.info("\nBuilding training features (25/50/75 %% samples per trip)...")
    feat_df = build_features(trips_df, pos_df)
    log.info("  %d training rows from %d trips", len(feat_df), len(trips_df))

    if len(feat_df) < 10:
        log.error("Not enough feature rows — are positions sparse during trips?")
        sys.exit(1)

    X = feat_df[FEATURE_COLS]
    y = feat_df["remaining_sec"]

    models: dict = {}
    for quantile, key in [(0.10, "p10"), (0.50, "p50"), (0.90, "p90")]:
        m = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=500,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=5,
            random_state=42,
        )
        m.fit(X, y)
        models[key] = m
        if key == "p50":
            n_cv  = min(5, max(2, len(X) // 20))
            maes  = -cross_val_score(m, X, y, cv=n_cv,
                                     scoring="neg_mean_absolute_error")
            log.info("  p50 CV MAE: %.1f min ± %.1f min",
                     maes.mean() / 60, maes.std() / 60)

    route_medians = trips_df.groupby("direction")["duration_sec"].median().to_dict()

    artifact = {
        "models":        models,
        "route_medians": route_medians,
        "trained_at":    datetime.now(tz=timezone.utc).isoformat(),
        "n_trips":       len(trips_df),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    log.info("\nModel saved → %s  (trained on %d crossings)", MODEL_PATH, len(trips_df))


# ── Online adaptation ──────────────────────────────────────────────────────────
def compute_today_bias(artifact: dict) -> dict[int, float]:
    """
    Compute a per-direction EWA bias from today's completed crossings.
    Returns {direction: bias_seconds}; positive = running late.

    Applied on top of the historical model at prediction time so the model
    quickly adapts to weather delays, terminal congestion, etc.
    """
    pos_today = load_positions(today_only=True)
    if pos_today.empty:
        return {}

    trips_today = detect_trips(pos_today)
    if trips_today.empty:
        return {}

    models        = artifact["models"]
    route_medians = artifact["route_medians"]
    bias: dict[int, float] = {}

    for direction, grp in trips_today.groupby("direction"):
        direction = int(direction)
        grp       = grp.sort_values("depart_ts")
        ewa: Optional[float] = None

        for _, trip in grp.iterrows():
            dest  = BY_NAME[trip["destination"]]
            mmsi  = trip["mmsi"]
            d_ts  = trip["depart_ts"]
            a_ts  = trip["arrive_ts"]

            # Pick the position at ~25 % through the trip for the prediction
            trip_pos = pos_today[
                (pos_today["mmsi"] == mmsi)
                & (pos_today["ts_epoch"] >= d_ts)
                & (pos_today["ts_epoch"] <= a_ts)
            ].sort_values("ts_epoch")

            if len(trip_pos) < 3:
                residual = 0.0
            else:
                s   = trip_pos.iloc[max(1, len(trip_pos) // 4)]
                lat = float(s["lat"])
                lon = float(s["lon"])
                sog = float(s["sog"] or 0)
                cog = float(s["cog"] or 0)
                ts  = float(s["ts_epoch"])

                dist_rem  = haversine_nm(lat, lon, dest.lat, dest.lon)
                brg       = bearing(lat, lon, dest.lat, dest.lon)
                alignment = math.cos(math.radians(cog - brg))

                feat = pd.DataFrame([{
                    "dist_remaining_nm": dist_rem,
                    "sog":               sog,
                    "cog_alignment":     alignment,
                    "elapsed_sec":       ts - d_ts,
                    **_time_features(ts),
                    "direction":         direction,
                    "route_median_sec":  route_medians.get(direction, 600.0),
                }])

                pred_rem   = max(0.0, float(models["p50"].predict(feat[FEATURE_COLS])[0]))
                actual_rem = a_ts - ts
                residual   = actual_rem - pred_rem

            ewa = (BIAS_ALPHA * residual + (1 - BIAS_ALPHA) * ewa
                   if ewa is not None else residual)

        if ewa is not None:
            bias[direction] = ewa

    log.info("Today bias: %s", {k: f"{v:+.0f}s" for k, v in bias.items()})
    return bias


# ── Prediction ─────────────────────────────────────────────────────────────────
def load_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model at {MODEL_PATH}. Run: python3 {__file__} train"
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_vessel(
    row: pd.Series,
    dest: Terminal,
    direction: int,
    artifact: dict,
    today_bias: dict,
) -> Optional[dict]:
    models        = artifact["models"]
    route_medians = artifact["route_medians"]

    lat = float(row["lat"])
    lon = float(row["lon"])
    sog = float(row.get("sog") or 0)
    cog = float(row.get("cog") or 0)
    ts  = float(row["ts_epoch"])

    dist_rem  = haversine_nm(lat, lon, dest.lat, dest.lon)
    brg       = bearing(lat, lon, dest.lat, dest.lon)
    alignment = math.cos(math.radians(cog - brg))

    feat = pd.DataFrame([{
        "dist_remaining_nm": dist_rem,
        "sog":               sog,
        "cog_alignment":     alignment,
        "elapsed_sec":       300.0,   # mid-trip estimate for live inference
        **_time_features(ts),
        "direction":         direction,
        "route_median_sec":  route_medians.get(direction, 600.0),
    }])

    p10 = max(0.0, float(models["p10"].predict(feat[FEATURE_COLS])[0]))
    p50 = max(0.0, float(models["p50"].predict(feat[FEATURE_COLS])[0]))
    p90 = max(0.0, float(models["p90"].predict(feat[FEATURE_COLS])[0]))

    # Apply today's bias: widen confidence interval proportionally to bias
    bias    = today_bias.get(direction, 0.0)
    p50_adj = max(0.0, p50 + bias)
    p10_adj = max(0.0, p10 + bias)
    p90_adj = max(0.0, p90 + bias)
    conf_pm = max(0.0, (p90_adj - p10_adj) / 2)

    now     = time.time()
    eta_ts  = now + p50_adj
    eta_dt  = datetime.fromtimestamp(eta_ts).strftime("%I:%M %p")
    eta_iso = datetime.fromtimestamp(eta_ts).astimezone().isoformat()
    name    = (str(row.get("name") or "") or ROUTE_MMSI.get(str(row["mmsi"]), str(row["mmsi"])))

    return {
        "vessel":        name,
        "mmsi":          str(row["mmsi"]),
        "destination":   dest.name,
        "eta":           eta_dt,
        "eta_iso":       eta_iso,
        "eta_ts":        round(eta_ts),
        "conf_pm_min":   round(conf_pm / 60, 1),
        "remaining_min": round(p50_adj / 60, 1),
        "dist_nm":       round(dist_rem, 2),
        "sog":           round(sog, 1),
        "delay_sec":     round(bias),
    }


def _current_period() -> Optional[tuple]:
    """Returns (direction, destination Terminal) for the current hour, else None."""
    h = datetime.now().hour
    if h in MORNING_HOURS:
        return DIR_HOB_NYC, NYC_W39
    if h in AFTERNOON_HOURS:
        return DIR_NYC_HOB, HOBOKEN_14
    return None


# ── Publish ────────────────────────────────────────────────────────────────────
def publish():
    artifact    = load_artifact()
    period_info = _current_period()
    today_bias  = compute_today_bias(artifact)

    active = load_active_positions()
    log.info("%d active route vessels in last %ds", len(active), ACTIVE_WINDOW)

    results: list[dict] = []
    for _, row in active.iterrows():
        lat  = float(row["lat"])
        lon  = float(row["lon"])
        sog  = float(row.get("sog") or 0)

        if at_terminal(lat, lon, sog):
            continue   # docked — no ETA to publish

        if period_info is not None:
            direction, dest = period_info
        else:
            # Outside standard hours: guess destination from course alignment
            best, best_align = None, -2.0
            cog = float(row.get("cog") or 0)
            for t in TERMINALS:
                brg   = bearing(lat, lon, t.lat, t.lon)
                align = math.cos(math.radians(cog - brg))
                if align > best_align:
                    best, best_align = t, align
            if best is None or best_align < 0.3:
                continue
            dest      = best
            direction = DIR_HOB_NYC if dest == NYC_W39 else DIR_NYC_HOB

        r = predict_vessel(row, dest, direction, artifact, today_bias)
        if r:
            results.append(r)

    results.sort(key=lambda r: r["eta_ts"])

    if period_info:
        direction_now = period_info[0]
        period_label  = "morning" if direction_now == DIR_HOB_NYC else "afternoon"
    else:
        direction_now = -1
        period_label  = "off-hours"

    payload = {
        "generated_at":   datetime.now().astimezone().isoformat(),
        "period":         period_label,
        "today_bias_sec": round(today_bias.get(direction_now, 0.0)),
        "n_trips":        artifact.get("n_trips", 0),
        "trained_at":     artifact.get("trained_at", ""),
        "ferries":        results,
    }

    DOCS_DIR.mkdir(exist_ok=True)
    DATA_JSON.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s (%d ferries)", DATA_JSON, len(results))

    _git_push()


def _git_push():
    git = ["git", "-C", str(REPO_DIR)]
    subprocess.run([*git, "add", str(DATA_JSON)], check=True)
    status = subprocess.run(
        [*git, "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not status:
        log.info("No changes to push (data.json unchanged)")
        return
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    subprocess.run([*git, "commit", "-m", f"Ferry ETAs {now_str}"], check=True)
    subprocess.run([*git, "push", "origin", "master"], check=True)
    log.info("Pushed to GitHub Pages")


# ── Live monitor ───────────────────────────────────────────────────────────────
def live_monitor(interval: int = 30):
    print(f"Live ETA monitor — Hoboken 14 St ↔ NYC W39  (refresh {interval}s)\n")
    while True:
        try:
            artifact    = load_artifact()
            today_bias  = compute_today_bias(artifact)
            active      = load_active_positions()
            period_info = _current_period()

            print(f"\n── {datetime.now():%H:%M:%S} ── {len(active)} active vessels ──")
            for _, row in active.iterrows():
                lat  = float(row["lat"])
                lon  = float(row["lon"])
                sog  = float(row.get("sog") or 0)
                name = str(row.get("name") or "") or ROUTE_MMSI.get(str(row["mmsi"]), str(row["mmsi"]))

                term = at_terminal(lat, lon, sog)
                if term:
                    print(f"  {name:<24}  AT {term.name}")
                    continue

                if period_info is None:
                    print(f"  {name:<24}  SOG {sog:.1f} kt (outside ferry hours)")
                    continue

                direction, dest = period_info
                r = predict_vessel(row, dest, direction, artifact, today_bias)
                if r:
                    bias_str = (f"+{r['delay_sec']//60:.0f} min late"
                                if r["delay_sec"] > 90
                                else f"{abs(r['delay_sec'])//60:.0f} min early"
                                if r["delay_sec"] < -90
                                else "on time")
                    print(f"  {name:<24}  → {dest.name:<16}  "
                          f"ETA {r['eta']} ±{r['conf_pm_min']:.0f} min  "
                          f"({r['remaining_min']:.1f} min, {r['dist_nm']:.2f} nm)  "
                          f"[{bias_str}]")

            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description="NYWaterway ETA predictor")
    ap.add_argument("mode", choices=["train", "publish", "live"])
    ap.add_argument("--interval", type=int, default=30,
                    help="Refresh interval for live mode (seconds)")
    args = ap.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "publish":
        publish()
    elif args.mode == "live":
        live_monitor(args.interval)


if __name__ == "__main__":
    main()
