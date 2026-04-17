#!/usr/bin/env python3
"""
upload_to_hf.py — Export today's AIS data from TimescaleDB and upload
to a HuggingFace dataset repository as a dated Parquet file.

Setup (one-time):
    pip3 install huggingface_hub pandas pyarrow psycopg2-binary
    huggingface-cli login          # saves token to ~/.cache/huggingface/token
    # OR set env var: export HF_TOKEN=hf_...

    Create the dataset repo at huggingface.co/new-dataset first, then
    set HF_REPO below to "your-username/your-dataset-name".

Usage:
    python3 upload_to_hf.py              # uploads today's data
    python3 upload_to_hf.py 2026-04-15  # upload a specific date
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
import psycopg2
import shutil
import subprocess
from huggingface_hub import HfFolder

# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO   = "menhao/ais-hudson-river"   # <-- set this
HF_TOKEN  = os.environ.get("HF_TOKEN") or HfFolder.get_token()
HF_CLONE  = os.path.expanduser("~/ais-data/hf-repo")

DB_DSN    = "host=localhost port=5432 dbname=aisdb user=ais password=aispass"
LOCAL_DIR = os.path.expanduser("~/ais-data/exports")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("hf-upload")


# ── Export ────────────────────────────────────────────────────────────────────
def export_day(date_str: str) -> str:
    """
    Query all positions for the given date (YYYY-MM-DD, local time),
    join with vessel metadata, and save as Parquet.
    Returns the local file path.
    """
    # Build UTC time range covering the full local calendar day
    local_date = datetime.strptime(date_str, "%Y-%m-%d")
    # America/New_York offset (EDT=-4, EST=-5). Use fixed offset from system.
    # Since system TZ is America/New_York we use local midnight → next midnight.
    day_start = datetime(local_date.year, local_date.month, local_date.day,
                         0, 0, 0).astimezone(timezone.utc)
    day_end   = day_start + timedelta(days=1)

    conn = psycopg2.connect(DB_DSN)

    query = """
        SELECT
            p.ts                        AS timestamp,
            p.mmsi,
            v.name                      AS vessel_name,
            v.vessel_type,
            v.length                    AS vessel_length_m,
            v.beam                      AS vessel_beam_m,
            v.is_nywaterway,
            p.lat,
            p.lon,
            p.sog,
            p.cog,
            p.heading,
            p.nav_status,
            p.msg_type
        FROM positions p
        LEFT JOIN vessels v ON p.mmsi = v.mmsi
        WHERE p.ts >= %s AND p.ts < %s
        ORDER BY p.ts ASC
    """

    log.info(f"Querying positions for {date_str} ({day_start} → {day_end} UTC)")
    df = pd.read_sql(query, conn, params=(day_start, day_end))
    conn.close()

    if df.empty:
        log.warning(f"No data found for {date_str}")
        return None

    log.info(f"  {len(df):,} rows, {df['mmsi'].nunique()} vessels")

    # Ensure timestamp is UTC-aware
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    os.makedirs(LOCAL_DIR, exist_ok=True)
    out_path = os.path.join(LOCAL_DIR, f"{date_str}.csv")
    df.to_csv(out_path, index=False)
    log.info(f"  Saved to {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")

    return out_path


# ── Upload via git ────────────────────────────────────────────────────────────
def run(cmd, **kwargs):
    log.info(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def upload_to_hf(local_path: str, date_str: str):
    if not HF_TOKEN:
        raise RuntimeError("No HuggingFace token — run: huggingface-cli login")

    repo_url = f"https://user:{HF_TOKEN}@huggingface.co/datasets/{HF_REPO}"

    # Clone once, reuse on subsequent runs
    if not os.path.exists(os.path.join(HF_CLONE, ".git")):
        log.info(f"Cloning {HF_REPO} to {HF_CLONE}")
        run(["git", "clone", repo_url, HF_CLONE])
        run(["git", "-C", HF_CLONE, "config", "user.email", "ais-collector@local"])
        run(["git", "-C", HF_CLONE, "config", "user.name", "AIS Collector"])
    else:
        run(["git", "-C", HF_CLONE, "pull"])

    # Copy CSV into the repo
    data_dir = os.path.join(HF_CLONE, "data")
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, f"{date_str}.csv")
    shutil.copy2(local_path, dest)
    log.info(f"  Copied to {dest}")

    # Stage the file, then check if there's anything new to commit
    run(["git", "-C", HF_CLONE, "add", "-f", dest])

    status = subprocess.run(
        ["git", "-C", HF_CLONE, "status", "--porcelain"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout.decode().strip()

    if not status:
        log.info("No changes — file already uploaded")
        return

    run(["git", "-C", HF_CLONE, "commit", "-m", f"AIS data {date_str}"])
    run(["git", "-C", HF_CLONE, "push", repo_url, "main"])
    log.info("Upload complete")


# ── Dataset card (run once) ───────────────────────────────────────────────────
def ensure_dataset_card():
    """
    Creates a README.md on the HF dataset repo if one doesn't exist yet.
    Call this manually once after creating the repo.
    """
    api  = HfApi()
    card = """---
language:
- en
tags:
- ais
- maritime
- vessel-tracking
- time-series
- hudson-river
license: cc-by-4.0
---

# AIS Hudson River — NYWaterway Ferry Tracking

Real-time AIS (Automatic Identification System) data collected from the
Hudson River using an RTL-SDR receiver and AIS-catcher.

## Schema

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime (UTC) | Position report time |
| mmsi | string | Maritime Mobile Service Identity |
| vessel_name | string | Vessel name (from AIS type 5/24) |
| vessel_type | int | AIS vessel type code (60-69 = passenger) |
| vessel_length_m | int | Length in meters |
| vessel_beam_m | int | Beam in meters |
| is_nywaterway | bool | Identified as NYWaterway ferry |
| lat | float | Latitude (WGS84) |
| lon | float | Longitude (WGS84) |
| sog | float | Speed over ground (knots) |
| cog | float | Course over ground (degrees) |
| heading | int | True heading (degrees) |
| nav_status | int | Navigational status (0=underway) |
| msg_type | int | AIS message type |

## Collection

- Receiver: RTL-SDR v3
- Software: AIS-catcher
- Location: Hudson River area, New York
- Schedule: Mon–Fri 07:00–10:00 and 16:00–19:00 (America/New_York)
- One CSV file per day under `data/`
"""
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="Add dataset card",
    )
    log.info("Dataset card created")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if HF_REPO == "your-username/ais-hudson-river":
        log.error("Set HF_REPO in upload_to_hf.py before running")
        sys.exit(1)

    # Accept optional date argument, default to today
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")

    local_path = export_day(date_str)
    if local_path:
        upload_to_hf(local_path, date_str)


if __name__ == "__main__":
    main()
