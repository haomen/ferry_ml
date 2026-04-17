#!/usr/bin/env python3
"""
AIS Collector — listens for NMEA sentences from AIS-catcher on UDP,
decodes with pyais, and archives to TimescaleDB.

Runs 24/7 as a receiver station. Raw NMEA is logged continuously.
DB writes only occur during the configured collection windows so that
only operationally relevant data (ferry peak hours) is stored.

Handles multi-part AIS messages (e.g. type 5 voyage data which comes
in 2 parts). Also writes daily raw NMEA logs to ~/ais-data/raw/.

Run as: python3 collector.py
Or via: systemctl start ais-collector
"""

import os
import socket
import time
import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytz
import psycopg2
import psycopg2.extras
from pyais import decode as ais_decode
from pyais.exceptions import UnknownMessageException, InvalidNMEAMessageException

# ── Config ────────────────────────────────────────────────────────────────────
UDP_HOST  = "0.0.0.0"
UDP_PORT  = 10110
RAW_DIR   = os.path.expanduser("~/ais-data/raw")

DB_DSN    = "host=localhost port=5432 dbname=aisdb user=ais password=aispass"

# How long to wait for the second part of a multi-part message before discarding
MULTIPART_TIMEOUT_SEC = 5.0

# ── Collection windows ────────────────────────────────────────────────────────
# DB writes only happen during these local-time windows.
# Each entry: (start_hour, end_hour, weekdays)
#   weekdays: set of integers 0=Monday … 6=Sunday
# Raw NMEA logs are written continuously regardless of these windows.

LOCAL_TZ = pytz.timezone("America/New_York")

COLLECTION_WINDOWS = [
    (7, 10, {0, 1, 2, 3, 4}),   # 07:00–10:00 Mon–Fri
    (16, 19, {0, 1, 2, 3, 4}),  # 16:00–19:00 Mon–Fri
]


def in_collection_window() -> bool:
    now = datetime.now(LOCAL_TZ)
    for start_h, end_h, weekdays in COLLECTION_WINDOWS:
        if now.weekday() in weekdays and start_h <= now.hour < end_h:
            return True
    return False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("ais-collector")


# ── Database connection ───────────────────────────────────────────────────────
def connect_db(dsn: str, retries: int = 10) -> psycopg2.extensions.connection:
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(dsn)
            conn.autocommit = False
            log.info("Connected to TimescaleDB")
            return conn
        except psycopg2.OperationalError as e:
            log.warning(f"DB connect attempt {attempt}/{retries} failed: {e}")
            time.sleep(3)
    raise RuntimeError("Could not connect to TimescaleDB after retries")


# ── Raw NMEA daily log ────────────────────────────────────────────────────────
def get_raw_log(raw_dir: str):
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = os.path.join(raw_dir, f"{date_str}.nmea")
    return open(path, "a", buffering=1)  # line-buffered


# ── NMEA checksum validator ───────────────────────────────────────────────────
NMEA_RE = re.compile(r"^[!$](.+)\*([0-9A-Fa-f]{2})$")

def nmea_valid(sentence: str) -> bool:
    m = NMEA_RE.match(sentence.strip())
    if not m:
        return False
    body, chk = m.group(1), m.group(2)
    computed = 0
    for ch in body:
        computed ^= ord(ch)
    return computed == int(chk, 16)


# ── Multi-part message buffer ─────────────────────────────────────────────────
class MultipartBuffer:
    """
    AIS type 5 voyage data arrives as 2 separate UDP datagrams.
    Buffer by (seq_id, channel) and return when all parts are present.
    """
    def __init__(self):
        self._buf: dict = {}

    def add(self, sentence: str):
        parts = sentence.split(",")
        if len(parts) < 7:
            return None
        try:
            total = int(parts[1])
            num   = int(parts[2])
            seq   = parts[3]
            chan  = parts[4]
        except ValueError:
            return None

        if total == 1:
            return [sentence]

        key = (seq, chan)
        now = time.monotonic()

        # Expire stale buffers
        for k in list(self._buf):
            if self._buf[k] and now - self._buf[k][0][2] > MULTIPART_TIMEOUT_SEC:
                del self._buf[k]

        if key not in self._buf:
            self._buf[key] = []
        self._buf[key].append((num, sentence, now))

        assembled = self._buf[key]
        if len(assembled) == total:
            del self._buf[key]
            assembled.sort(key=lambda x: x[0])
            return [x[1] for x in assembled]

        return None


# ── DB write helpers ──────────────────────────────────────────────────────────
def upsert_vessel(cur, mmsi: str, **kwargs):
    """
    INSERT new vessel or UPDATE only the non-null fields provided.
    Uses PostgreSQL INSERT ... ON CONFLICT DO UPDATE.
    """
    now = datetime.now(timezone.utc)

    # Build the conflict update clause — only update fields we actually have
    update_pairs = {k: v for k, v in kwargs.items() if v is not None}
    update_pairs["last_seen"] = now

    cols   = ["mmsi", "first_seen", "last_seen"] + list(kwargs.keys())
    vals   = [mmsi, now, now] + list(kwargs.values())
    placeholders = ", ".join(["%s"] * len(vals))

    set_clause = ", ".join(
        f"{k} = EXCLUDED.{k}" for k in update_pairs
    )

    cur.execute(f"""
        INSERT INTO vessels ({', '.join(cols)})
        VALUES ({placeholders})
        ON CONFLICT (mmsi) DO UPDATE SET {set_clause}
    """, vals)


def insert_position(cur, ts: datetime, mmsi: str, **kwargs):
    cur.execute("""
        INSERT INTO positions (ts, mmsi, lat, lon, sog, cog, heading, nav_status, msg_type)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        ts, mmsi,
        kwargs.get("lat"),
        kwargs.get("lon"),
        kwargs.get("sog"),
        kwargs.get("cog"),
        kwargs.get("heading"),
        kwargs.get("nav_status"),
        kwargs.get("msg_type"),
    ))


# ── Message dispatcher ────────────────────────────────────────────────────────
def process_sentences(cur, sentences: list):
    try:
        msg = ais_decode(*sentences).asdict()
    except (UnknownMessageException, InvalidNMEAMessageException, Exception):
        return

    msg_type = msg.get("msg_type")
    mmsi     = str(msg.get("mmsi", "")).strip()
    if not mmsi:
        return

    ts = datetime.now(timezone.utc)

    # Position reports — Class A (1/2/3) and Class B (18/19)
    if msg_type in (1, 2, 3, 18, 19):
        lat = msg.get("lat")
        lon = msg.get("lon")
        sog = msg.get("speed")
        cog = msg.get("course")
        hdg = msg.get("heading")
        sta = msg.get("status")

        # Drop invalid positions
        if lat is not None and lon is not None:
            if (lat == 0.0 and lon == 0.0) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                lat, lon = None, None

        insert_position(cur, ts, mmsi,
                        lat=lat, lon=lon, sog=sog, cog=cog,
                        heading=hdg, nav_status=sta, msg_type=msg_type)
        upsert_vessel(cur, mmsi)

    # Voyage / static data — Class A type 5 (vessel name, dimensions)
    elif msg_type == 5:
        name     = (msg.get("shipname") or "").strip() or None
        callsign = (msg.get("callsign") or "").strip() or None
        imo      = str(msg.get("imo") or "").strip() or None
        vtype    = msg.get("ship_type")
        to_bow   = msg.get("to_bow")   or 0
        to_stern = msg.get("to_stern") or 0
        to_port  = msg.get("to_port")  or 0
        to_stbd  = msg.get("to_starboard") or 0
        length   = (to_bow + to_stern) or None
        beam     = (to_port + to_stbd) or None

        upsert_vessel(cur, mmsi,
                      name=name, callsign=callsign, imo=imo,
                      vessel_type=vtype, length=length, beam=beam)

    # Class B static — type 24
    elif msg_type == 24:
        part = msg.get("partno")
        if part == 0:
            name = (msg.get("shipname") or "").strip() or None
            upsert_vessel(cur, mmsi, name=name)
        elif part == 1:
            callsign = (msg.get("callsign") or "").strip() or None
            vtype    = msg.get("ship_type")
            upsert_vessel(cur, mmsi, callsign=callsign, vessel_type=vtype)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    conn = connect_db(DB_DSN)
    cur  = conn.cursor()
    buf  = MultipartBuffer()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, UDP_PORT))
    sock.settimeout(5.0)

    log.info(f"Listening on UDP {UDP_HOST}:{UDP_PORT}")
    log.info("DB writes active only during collection windows; NMEA logged 24/7")

    raw_log         = get_raw_log(RAW_DIR)
    last_date       = datetime.now(timezone.utc).date()
    last_commit     = time.time()
    last_window_log = time.time()
    msg_count       = 0
    db_active       = False
    BATCH_SIZE      = 50   # commit every N messages or every 10s

    try:
        while True:
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                # Periodic tasks even when no UDP data arrives
                now_in_window = in_collection_window()
                if db_active and not now_in_window:
                    conn.commit()
                    db_active = False
                    log.info("Collection window ended — DB writes paused")
                elif not db_active and now_in_window:
                    db_active = True
                    log.info("Collection window started — DB writes active")

                if db_active and time.time() - last_commit > 10:
                    conn.commit()
                    last_commit = time.time()
                continue

            # Rotate raw log at midnight local time
            today = datetime.now(LOCAL_TZ).date()
            if today != last_date:
                raw_log.close()
                raw_log   = get_raw_log(RAW_DIR)
                last_date = today

            now_in_window = in_collection_window()

            # Log window state transitions
            if now_in_window and not db_active:
                db_active = True
                log.info("Collection window started — DB writes active")
            elif not now_in_window and db_active:
                conn.commit()
                db_active = False
                log.info("Collection window ended — DB writes paused")

            for line in data.decode("ascii", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                if not (line.startswith("!AIVDM") or line.startswith("!AIVDO")):
                    continue
                if not nmea_valid(line):
                    continue

                # Always write raw NMEA log
                raw_log.write(line + "\n")

                # Only decode and write to DB during collection windows
                if not db_active:
                    continue

                complete = buf.add(line)
                if complete:
                    try:
                        process_sentences(cur, complete)
                        msg_count += 1
                    except Exception as e:
                        log.warning(f"Failed to process message: {e}")
                        conn.rollback()

            if db_active and (msg_count >= BATCH_SIZE or time.time() - last_commit > 10):
                try:
                    conn.commit()
                except Exception as e:
                    log.error(f"Commit failed: {e}")
                    conn.rollback()
                last_commit = time.time()
                if msg_count > 0:
                    log.info(f"Committed {msg_count} messages")
                    msg_count = 0

    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        try:
            conn.commit()
        except Exception:
            pass
        cur.close()
        conn.close()
        sock.close()
        raw_log.close()


if __name__ == "__main__":
    main()
