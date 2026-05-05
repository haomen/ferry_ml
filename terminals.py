"""
Terminal definitions and geometry for the Hoboken 14 St ↔ NYC W39 route.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Terminal:
    name: str
    lat: float
    lon: float
    radius_nm: float = 0.12


# Coordinates derived from actual docking positions observed in AIS data.
# Radius 0.20 nm (~370 m) comfortably captures the approach / departure.
HOBOKEN_14 = Terminal("Hoboken 14 St", 40.7353, -74.0265, 0.20)
NYC_W39    = Terminal("NYC W39",       40.7604, -74.0040, 0.20)
TERMINALS  = [HOBOKEN_14, NYC_W39]
BY_NAME    = {t.name: t for t in TERMINALS}

# Confirmed NYWaterway vessels on this route (from data exploration)
ROUTE_MMSI: dict[str, str] = {
    "366853410": "FRED V MORRONE",
    "366629680": "GARDEN STATE",
    "366902350": "GOV THOMAS H KEAN",
    "366851680": "SEN FRANK LAUTENBERG",
    "367434220": "YOGI BERRA",
}

ARRIVAL_SOG_KT = 3.0   # vessel considered docked when SOG drops below this


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3440.065
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dλ = math.radians(lon2 - lon1)
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dλ) * math.cos(φ2)
    y = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(dλ)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def at_terminal(lat: float, lon: float, sog: Optional[float]) -> Optional[Terminal]:
    for t in TERMINALS:
        if haversine_nm(lat, lon, t.lat, t.lon) <= t.radius_nm:
            if sog is None or sog <= ARRIVAL_SOG_KT:
                return t
    return None
