"""Quantity registry and physical plausibility bounds.

This is the hardcoded knowledge the `verify` node checks retrieved values
against. It is intentionally generous: the job of the bounds table is to catch
retrieval or parsing failures (a density of 900, a drag coefficient of 47),
not to adjudicate whether a value is the best available estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Quantity:
    key: str
    canonical_unit: str
    low: float
    high: float
    # Phrases that identify this quantity in a document table row.
    doc_aliases: Tuple[str, ...]
    # Phrases that identify this quantity in a user query.
    query_aliases: Tuple[str, ...] = field(default_factory=tuple)
    note: str = ""

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


QUANTITIES: List[Quantity] = [
    Quantity(
        key="drag_coefficient",
        canonical_unit="dimensionless",
        low=0.01,
        high=1.5,
        doc_aliases=("drag coefficient",),
        query_aliases=("drag coefficient", "cd", "c_d", "drag"),
        note="Frontal-area-referenced Cd; bluff bodies approach 1.1-1.3.",
    ),
    Quantity(
        key="air_density",
        canonical_unit="kg/m^3",
        low=1e-6,
        high=1.5,
        doc_aliases=("air density", "density"),
        query_aliases=("air density", "density", "rho"),
        note="ISA sea level is 1.225 kg/m^3 and falls monotonically with altitude.",
    ),
    Quantity(
        key="navigation_gain",
        canonical_unit="dimensionless",
        low=2.0,
        high=6.0,
        doc_aliases=("navigation gain",),
        query_aliases=(
            "navigation gain",
            "nav gain",
            "pn gain",
            "proportional navigation gain",
            "n prime",
            "n'",
        ),
        note="Effective navigation ratio N'; 3-5 is the usual working band.",
    ),
    Quantity(
        key="temperature",
        canonical_unit="K",
        low=150.0,
        high=340.0,
        doc_aliases=("temperature",),
        query_aliases=("temperature", "air temperature"),
    ),
    Quantity(
        key="pressure",
        canonical_unit="Pa",
        low=0.0,
        high=110000.0,
        doc_aliases=("pressure",),
        query_aliases=("pressure", "static pressure"),
    ),
    Quantity(
        key="speed_of_sound",
        canonical_unit="m/s",
        low=250.0,
        high=350.0,
        doc_aliases=("speed of sound",),
        query_aliases=("speed of sound", "sonic velocity", "mach 1"),
    ),
    Quantity(
        key="scale_height",
        canonical_unit="m",
        low=5000.0,
        high=10000.0,
        doc_aliases=("scale height",),
        query_aliases=("scale height", "exponential atmosphere"),
    ),
    Quantity(
        key="lapse_rate",
        canonical_unit="K/m",
        low=-0.01,
        high=0.01,
        doc_aliases=("lapse rate",),
        query_aliases=("lapse rate",),
    ),
    Quantity(
        key="gravity",
        canonical_unit="m/s^2",
        low=9.5,
        high=9.9,
        doc_aliases=("gravity",),
        query_aliases=("gravity", "g0", "gravitational acceleration"),
    ),
    Quantity(
        key="specific_gas_constant",
        canonical_unit="J/(kg*K)",
        low=250.0,
        high=320.0,
        doc_aliases=("specific gas constant",),
        query_aliases=("specific gas constant", "gas constant"),
    ),
    Quantity(
        key="ratio_of_specific_heats",
        canonical_unit="dimensionless",
        low=1.0,
        high=1.8,
        doc_aliases=("ratio of specific heats",),
        query_aliases=("ratio of specific heats", "gamma", "adiabatic index"),
    ),
    Quantity(
        key="dynamic_viscosity",
        canonical_unit="Pa*s",
        low=1e-6,
        high=1e-4,
        doc_aliases=("dynamic viscosity",),
        query_aliases=("dynamic viscosity", "viscosity", "mu"),
    ),
    Quantity(
        key="mach_number",
        canonical_unit="dimensionless",
        low=0.0,
        high=10.0,
        doc_aliases=("peak mach", "transonic peak mach"),
        query_aliases=("mach number", "peak mach"),
    ),
    Quantity(
        key="lateral_acceleration",
        canonical_unit="g",
        low=0.0,
        high=60.0,
        doc_aliases=("maximum lateral acceleration", "target lateral acceleration"),
        query_aliases=("lateral acceleration", "manoeuvre limit", "maneuver limit", "g limit"),
    ),
    Quantity(
        key="launch_mass",
        canonical_unit="kg",
        low=1.0,
        high=5000.0,
        doc_aliases=("launch mass",),
        query_aliases=("launch mass", "interceptor mass", "missile mass"),
    ),
    Quantity(
        key="body_diameter",
        canonical_unit="m",
        low=0.01,
        high=2.0,
        doc_aliases=("body diameter",),
        query_aliases=("body diameter", "airframe diameter", "calibre", "caliber"),
    ),
    Quantity(
        key="body_length",
        canonical_unit="m",
        low=0.1,
        high=20.0,
        doc_aliases=("body length",),
        query_aliases=("body length", "airframe length"),
    ),
    Quantity(
        key="reference_area",
        canonical_unit="m^2",
        low=1e-4,
        high=5.0,
        doc_aliases=("reference area",),
        query_aliases=("reference area", "frontal area"),
    ),
    Quantity(
        key="burn_time",
        canonical_unit="s",
        low=0.1,
        high=120.0,
        doc_aliases=("burn time",),
        query_aliases=("burn time", "motor burn"),
    ),
    Quantity(
        key="target_speed",
        canonical_unit="m/s",
        low=10.0,
        high=3000.0,
        doc_aliases=("target speed",),
        query_aliases=("target speed", "target velocity"),
    ),
    Quantity(
        key="closing_velocity",
        canonical_unit="m/s",
        low=10.0,
        high=5000.0,
        doc_aliases=("closing velocity",),
        query_aliases=("closing velocity", "closing speed", "vc"),
    ),
    Quantity(
        key="altitude",
        canonical_unit="m",
        low=0.0,
        high=100000.0,
        doc_aliases=("intercept altitude", "target altitude"),
        query_aliases=("intercept altitude", "target altitude"),
    ),
    Quantity(
        key="guidance_update_rate",
        canonical_unit="Hz",
        low=1.0,
        high=5000.0,
        doc_aliases=("guidance update rate",),
        query_aliases=("update rate", "loop rate", "guidance rate"),
    ),
    Quantity(
        key="simulation_timestep",
        canonical_unit="s",
        low=1e-6,
        high=1.0,
        doc_aliases=("simulation timestep", "terminal guidance cutoff range"),
        query_aliases=("timestep", "time step", "integration step"),
    ),
]

BY_KEY: Dict[str, Quantity] = {q.key: q for q in QUANTITIES}


def match_doc_alias(label: str) -> Optional[Quantity]:
    """Map a table-row label onto a registry quantity (longest alias wins)."""
    text = label.strip().lower()
    best: Optional[Quantity] = None
    best_len = 0
    for quantity in QUANTITIES:
        for alias in quantity.doc_aliases:
            if alias in text and len(alias) > best_len:
                best, best_len = quantity, len(alias)
    return best


def match_query(query: str) -> Optional[Quantity]:
    """Guess which quantity a natural-language query is asking about."""
    text = " " + query.strip().lower() + " "
    best: Optional[Quantity] = None
    best_len = 0
    for quantity in QUANTITIES:
        for alias in quantity.query_aliases:
            needle = alias if len(alias) > 3 else f" {alias} "
            if needle in text and len(alias) > best_len:
                best, best_len = quantity, len(alias)
    return best
