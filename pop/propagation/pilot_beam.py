"""Pilot beam ABCD updates."""

from __future__ import annotations

import numpy as np

from pop.core import PilotBeamParams


def apply_free_space(pilot: PilotBeamParams, distance_mm: float) -> PilotBeamParams:
    return pilot.propagate(distance_mm)


def apply_mirror(
    pilot: PilotBeamParams,
    radius_mm: float,
    d_off_axis_mm: float = 0.0,
    sign_factor: float = 1.0,
) -> PilotBeamParams:
    if np.isinf(radius_mm):
        return pilot
    effective_radius = radius_mm
    if d_off_axis_mm != 0.0:
        effective_radius = radius_mm + (d_off_axis_mm**2) / radius_mm
    return pilot.apply_mirror(effective_radius * sign_factor)


def apply_refraction(
    pilot: PilotBeamParams,
    radius_mm: float,
    n1: float,
    n2: float,
    sign_factor: float = 1.0,
) -> PilotBeamParams:
    return pilot.apply_refraction(radius_mm * sign_factor, n1, n2)
