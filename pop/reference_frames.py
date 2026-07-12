"""Explicit reference-relative and physical-field bookkeeping.

The native PROPER array and a physical total field are different
representations of the same wavefront.  This module records the phase needed
to lift one representation into the other at a state boundary.  It never
changes the propagation kernel and it never combines an unrelated PROPER
``wfarr`` with a Zemax ZBF reference phase.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def reference_relative_field_from_state(state: Any) -> np.ndarray:
    """Return the saved native reference-relative field of a POP state."""

    explicit = getattr(state, "reference_relative_field", None)
    if explicit is not None:
        return np.asarray(explicit, dtype=np.complex128).copy()
    wfo = getattr(state, "proper_wfo", None)
    wfarr = getattr(wfo, "wfarr", None)
    if wfarr is not None:
        import proper

        return np.asarray(proper.prop_shift_center(wfarr), dtype=np.complex128).copy()
    if hasattr(state, "get_complex_amplitude"):
        return np.asarray(state.get_complex_amplitude(), dtype=np.complex128).copy()
    amplitude = np.asarray(getattr(state, "amplitude"), dtype=np.float64)
    phase = np.asarray(getattr(state, "phase"), dtype=np.float64)
    return amplitude * np.exp(1j * phase)


def physical_field_from_reference_relative(
    field: np.ndarray,
    reference_phase: np.ndarray,
) -> np.ndarray:
    """Lift a reference-relative field with its paired phase."""

    field_arr = np.asarray(field, dtype=np.complex128)
    phase_arr = np.asarray(reference_phase, dtype=np.float64)
    if field_arr.shape != phase_arr.shape:
        raise ValueError(
            f"reference phase shape {phase_arr.shape} does not match field shape {field_arr.shape}"
        )
    return field_arr * np.exp(1j * phase_arr)


def snapshot_reference_frame(
    wfo: Any,
    physical_phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Freeze PROPER's field and its paired lift phase at a state boundary."""

    import proper

    field = np.asarray(proper.prop_shift_center(wfo.wfarr), dtype=np.complex128).copy()
    phase_arr = np.asarray(physical_phase, dtype=np.float64)
    if phase_arr.shape != field.shape:
        raise ValueError(
            f"physical phase shape {phase_arr.shape} does not match field shape {field.shape}"
        )
    lift_phase = np.angle(np.exp(1j * (phase_arr - np.angle(field))))
    return field, lift_phase
