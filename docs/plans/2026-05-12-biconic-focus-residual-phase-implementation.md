# Biconic Focus Residual Phase Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add controlled residual-phase experiment and S23 core reconstruction modes so the expand validation can prove and reduce residual-phase-induced focus ring artifacts.

**Architecture:** Keep the current propagation path as the default. Add residual phase mode options to `PropagationOptions`, route them through `pop.__init__.propagate()` into `propagation.element.propagate_element()`, and isolate the reconstruction logic in small helpers that can be unit tested without running the full Zemax sandbox script.

**Tech Stack:** Python dataclasses, NumPy, SciPy `griddata` through existing `pop.wavefront.reconstructor`, pytest, existing sandbox validation scripts.

---

### Task 1: Add Option Fields

**Files:**
- Modify: `pop/options.py`
- Modify: `sandbox/run_biconic_focus_expand_pop.py`
- Test: `tests/test_residual_phase_options.py`

**Step 1: Write the failing test**

Create `tests/test_residual_phase_options.py`:

```python
from pop.options import PropagationOptions


def test_residual_phase_options_defaults_are_normal_mode():
    options = PropagationOptions()

    assert options.residual_phase_mode == "normal"
    assert options.zero_residual_phase_surface_indices is None
    assert options.core_recon_surface_indices is None
    assert options.core_intensity_threshold == 0.1
    assert options.core_taper_width_pixels == 8
    assert options.core_min_rays == 64
    assert options.core_fallback_mode == "normal"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_residual_phase_options.py -v`

Expected: FAIL with `AttributeError` for `residual_phase_mode`.

**Step 3: Add fields to `PropagationOptions`**

In `pop/options.py`, add fields near `phase_method`:

```python
    residual_phase_mode: str = "normal"
    zero_residual_phase_surface_indices: Optional[Sequence[int] | str] = None
    core_recon_surface_indices: Optional[Sequence[int] | str] = None
    core_intensity_threshold: float = 0.1
    core_taper_width_pixels: int = 8
    core_min_rays: int = 64
    core_fallback_mode: str = "normal"
```

**Step 4: Wire script config**

In `sandbox/run_biconic_focus_expand_pop.py`, add top-level config keys with default normal behavior:

```python
    "RESIDUAL_PHASE_MODE": "normal",
    "ZERO_RESIDUAL_PHASE_SURFACE_INDICES": None,
    "CORE_RECON_SURFACE_INDICES": None,
    "CORE_INTENSITY_THRESHOLD": 0.1,
    "CORE_TAPER_WIDTH_PIXELS": 8,
    "CORE_MIN_RAYS": 64,
    "CORE_FALLBACK_MODE": "normal",
```

Pass them into `PropagationOptions(...)` using matching lowercase names.

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_residual_phase_options.py -v`

Expected: PASS.

**Step 6: Commit**

```bash
git add pop/options.py sandbox/run_biconic_focus_expand_pop.py tests/test_residual_phase_options.py
git commit -m "feat: add residual phase mode options"
```

### Task 2: Route Options To Element Propagation

**Files:**
- Modify: `pop/__init__.py`
- Modify: `pop/propagation/element.py`
- Test: `tests/test_residual_phase_option_routing.py`

**Step 1: Inspect current call sites**

Find the `propagation.element.propagate_element(...)` call in `pop/__init__.py` near line 906 and the `propagate_element` signature in `pop/propagation/element.py` near line 900.

**Step 2: Write the failing test**

Create `tests/test_residual_phase_option_routing.py`:

```python
import inspect

from pop.propagation.element import propagate_element


def test_propagate_element_accepts_residual_phase_controls():
    params = inspect.signature(propagate_element).parameters

    assert "residual_phase_mode" in params
    assert "zero_residual_phase_surface_indices" in params
    assert "core_recon_surface_indices" in params
    assert "core_intensity_threshold" in params
    assert "core_taper_width_pixels" in params
    assert "core_min_rays" in params
    assert "core_fallback_mode" in params
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_residual_phase_option_routing.py -v`

Expected: FAIL because the parameters do not exist.

**Step 4: Add keyword parameters**

Add the new keyword parameters to `propagate_element(...)` with the same defaults as `PropagationOptions`.

In `pop/__init__.py`, pass values from `options` into the element call:

```python
residual_phase_mode=options.residual_phase_mode,
zero_residual_phase_surface_indices=options.zero_residual_phase_surface_indices,
core_recon_surface_indices=options.core_recon_surface_indices,
core_intensity_threshold=options.core_intensity_threshold,
core_taper_width_pixels=options.core_taper_width_pixels,
core_min_rays=options.core_min_rays,
core_fallback_mode=options.core_fallback_mode,
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_residual_phase_option_routing.py -v`

Expected: PASS.

**Step 6: Commit**

```bash
git add pop/__init__.py pop/propagation/element.py tests/test_residual_phase_option_routing.py
git commit -m "feat: route residual phase controls to elements"
```

### Task 3: Add Surface Selection Helper

**Files:**
- Modify: `pop/propagation/element.py`
- Test: `tests/test_residual_phase_surface_selection.py`

**Step 1: Write failing tests**

Create `tests/test_residual_phase_surface_selection.py`:

```python
from pop.propagation.element import _surface_selected


def test_surface_selected_supports_none_all_and_lists():
    assert not _surface_selected(23, None)
    assert _surface_selected(23, "all")
    assert _surface_selected(23, [20, 23])
    assert not _surface_selected(23, [20])


def test_surface_selected_supports_comma_string():
    assert _surface_selected(23, "20,23")
    assert not _surface_selected(23, "20,25")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_residual_phase_surface_selection.py -v`

Expected: FAIL because `_surface_selected` does not exist.

**Step 3: Implement helper**

In `pop/propagation/element.py`, add near the existing helper functions:

```python
def _surface_selected(surface_index: int, selection: Optional[Sequence[int] | str]) -> bool:
    if selection is None:
        return False
    if isinstance(selection, str):
        raw = selection.strip().lower()
        if raw == "all":
            return True
        if not raw:
            return False
        selected: set[int] = set()
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                selected.add(int(token))
            except ValueError:
                continue
        return surface_index in selected
    return int(surface_index) in {int(value) for value in selection}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_residual_phase_surface_selection.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pop/propagation/element.py tests/test_residual_phase_surface_selection.py
git commit -m "feat: add residual phase surface selector"
```

### Task 4: Add Core Mask Helper

**Files:**
- Modify: `pop/propagation/element.py`
- Test: `tests/test_residual_phase_core_mask.py`

**Step 1: Write failing tests**

Create `tests/test_residual_phase_core_mask.py`:

```python
import numpy as np

from pop.propagation.element import _build_core_reconstruction_mask


def test_core_mask_selects_high_intensity_rays():
    amplitude = np.array([1.0, 0.8, 0.2, 0.01])

    mask, reason = _build_core_reconstruction_mask(
        input_amplitude=amplitude,
        threshold=0.25,
        min_rays=2,
    )

    assert reason is None
    assert mask.tolist() == [True, True, False, False]


def test_core_mask_reports_too_few_rays():
    amplitude = np.array([1.0, 0.1, 0.01])

    mask, reason = _build_core_reconstruction_mask(
        input_amplitude=amplitude,
        threshold=0.5,
        min_rays=2,
    )

    assert mask.tolist() == [True, False, False]
    assert reason == "core_min_rays"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_residual_phase_core_mask.py -v`

Expected: FAIL because `_build_core_reconstruction_mask` does not exist.

**Step 3: Implement helper**

Add to `pop/propagation/element.py`:

```python
def _build_core_reconstruction_mask(
    *,
    input_amplitude: NDArray[np.floating],
    threshold: float,
    min_rays: int,
) -> tuple[NDArray[np.bool_], Optional[str]]:
    amp = np.asarray(input_amplitude, dtype=float).ravel()
    finite = np.isfinite(amp)
    if not np.any(finite):
        return np.zeros_like(amp, dtype=bool), "no_finite_amplitude"

    peak = float(np.max(amp[finite]))
    if peak <= 0.0:
        return np.zeros_like(amp, dtype=bool), "non_positive_peak"

    threshold = float(np.clip(threshold, 0.0, 1.0))
    mask = finite & (amp >= threshold * peak)
    if int(np.count_nonzero(mask)) < int(min_rays):
        return mask, "core_min_rays"
    return mask, None
```

Use amplitude threshold, not intensity threshold, unless explicitly renamed. If using intensity threshold, square the amplitude and name the option accordingly.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_residual_phase_core_mask.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pop/propagation/element.py tests/test_residual_phase_core_mask.py
git commit -m "feat: add core reconstruction mask helper"
```

### Task 5: Add Residual Phase Post-Processing Helper

**Files:**
- Modify: `pop/propagation/element.py`
- Modify: `pop/result.py`
- Test: `tests/test_residual_phase_modes.py`

**Step 1: Write failing tests**

Create `tests/test_residual_phase_modes.py`:

```python
import numpy as np

from pop.propagation.element import _apply_residual_phase_mode


def test_zero_mode_zeros_residual_phase_on_selected_surface():
    residual_phase = np.ones((4, 4))
    amplitude = np.ones((4, 4))
    mask = np.ones(10, dtype=bool)

    new_phase, new_mask, info = _apply_residual_phase_mode(
        surface_index=23,
        residual_phase_grid=residual_phase,
        amplitude_grid=amplitude,
        ray_mask=mask,
        mode="zero_at_surfaces",
        zero_selection=[23],
        core_selection=None,
        taper_width_pixels=8,
    )

    assert np.all(new_phase == 0.0)
    assert np.all(new_mask)
    assert info["mode_applied"] == "zero_at_surfaces"


def test_normal_mode_returns_phase_unchanged():
    residual_phase = np.arange(9, dtype=float).reshape(3, 3)
    amplitude = np.ones((3, 3))
    mask = np.ones(4, dtype=bool)

    new_phase, new_mask, info = _apply_residual_phase_mode(
        surface_index=23,
        residual_phase_grid=residual_phase,
        amplitude_grid=amplitude,
        ray_mask=mask,
        mode="normal",
        zero_selection=[23],
        core_selection=[23],
        taper_width_pixels=8,
    )

    assert np.array_equal(new_phase, residual_phase)
    assert np.all(new_mask)
    assert info["mode_applied"] == "normal"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_residual_phase_modes.py -v`

Expected: FAIL because `_apply_residual_phase_mode` does not exist.

**Step 3: Add debug metadata field**

In `pop/result.py`, extend `SurfaceDebugInfo`:

```python
    residual_phase_mode_info: Optional[dict[str, object]] = None
```

**Step 4: Implement helper**

Add a minimal helper in `pop/propagation/element.py`:

```python
def _apply_residual_phase_mode(
    *,
    surface_index: int,
    residual_phase_grid: NDArray[np.floating],
    amplitude_grid: NDArray[np.floating],
    ray_mask: NDArray[np.bool_],
    mode: str,
    zero_selection: Optional[Sequence[int] | str],
    core_selection: Optional[Sequence[int] | str],
    taper_width_pixels: int,
) -> tuple[NDArray[np.floating], NDArray[np.bool_], dict[str, object]]:
    mode = (mode or "normal").strip().lower()
    info: dict[str, object] = {
        "requested_mode": mode,
        "mode_applied": "normal",
        "surface_index": int(surface_index),
    }

    if mode == "zero_at_surfaces" and _surface_selected(surface_index, zero_selection):
        info["mode_applied"] = "zero_at_surfaces"
        return np.zeros_like(residual_phase_grid, dtype=float), ray_mask, info

    info["mode_applied"] = "normal"
    return residual_phase_grid, ray_mask, info
```

Leave `core_griddata` out of this helper for now; Task 6 will apply it before reconstruction.

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_residual_phase_modes.py -v`

Expected: PASS.

**Step 6: Commit**

```bash
git add pop/propagation/element.py pop/result.py tests/test_residual_phase_modes.py
git commit -m "feat: add residual phase mode metadata"
```

### Task 6: Implement Zero-Residual Experiment Path

**Files:**
- Modify: `pop/propagation/element.py`
- Test: `tests/test_residual_phase_modes.py`

**Step 1: Add integration test with monkeypatch**

Add a focused test that calls `_apply_residual_phase_mode` with non-selected surface and selected surface. This avoids needing full optical system fixtures.

```python
def test_zero_mode_ignores_unselected_surface():
    residual_phase = np.ones((2, 2))
    amplitude = np.ones((2, 2))
    mask = np.ones(3, dtype=bool)

    new_phase, _, info = _apply_residual_phase_mode(
        surface_index=20,
        residual_phase_grid=residual_phase,
        amplitude_grid=amplitude,
        ray_mask=mask,
        mode="zero_at_surfaces",
        zero_selection=[23],
        core_selection=None,
        taper_width_pixels=8,
    )

    assert np.array_equal(new_phase, residual_phase)
    assert info["mode_applied"] == "normal"
```

**Step 2: Run test**

Run: `pytest tests/test_residual_phase_modes.py -v`

Expected: PASS before integration or after; this test protects selection logic.

**Step 3: Integrate into `propagate_element`**

After `reconstruct_wavefront(...)` and before `pilot_phase = new_pilot.compute_phase_grid(...)`, call:

```python
residual_phase_grid, reconstruction_mask, residual_phase_mode_info = _apply_residual_phase_mode(
    surface_index=target_surface_index,
    residual_phase_grid=residual_phase_grid,
    amplitude_grid=amplitude_grid,
    ray_mask=reconstruction_mask,
    mode=residual_phase_mode,
    zero_selection=zero_residual_phase_surface_indices,
    core_selection=core_recon_surface_indices,
    taper_width_pixels=core_taper_width_pixels,
)
```

Store `residual_phase_mode_info` in `SurfaceDebugInfo`.

**Step 4: Run focused tests**

Run:

```bash
pytest tests/test_residual_phase_options.py tests/test_residual_phase_option_routing.py tests/test_residual_phase_surface_selection.py tests/test_residual_phase_modes.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add pop/propagation/element.py tests/test_residual_phase_modes.py
git commit -m "feat: add zero residual phase experiment"
```

### Task 7: Implement Core Griddata Ray Filtering

**Files:**
- Modify: `pop/propagation/element.py`
- Test: `tests/test_residual_phase_core_mask.py`

**Step 1: Write failing behavior test for reconstruction arrays**

Add to `tests/test_residual_phase_core_mask.py`:

```python
from pop.propagation.element import _select_reconstruction_inputs


def test_select_reconstruction_inputs_filters_core_mode():
    ray_x = np.array([0.0, 1.0, 2.0])
    ray_y = np.array([0.0, 1.0, 2.0])
    residual = np.array([0.0, 0.1, 2.0])
    amplitude = np.array([1.0, 0.8, 0.01])

    selected, info = _select_reconstruction_inputs(
        surface_index=23,
        mode="core_griddata",
        core_selection=[23],
        ray_x_in=ray_x,
        ray_y_in=ray_y,
        ray_x_out=ray_x,
        ray_y_out=ray_y,
        residual_opd_waves=residual,
        input_amplitude=amplitude,
        threshold=0.1,
        min_rays=2,
        fallback_mode="normal",
    )

    assert selected["ray_x_in"].tolist() == [0.0, 1.0]
    assert selected["residual_opd_waves"].tolist() == [0.0, 0.1]
    assert selected["reconstruction_mask"].tolist() == [True, True, False]
    assert info["mode_applied"] == "core_griddata"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_residual_phase_core_mask.py -v`

Expected: FAIL because `_select_reconstruction_inputs` does not exist.

**Step 3: Implement helper**

Add `_select_reconstruction_inputs(...)` in `pop/propagation/element.py`. It should:

- return unmodified arrays and all-True mask for `normal`;
- apply `_build_core_reconstruction_mask(...)` only when `mode == "core_griddata"` and the surface is selected;
- if mask is invalid, return unmodified arrays with `mode_applied` set to fallback mode and `fallback_reason` populated;
- include full-length `reconstruction_mask` for debug plots.

**Step 4: Integrate before `reconstruct_wavefront`**

Replace the current block:

```python
residual_opd_waves_recon = residual_opd_waves
input_amplitude_recon = input_amplitude
reconstruction_mask = np.ones(len(ray_x_in), dtype=bool)
```

with a call to `_select_reconstruction_inputs(...)`, then pass selected arrays to `reconstruct_wavefront`.

**Step 5: Store metadata**

Merge selection metadata with post-processing metadata into `residual_phase_mode_info`.

**Step 6: Run tests**

Run:

```bash
pytest tests/test_residual_phase_core_mask.py tests/test_residual_phase_modes.py -v
```

Expected: PASS.

**Step 7: Commit**

```bash
git add pop/propagation/element.py tests/test_residual_phase_core_mask.py
git commit -m "feat: filter residual reconstruction to beam core"
```

### Task 8: Add Core Phase Taper

**Files:**
- Modify: `pop/propagation/element.py`
- Test: `tests/test_residual_phase_taper.py`

**Step 1: Write failing tests**

Create `tests/test_residual_phase_taper.py`:

```python
import numpy as np

from pop.propagation.element import _taper_residual_phase_to_core


def test_taper_keeps_high_amplitude_core_and_suppresses_edges():
    phase = np.ones((5, 5))
    amplitude = np.zeros((5, 5))
    amplitude[2, 2] = 1.0
    amplitude[1:4, 1:4] = 0.5

    tapered = _taper_residual_phase_to_core(
        residual_phase_grid=phase,
        amplitude_grid=amplitude,
        threshold=0.4,
        taper_width_pixels=1,
    )

    assert tapered[2, 2] == 1.0
    assert tapered[0, 0] == 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_residual_phase_taper.py -v`

Expected: FAIL because helper does not exist.

**Step 3: Implement helper**

Use `scipy.ndimage.distance_transform_edt` if SciPy is available. Since SciPy is already required by `griddata`, this is acceptable.

Implementation behavior:

- compute `core_grid = amplitude_grid >= threshold * max(amplitude_grid)`;
- if no core exists, return residual unchanged and let metadata record fallback elsewhere;
- compute distance from core with `distance_transform_edt(~core_grid)`;
- weights are `1` inside core, decay linearly to `0` by `taper_width_pixels`, and stay `0` outside;
- return `np.nan_to_num(residual_phase_grid * weights)`.

**Step 4: Apply taper after core reconstruction**

In `propagate_element`, after `reconstruct_wavefront(...)`, if `mode_applied == "core_griddata"`, call `_taper_residual_phase_to_core(...)`.

**Step 5: Run tests**

Run:

```bash
pytest tests/test_residual_phase_taper.py tests/test_residual_phase_core_mask.py -v
```

Expected: PASS.

**Step 6: Commit**

```bash
git add pop/propagation/element.py tests/test_residual_phase_taper.py
git commit -m "feat: taper core residual phase edges"
```

### Task 9: Add Sandbox Experiment Runs

**Files:**
- Modify: `sandbox/run_biconic_focus_expand_pop.py`
- Create: `sandbox/run_biconic_focus_expand_residual_phase_sweep.py`

**Step 1: Create sweep script**

Create a script that imports `run_simulation` from `sandbox/run_biconic_focus_expand_pop.py` and runs:

```python
CASES = {
    "normal": {
        "RESIDUAL_PHASE_MODE": "normal",
        "OUTPUT_DIR": OUTPUT_ROOT / "normal",
    },
    "zero_at_s23": {
        "RESIDUAL_PHASE_MODE": "zero_at_surfaces",
        "ZERO_RESIDUAL_PHASE_SURFACE_INDICES": [23],
        "OUTPUT_DIR": OUTPUT_ROOT / "zero_at_s23",
    },
    "core_griddata_s23": {
        "RESIDUAL_PHASE_MODE": "core_griddata",
        "CORE_RECON_SURFACE_INDICES": [23],
        "CORE_INTENSITY_THRESHOLD": 0.1,
        "CORE_TAPER_WIDTH_PIXELS": 8,
        "CORE_MIN_RAYS": 64,
        "OUTPUT_DIR": OUTPUT_ROOT / "core_griddata_s23",
    },
}
```

Print one summary line per case with final beam width and pilot size.

**Step 2: Run syntax check**

Run: `python -m py_compile sandbox/run_biconic_focus_expand_residual_phase_sweep.py`

Expected: PASS.

**Step 3: Commit**

```bash
git add sandbox/run_biconic_focus_expand_pop.py sandbox/run_biconic_focus_expand_residual_phase_sweep.py
git commit -m "chore: add residual phase sweep script"
```

### Task 10: Run Verification

**Files:**
- Read: `sandbox/output/biconic_focus_expand_residual_phase_sweep/**`
- Read: `sandbox/Zemax_baseline/biconic_focus_test_expand_validation.txt`

**Step 1: Run focused tests**

Run:

```bash
pytest tests/test_residual_phase_options.py tests/test_residual_phase_option_routing.py tests/test_residual_phase_surface_selection.py tests/test_residual_phase_core_mask.py tests/test_residual_phase_modes.py tests/test_residual_phase_taper.py -v
```

Expected: all PASS.

**Step 2: Run existing broad tests if available**

Run: `pytest -v`

Expected: PASS, or document unrelated failures if existing dirty workspace changes affect them.

**Step 3: Run the sweep**

Run: `python sandbox/run_biconic_focus_expand_residual_phase_sweep.py`

Expected:

- `normal` reproduces beam width near `0.178 mm`;
- `zero_at_s23` beam width moves near `0.0505 mm` if residual phase is causal;
- `core_griddata_s23` improves beam width and reduces ring artifact.

**Step 4: Compare with Zemax validation**

Use the existing validation tooling or read each case's `sim_metrics.json`. Record:

- `beam_width_x_mm`
- `beam_width_y_mm`
- `pilot_size_mm`
- S23/S25 `exit_wavefront` diagnostics

**Step 5: Commit verification notes if useful**

If the sweep script produces a compact JSON/Markdown summary, commit only that summary, not large generated images:

```bash
git add sandbox/output/biconic_focus_expand_residual_phase_sweep/summary.json
git commit -m "test: record residual phase sweep summary"
```

Do not commit large debug PNGs unless explicitly requested.

### Task 11: Decide Default Expand Validation Mode

**Files:**
- Modify: `sandbox/run_biconic_focus_expand_pop.py`
- Optional Modify: `docs/plans/2026-05-12-biconic-focus-residual-phase-design.md`

**Step 1: Inspect sweep results**

Choose the default for `sandbox/run_biconic_focus_expand_pop.py` only:

- keep `"normal"` if `zero_at_s23` does not prove causality;
- set `"core_griddata"` with S23 selected if it improves final beam width without degrading pilot;
- keep global library defaults as `"normal"` either way.

**Step 2: Update script defaults if justified**

Only change script-level defaults, not global defaults.

**Step 3: Run targeted validation**

Run: `python sandbox/run_biconic_focus_expand_pop.py`

Expected: final metrics improve relative to current `0.178 mm` beam width.

**Step 4: Commit**

```bash
git add sandbox/run_biconic_focus_expand_pop.py
git commit -m "fix: use core residual reconstruction for expand validation"
```
