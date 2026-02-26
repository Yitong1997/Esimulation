# POP (Physical Optical Propagation)

POP is a lightweight physical optics propagation package built on optiland and PROPER.

## Quick Start

```python
import pop

system = pop.load_zmx("system.zmx")
source = pop.GaussianSource(wavelength_um=0.633, w0_mm=5.0)
result = pop.propagate(system, source)
result.plot()
```

## Options & Reports

```python
import pop
from pop import DebugOptions, PlotOptions, PropagationOptions

system = pop.load_zmx("system.zmx")
source = pop.GaussianSource(wavelength_um=0.633, w0_mm=5.0)

options = PropagationOptions(num_rays=2000, auto_asm=True)
plot_options = PlotOptions(mode="normal", output_dir="output/run1", show=False, axis_3d=True)
debug_options = DebugOptions(enabled=True, plot_3d=True, plot_3d_ray_count=8)

result = pop.propagate(
    system,
    source,
    options=options,
    plot_options=plot_options,
    debug_options=debug_options,
)

# Generate a report index.md / index.html (plots saved under output/report)
result.save_report(output_dir="output/report", plot_set="report", include_layout=True)
```

## Notes

- Coordinate systems are built with minimal rotation to avoid roll flips.
- Exit planes are explicitly defined and aligned with the optical axis.
- Free-space propagation uses the effective distance d/n and syncs grid sampling.
- Reversed surfaces are normalized before propagation (orientation flip + radius/radius_x sign flip; even-asphere coefficients sign flip). Conic constants and material order stay unchanged.
- Continuous air surfaces and coordinate-break entries are treated as free-space only (no ray tracing).