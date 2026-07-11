# Task 11 implementation report

## Scope

Implemented only the six authorized Task 11 files:

- `sandbox/free_space_algorithm_identification/native_report.py`
- `tests/free_space_identification/test_native_report.py`
- `tests/free_space_identification/fixtures/native_oo_report.txt`
- `tests/free_space_identification/fixtures/native_oi_report.txt`
- `tests/free_space_identification/fixtures/native_io_report.txt`
- `.superpowers/sdd/task-11-report.md`

No ZOS-API runner, propagation candidate, ZBF codec, physical-field
reconstruction, or decision code was changed. The fixtures are parser inputs
copied from a historical report; they are not experimental evidence for the
final algorithm decision.

## TDD record

Every Python command used `PYTHONDONTWRITEBYTECODE=1`, `python -B`, and disabled
the pytest cache provider.

The four high-information test functions and three fixtures were added before
the production module. The first focused command stopped during collection as
required:

```text
ModuleNotFoundError: No module named
'sandbox.free_space_algorithm_identification.native_report'
1 error during collection
```

After the minimum implementation, the focused run with warnings promoted to
errors was GREEN:

```text
6 passed in 0.41s
```

A runner-interface follow-up then parameterized the existing settings-readback
test for both a transfer (`Start=7`, `End=8`) and an identity probe (`Start=7`,
`End=7`). Before changing the setting contract, the identity case reproduced
the restrictive behavior:

```text
1 failed, 6 passed in 0.48s
ValueError: end_surface must follow start_surface
```

The contract now permits `Start=End` and rejects only `End<Start`.
`validate_native_transfer()` still accepts a real `SegmentSpec`, so an identity
settings probe is not treated as a physical propagation segment. Final focused
result:

```text
7 passed in 0.40s
```

There remain exactly four top-level test functions. Parameterization produces
three OO/OI/IO parser cases and two transfer/identity readback cases, for seven
pytest items total.

## Implemented contracts

- The parser accepts one compact transfer and retains only the literal OO/OI/IO
  label, signed total distance, two printed X/Y interval/width grids, two
  printed pilot waist/position/Rayleigh states, and every warning line in
  order, including duplicates.
- `ReportNumber` tokens retain printed resolution. Rounded TXT values are used
  only as containment tolerances; they are never promoted to exact binary
  settings.
- TXT-derived `NativePopReport` deliberately has no Start/End, N, sample enum,
  wavelength/field, polarization, or normalization fields. Those values live
  only in explicit request and API-readback records.
- Request/readback validation covers Start/End, N, the actual sample enum,
  widths, wavelength and field numbers, polarization, normalization, input and
  output filenames, and output-save flags.
- Sampling validation closes the source/output ZBF-header counts and intervals,
  source `N*dx`/`N*dy` versus read-back widths, polarization, and rounded native
  input/output grids.
- Transfer validation enforces the segment branch, both surface conventions,
  `axis_sign * d_report = d_model`, both raw ZBF pilot-position deltas equal to
  `d_report`, endpoint pilot values, and strict Rayleigh inside/outside states.
  The S7 convention therefore requires the literal `-368.6 mm` report distance
  for the positive `368.6 mm` model segment.
- The literal propagator label exposes only `branch`; `kernel_identity` is
  always `None`. No Fresnel, ASM, or RS identity is inferred.
- All malformed, ambiguous, omitted, non-finite, or altered critical fields
  fail closed with `ValueError`.

## Fixture provenance check

Each nonblank fixture line was compared case-sensitively and in order against
the read-only historical
`D:\BTS\.worktrees\residual-phase\sandbox\Zemax_baseline\biconic_focus_test.txt`.
The result was:

```text
native_io_report.txt: literal ordered subsequence PASS (27 lines)
native_oi_report.txt: literal ordered subsequence PASS (26 lines)
native_oo_report.txt: literal ordered subsequence PASS (22 lines)
```

## Verification

The cumulative command used the Git-tracked free-space tests plus Task 11 and
explicitly excluded the parallel Task 9 test file as requested:

```text
cumulative files (Task 9 excluded): 12
82 passed, 2 skipped in 1.97s
```

The focused and cumulative runs used `-W error`. No `.pyc` file exists under
the touched package or test tree. The scoped staged diff is checked separately
before commit.

## Remaining work

Task 11 performs no live OpticStudio run and makes no kernel-identification
claim. A later runner task must save new request/readback/report/ZBF receipts
and apply these validators to those fresh artifacts.
