# S7/S8 Phase Audit And Rewrite Design

## Purpose

The user does not want a paraphrase of the existing S7/S8 diagnostic note. The goal is to audit whether the note has actually demonstrated physical correctness, or whether it only achieved numerical agreement by changing phase-reference conventions.

The work must produce a physically cautious critique and then a standard, academic-style rewrite. It must avoid treating the informal term "ledger" as accepted terminology.

## Scope

This design covers the S7-to-S8 free-space segment described in:

- `D:/BTS/sandbox/diagnostics/s7_s8_phase_root_cause.md`
- supporting S7/S8 diagnostic artifacts under `D:/BTS/sandbox/diagnostics/`
- relevant reference-frame implementation in `pop/io/zbf.py` and `pop/reference_frames.py`

It must not claim to solve VM1, S8-to-S12, or whole-system physical phase closure.

## Terminology

Replace the informal "ledger" vocabulary with standard terms:

- physical total field: `U(x,y)`
- reference-relative field or residual complex field: `u(x,y)`
- reference phase: `Phi_ref(x,y)`
- field representation
- reference phase convention
- endpoint reconstruction
- mixed-reference comparison

Use "reference phase convention" when describing the choice of carrier/reference phase removed from the total field. Use "mixed-reference residual" for residuals produced by comparing fields reconstructed with different reference phases.

## Physical Model

Use the factorization:

```text
U(x,y) = u(x,y) * exp(i * Phi_ref(x,y))
```

The central question is not whether a phase RMS is small in one comparison. The question is whether one physical field can be represented, propagated, and reconstructed consistently:

```text
U_S7 -> u_S7 under a defined Phi_ref,S7
u_S7 propagates to u_S8
u_S8 -> U_S8 under a defined Phi_ref,S8
```

For Zemax ZBF fields in POP/PROPER phasor convention, the expected conversion is:

```text
u_ZBF = conj(Ex_Zemax)
U_ZBF = conj(Ex_Zemax) * exp(i * axis_sign * Phi_ZBF)
```

For the validated equal-radius ZBF case, the ZBF reference phase is:

```text
Phi_ZBF = k * sign(R) * (sqrt(abs(R)^2 + x^2 + y^2) - abs(R))
k = 2*pi*n/lambda
```

This expression must not be confused with a PROPER pilot/qphase quadratic reference. If the report compares a PROPER-lifted physical field to a ZBF-exact-lifted field, the residual includes the difference between reference phase conventions.

## Audit Criteria

The critique must separate three levels of evidence.

1. Residual-field propagation closure:
   Does the propagated reference-relative field at S8 match the Zemax ZBF reference-relative field under the same phasor convention?

2. Endpoint physical reconstruction closure:
   After using the explicitly stated endpoint reference phase, does the reconstructed physical total field match the endpoint ZBF physical reconstruction?

3. Global physical correctness:
   Does the same convention survive later surfaces, especially VM1 and S8-to-S12, without diagnostic splicing or fitted scale factors?

Only the first two can be claimed for S7/S8 from the current evidence. The third must remain unproven.

## Suspicion Checks

The final audit must explicitly ask whether the small S7/S8 residuals could be numerical agreement by convention choice. It should treat the following as warning signs:

- success depends on choosing one endpoint reference convention but fails under another;
- a low RMS is reported without stating which reference phase was used;
- physical total phase and propagation residual field are corrected by different fitted factors;
- a comparison improves phase RMS but does not preserve downstream propagation or intensity;
- a reference-phase formula is asserted without a validated analytic expression.

For S7/S8, the current evidence is allowed to support local representation consistency, but not absolute global physical correctness.

## Required Rewrite Position

The rewritten S7/S8 conclusion should say:

```text
The S7-to-S8 free-space segment is locally consistent when both fields are compared in the same reference-relative representation and the endpoint physical field is reconstructed with the corresponding ZBF reference phase.
```

It should not say:

```text
The physical phase is solved.
```

It should preserve the key numbers with corrected interpretation:

- S7 ZBF-reference reconstructed physical RMS: `2.3866e-17 waves`
- S8 ZBF-reference reconstructed physical RMS: `0.000203888 waves`
- maximum S7/S8 intensity RMS: `0.000741144 %`
- mixed-reference S8 residual: `0.0319913 waves`

The `0.0319913 waves` value should be described as a mixed-reference residual. It is diagnostic evidence of a reference-phase mismatch, not direct proof of a free-space propagation-kernel error.

## Deliverables

After approval of this design, produce:

1. A Chinese audit report that explains the terminology, equations, and evidence hierarchy in standard technical language.
2. A rewritten S7/S8 section suitable for replacing or prepending to `s7_s8_phase_root_cause.md`.
3. A short list of anti-overclaim rules for future diagnostics.

## Non-Goals

- Do not edit production propagation code.
- Do not promote any fitted S12 scale factor as a physical correction.
- Do not use "ledger" as an unexplained technical term.
- Do not claim VM1 or whole-path physical phase closure.

## Acceptance Criteria

The final output is acceptable if it:

- uses standard terminology rather than "ledger";
- states the analytic ZBF reference phase explicitly and separately from the PROPER reference phase;
- distinguishes residual-field closure from physical-total-field correctness;
- preserves the user's skeptical stance;
- identifies which S7/S8 conclusions are supported and which are not;
- avoids presenting small RMS values as proof without naming the reference convention.
