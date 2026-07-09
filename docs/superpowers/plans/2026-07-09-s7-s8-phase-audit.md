# S7/S8 Phase Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a skeptical, standards-based Chinese audit of the S7/S8 phase argument and add a safer rewritten interpretation to the diagnostic record.

**Architecture:** This is a documentation-only implementation. A standalone audit report will hold the full critique, equations, evidence hierarchy, and anti-overclaim rules; the existing root-cause log will receive a concise audited interpretation at the top without deleting historical evidence.

**Tech Stack:** Markdown documentation, existing diagnostic JSON artifacts, existing Python source as evidence only.

## Global Constraints

- Do not edit production propagation code.
- Do not promote any fitted S12 scale factor as a physical correction.
- Do not use "ledger" as an unexplained technical term.
- Do not claim VM1 or whole-path physical phase closure.
- State the ZBF reference phase explicitly and keep it separate from the PROPER pilot/qphase reference.
- Preserve the user's skeptical stance: small RMS values are evidence only after naming the reference phase convention.

---

### Task 1: Create The Audit Report

**Files:**
- Create: `sandbox/diagnostics/s7_s8_phase_audit.md`
- Create tracked mirror: `docs/s7_s8_phase_audit.md`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-07-09-s7-s8-phase-audit-design.md`, `sandbox/diagnostics/s7_s8_reference_relative_handoff/s7_s8_reference_relative_handoff_summary.json`, `sandbox/diagnostics/s7_s8_pilot_sampling_tilt_audit/s7_s8_pilot_sampling_tilt_audit.json`, `pop/io/zbf.py`
- Produces: A Chinese audit report with definitions, equations, evidence table, skeptical verdict, and anti-overclaim rules. The sandbox copy is the in-place diagnostic artifact; the `docs/` copy is the tracked mirror for review and commit because `sandbox/` is ignored.

- [ ] **Step 1: Write the report**

Create `sandbox/diagnostics/s7_s8_phase_audit.md` with these sections and required content:

```markdown
# S7/S8 Phase Interpretation Audit

## 审计结论
State that S7/S8 is locally consistent only under the explicitly named
reference-relative representation and ZBF endpoint reconstruction. State that
global physical phase correctness remains unproven.

## 术语替换
Define "field representation", "reference phase convention",
"reference-relative field", "physical total field", "endpoint reconstruction",
and "mixed-reference residual". State that "ledger" is an informal historical
label and not a formal optics term.

## 场表示与参考相位
Introduce `U(x,y) = u(x,y) exp(i Phi_ref(x,y))`, the POP-convention ZBF
conversion `u_ZBF = conj(Ex_Zemax)`, and the ZBF endpoint reconstruction
formula.

## 证据分层
Separate residual-field propagation closure, endpoint physical reconstruction
closure, and global physical correctness. Assign the S7/S8 evidence only to the
first two categories.

## 对 0.0319913 waves 的解释
Explain that this is a mixed-reference residual from comparing a PROPER-lifted
field against a ZBF-exact endpoint field, not direct evidence of free-space
propagation-kernel failure.

## 怀疑性审查
List the current evidence supporting local consistency and the reasons it does
not exclude every possible reference-phase convention error.

## 规范改写文本
Provide a replacement/prependable English section for
`s7_s8_phase_root_cause.md`.

## 后续诊断不得越界的规则
List concise rules that require every low RMS claim to name the field
representation, reference phase, comparison target, and scope.
```

The report must include the formulas:

```text
U(x,y) = u(x,y) exp(i Phi_ref(x,y))
u_ZBF = conj(Ex_Zemax)
U_ZBF = conj(Ex_Zemax) exp(i axis_sign Phi_ZBF)
Phi_ZBF = k sign(R) (sqrt(abs(R)^2 + x^2 + y^2) - abs(R))
k = 2 pi n / lambda
```

- [ ] **Step 2: Verify report terminology**

Run:

```powershell
rg -n "账本|ledger|physical phase is solved|全局物理相位.*已" sandbox/diagnostics/s7_s8_phase_audit.md
```

Expected: no matches except if the report explicitly says these terms must not be used as conclusions.

### Task 2: Add Audited Interpretation To The Root-Cause Log

**Files:**
- Modify: `sandbox/diagnostics/s7_s8_phase_root_cause.md`

**Interfaces:**
- Consumes: `sandbox/diagnostics/s7_s8_phase_audit.md`
- Produces: A short top-level audited interpretation that replaces overconfident wording without deleting historical notes.

- [ ] **Step 1: Insert audited note near the top**

Add a section immediately after the title:

```markdown
## Audited interpretation - 2026-07-09

This note uses the historical word "ledger" in many places. For the current
audited interpretation, read it as "field representation and reference phase
convention", not as a formal optics term.

The S7-to-S8 free-space segment is locally consistent when both fields are
compared in the same reference-relative representation and when endpoint
physical fields are reconstructed with the corresponding ZBF reference phase.
The value `0.0319913 waves` is retained as a mixed-reference residual, not as
direct proof of a free-space propagation-kernel error. This does not prove VM1,
S8-to-S12, or whole-path physical phase closure.
```

This section must state:

- S7/S8 is locally consistent only after the propagated residual field is
  reconstructed into a physical total field with the accepted ZBF endpoint
  reference phase formula.
- `0.0319913 waves` is a mixed-reference residual.
- This does not prove VM1, S8-to-S12, or global physical phase closure.
- The detailed Chinese audit is in `sandbox/diagnostics/s7_s8_phase_audit.md`.

- [ ] **Step 2: Keep historical evidence intact**

Do not delete the long historical diagnostic log. The audit section is an interpretive correction, not a rewrite of every old stage entry.

### Task 3: Verify And Commit

**Files:**
- Verify: `sandbox/diagnostics/s7_s8_phase_audit.md`
- Verify: `docs/s7_s8_phase_audit.md`
- Verify: `sandbox/diagnostics/s7_s8_phase_root_cause.md`
- Commit: `docs/superpowers/plans/2026-07-09-s7-s8-phase-audit.md`, `docs/s7_s8_phase_audit.md`

**Interfaces:**
- Consumes: outputs of Task 1 and Task 2.
- Produces: A committed documentation audit.

- [ ] **Step 1: Run focused text verification**

Run:

```powershell
rg -n "0\\.0319913|0\\.000203888|2\\.3866e-17|0\\.000741144|Phi_ZBF|conj\\(Ex_Zemax\\)|mixed-reference|参考相位" docs/s7_s8_phase_audit.md sandbox/diagnostics/s7_s8_phase_audit.md sandbox/diagnostics/s7_s8_phase_root_cause.md
```

Expected: matches in both files showing the key numbers, equations, and mixed-reference wording are present.

- [ ] **Step 2: Inspect git diff**

Run:

```powershell
git diff -- docs/superpowers/plans/2026-07-09-s7-s8-phase-audit.md docs/s7_s8_phase_audit.md
```

Expected: only documentation changes in tracked files; `sandbox/diagnostics/...` is intentionally ignored and checked by file content, not by Git diff.

- [ ] **Step 3: Commit only audit artifacts**

Run:

```powershell
git add -- docs/superpowers/plans/2026-07-09-s7-s8-phase-audit.md docs/s7_s8_phase_audit.md
git commit -m "docs: audit S7 S8 phase interpretation"
```

Expected: one commit containing the audit plan and tracked audit report mirror. The in-place sandbox audit report and root-cause-log interpretation remain in the working tree because `sandbox/` is ignored by repository policy.
