import os
import shutil

src_dir = r"d:\BTS\.worktrees\residual-phase"
dst_dir = r"d:\BTS\.worktrees\BSP"

files_to_sync = [
    "pop/__init__.py",
    "pop/analysis.py",
    "pop/core.py",
    "pop/options.py",
    "pop/propagation/element.py",
    "pop/propagation/free_space.py",
    "pop/source.py",
    "pop/visualization.py",
    "pop/wavefront/sampler.py",
    "tests/test_residual_phase_modes.py",
    "tests/test_residual_phase_options.py",
    "tests/test_visualization_phase_residual.py",
    "tests/test_zbf_pop_comparison.py",
    "docs/plans/2026-05-13-zbf-pop-comparison-design.md",
    "docs/plans/2026-05-14-s4-residual-phase-diagnostics.md",
    "sandbox/run_biconic_focus_expand_pop.py",
    "sandbox/run_biconic_focus_pop.py",
    "sandbox/_gen_run_biconic_focus_pop.py"
]

print(f"Starting sync from {src_dir} to {dst_dir}")
success_count = 0
for rel_path in files_to_sync:
    src_file = os.path.join(src_dir, rel_path)
    dst_file = os.path.join(dst_dir, rel_path)
    
    if os.path.exists(src_file):
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy2(src_file, dst_file)
        print(f"Copied: {rel_path}")
        success_count += 1
    else:
        print(f"WARNING: Source file not found: {rel_path}")

print(f"Successfully copied {success_count} files.")
