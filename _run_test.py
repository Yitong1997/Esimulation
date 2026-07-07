import runpy
import sys
import os

print("Running d:/BTS/.worktrees/BSP/sandbox/run_biconic_focus_pop.py indirectly...")
os.chdir(r"d:\BTS\.worktrees\BSP")
sys.path.insert(0, r"d:\BTS\.worktrees\BSP")

try:
    runpy.run_path(r"d:\BTS\.worktrees\BSP\sandbox\run_biconic_focus_pop.py")
    print("Execution completed successfully.")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
