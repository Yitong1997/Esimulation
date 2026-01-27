
import re

try:
    with open(r'd:\BTS\debug_output_ver5.txt', 'r', encoding='utf-16') as f:
        content = f.read()
except UnicodeError:
    try:
        with open(r'd:\BTS\debug_output.txt', 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Read error: {e}")
        exit(1)

# Print lines containing [DEBUG] and surrounding lines
lines = content.splitlines()
for i, line in enumerate(lines):
    if "[DEBUG]" in line or "Absolute OPD" in line:
        print(f"LINE {i}: {line.strip()}")
