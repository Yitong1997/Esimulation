
import os

filepath = r"d:\BTS\debug_s8_output.txt"
try:
    with open(filepath, 'r', encoding='utf-16') as f:
        lines = f.readlines()
except:
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file: {e}")
        lines = []







print("Extracting Surface 9 Detailed Debug Info...")
with open("extracted_s9_info.txt", "w", encoding="utf-8") as outfile:
    found_s9 = False
    for i, line in enumerate(lines):
        if "Surface 9" in line and "Debug Info" in line:
            found_s9 = True
            msg = f"Found Surface 9 Debug Info at line {i}:\n"
            print(msg)
            outfile.write(msg)
            # Print next 200 lines to capture matrix, normal, vectors etc.
            for j in range(i, min(i+200, len(lines))):
                outfile.write(lines[j])
            break
