
import sys

filename = "d:\\BTS\\final_verify_output_utf8.txt"

encodings = ['utf-8', 'gbk', 'latin-1', 'utf-16le']

content = None
for enc in encodings:
    try:
        with open(filename, 'r', encoding=enc) as f:
            content = f.readlines()
        print(f"Successfully read with {enc}")
        break
    except UnicodeDecodeError:
        continue


if content:
    lines = content
    print(f"File has {len(lines)} lines.")
    print("--- Last 50 lines ---")
    for i in range(max(0, len(lines)-50), len(lines)):
        print(lines[i].strip())
    
    print("\n--- Searching for RMS ---")
    for i, line in enumerate(lines):
        if "RMS" in line:
             print(f"\n--- Found RMS at line {i} ---")
             print(line.strip())
             if i+1 < len(lines): print(lines[i+1].strip())

    print("\n--- Searching for Surface 9 ---")
    for i, line in enumerate(lines):
        if "Surface 9" in line:
             print(f"\n--- Found Surface 9 at line {i} ---")
             # Print surrounding lines
             for j in range(max(0, i-5), min(len(lines), i+30)):
                 print(f"{lines[j].strip()}")
        if "Zernike" in line and ("Surface 9" in lines[max(0, i-10)] or "rad" in line):
             # Ensure it's likely relevant
             pass
else:
    print("Failed to read file with any encoding.")
