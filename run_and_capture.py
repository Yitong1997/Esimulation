
import subprocess
import sys

def run_test():
    cmd = [sys.executable, "d:/BTS/tests/integration/伽利略式离轴抛物面扩束镜传输误差标准测试文件.py"]
    print(f"Running command: {cmd}")
    
    with open("d:/BTS/captured_output.txt", "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
            
        process.wait()
        print(f"\nProcess finished with exit code {process.returncode}")

if __name__ == "__main__":
    run_test()
