# -*- coding: utf-8 -*-
"""临时诊断脚本"""
import sys
from pathlib import Path
project_root = Path(r'd:\BTS')
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))
sys.path.insert(0, str(project_root / 'proper_v3.3.4_python'))

# 直接导入测试模块的函数
import importlib.util
spec = importlib.util.spec_from_file_location(
    "galileo_test", 
    project_root / 'tests' / 'integration' / '伽利略式离轴抛物面扩束镜传输误差标准测试文件.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# 运行测试
result = module.run_galilean_oap_expander_test(verbose=True)
print('\n\n=== RESULT DICT ===')
for k, v in result.items():
    print(f'{k}: {v}')
