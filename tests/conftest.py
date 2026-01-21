"""
pytest 配置文件

本文件包含 pytest 的全局配置和 fixtures。
"""

import sys
from pathlib import Path

# 将 src 目录添加到 Python 路径
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
