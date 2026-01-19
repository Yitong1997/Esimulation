---
trigger: always_on
---

# 混合光学仿真项目 (Hybrid Optical Simulation) - Agent Steering 规则

本文件定义了混合物理光学-几何光学仿真项目的开发规范和指导原则。
项目核心：使用 PROPER 进行物理光学传输，optiland 进行几何光线追迹计算 OPD。
 
## 项目概述

这是一个类似 Zemax 的混合光学仿真系统，结合：
- **PROPER 库**：物理光学波前传输（衍射传播、Fresnel/Fraunhofer 传播）
- **optiland 库**：几何光线追迹（OPD 计算、像差分析、光学系统建模）

## ⚠️ 重要：库的安装与调用方式

### 安装方式
两个核心库都已通过 pip 安装到 Python 环境中：
- `optiland`：通过 `pip install optiland` 安装
- `proper`（PyPROPER3）：通过 `pip install .` 在 `proper_v3.3.4_python` 目录下安装

### 正确的调用方式
**必须使用 pip 安装的库，而不是直接调用工作区内的源码文件夹！**

```python
# ✅ 正确：从 pip 安装的包导入（注意：optiland 需要从子模块导入类）
import proper
import optiland
from optiland.optic import Optic  # 注意：不是 from optiland import Optic
from optiland.phase import GridPhaseProfile

# ❌ 错误：不要直接引用工作区内的源码路径
# import sys
# sys.path.insert(0, 'optiland-master/optiland')  # 不要这样做！
# sys.path.insert(0, 'proper_v3.3.4_python/proper')  # 不要这样做！

# ❌ 错误：不要从顶层包直接导入类
# from optiland import Optic  # 这样会报错！
```

### 工作区内源码文件夹的用途
- `optiland-master/`：仅作为 API 文档和示例参考，查看 `docs/` 目录了解用法
- `proper_v3.3.4_python/`：仅作为 API 文档参考，查看 `PROPER_manual_v3.3.4.pdf`

### 项目自有代码
项目自有代码位于 `src/` 目录下，测试代码位于 `tests/` 目录下：
```python
# 导入项目自有模块时，需要将 src 添加到路径
import sys
sys.path.insert(0, 'src')

from wavefront_to_rays import WavefrontToRaysSampler
```

## 核心架构原则

### 1. 混合传输模型
- 元器件之间的传输：使用 PROPER 的物理光学传输功能（`prop_propagate`、`prop_lens` 等）
- 元器件处的波前 OPD 计算：使用 optiland 的几何光线追迹
- 两个库之间需要建立统一的波前数据接口

### 2. 数据流设计
```
输入光源 → [PROPER: 初始波前]  → [PROPER: 传输] → [optiland: 元件OPD] → [PROPER: 传输] → ... → 输出
```

### 3. 坐标系统统一
- 确保 PROPER 和 optiland 使用一致的坐标系定义
- 注意两个库的采样网格差异，需要进行适当的插值或重采样

## 代码规范

### Python 风格
- 遵循 PEP 8 代码风格
- 使用类型注解（Type Hints）
- 函数和类需要完整的 docstring（中文说明）
- 变量命名使用英文，注释使用中文

### 模块组织
```
project/
├── core/                    # 核心功能模块
│   ├── wavefront.py         # 波前数据结构和操作
│   ├── propagation.py       # PROPER 传输封装
│   ├── raytracing.py        # optiland 光线追迹封装
│   └── interface.py         # 两个库之间的接口层
├── components/              # 光学元件定义
│   ├── lens.py              # 透镜
│   ├── mirror.py            # 反射镜
│   ├── aperture.py          # 光阑
│   └── surface.py           # 通用光学面
├── analysis/                # 分析功能
│   ├── psf.py               # PSF 计算
│   ├── mtf.py               # MTF 计算
│   ├── aberration.py        # 像差分析
│   └── wavefront_analysis.py # 波前分析
├── utils/                   # 工具函数
│   ├── grid.py              # 网格操作
│   ├── interpolation.py     # 插值工具
│   └── visualization.py     # 可视化
└── tests/                   # 测试代码
```
 

### 光线追迹
- 使用 `raytrace` 模块进行光线追迹
- 支持实际光线和近轴光线
- 可获取各面的光线数据

## 接口层设计要求

### 波前数据转换
```python
class WavefrontInterface:
    """PROPER 和 optiland 之间的波前数据转换接口"""
    
    @staticmethod
    def proper_to_opd(wfo) -> np.ndarray:
        """从 PROPER 波前对象提取 OPD 数据"""
        pass
    
    @staticmethod
    def opd_to_proper(opd: np.ndarray, wfo) -> None:
        """将 OPD 数据应用到 PROPER 波前对象"""
        pass
    
    @staticmethod
    def optiland_opd_to_grid(optic, field, wavelength, grid_size) -> np.ndarray:
        """从 optiland 计算 OPD 并转换为网格数据"""
        pass
```

### 网格重采样
- 当两个库的网格大小不一致时，需要进行插值
- 推荐使用双三次插值（bicubic interpolation）
- 注意边界处理
 

### 计算加速
- 利用 NumPy 向量化操作
- 考虑使用 optiland 的 PyTorch 后端进行 GPU 加速
- FFT 操作可使用 FFTW（通过 pyfftw）
