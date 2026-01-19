<!------------------------------------------------------------------------------------
# 混合光学仿真项目 (Hybrid Optical Simulation) - Agent Steering 规则

本文件定义了混合物理光学-几何光学仿真项目的开发规范和指导原则。
项目核心：使用 PROPER 进行物理光学传输，optiland 进行几何光线追迹计算 OPD。
-------------------------------------------------------------------------------------> 

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
输入光源 → [PROPER: 初始波前] → [optiland: 元件OPD] → [PROPER: 传输] → ... → 输出
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

## PROPER 使用规范

### 波前初始化
```python
# 使用 prop_begin 初始化波前
wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_ratio)
```

### 传输操作
- `prop_propagate`: 自动选择合适的传播算法
- `prop_lens`: 添加理想透镜相位
- `prop_circular_aperture`: 圆形光阑
- `prop_zernikes`: 添加 Zernike 像差

### 注意事项
- PROPER 使用复数波前表示（振幅 + 相位）
- 网格采样需要满足 Nyquist 准则，相邻像素间相位差不应大于pi
- 注意 `prop_end` 的调用以获取最终结果

## optiland 使用规范

### 光学系统定义
```python
from optiland.optic import Optic  # 注意：从子模块导入

lens = Optic()
lens.add_surface(index=0, ...)  # 添加光学面
lens.set_aperture(...)          # 设置光阑
lens.set_field_type(...)        # 设置视场类型
```

### OPD 计算
- 使用 `wavefront` 模块进行 OPD 计算
- 支持 Zernike 分解
- 可获取各视场点的波前数据

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

## 测试规范

### 单元测试
- 每个模块都需要对应的测试文件
- 使用 pytest 框架
- 测试覆盖率目标 > 80%

### 验证测试
- 与 Zemax 或其他商业软件的结果进行对比验证
- 建立标准测试用例库（如 Cooke Triplet、Double Gauss 等）

### 物理一致性测试
- 验证能量守恒
- 验证相位连续性
- 验证几何光学极限下的一致性

## 性能优化指南

### 内存管理
- 大型波前数组使用 `np.float32` 或 `np.complex64` 节省内存
- 及时释放不再使用的大数组
- 考虑使用内存映射文件处理超大数据

### 计算加速
- 利用 NumPy 向量化操作
- 考虑使用 optiland 的 PyTorch 后端进行 GPU 加速
- FFT 操作可使用 FFTW（通过 pyfftw）

## 文档要求

### 代码文档
- 所有公共 API 需要完整的 docstring
- 包含参数说明、返回值说明、使用示例
- 复杂算法需要添加数学公式说明

### 用户文档
- 提供快速入门指南
- 提供详细的 API 参考
- 提供典型应用案例

## 错误处理

### 异常类型
```python
class OpticalSimulationError(Exception):
    """光学仿真基础异常"""
    pass

class WavefrontError(OpticalSimulationError):
    """波前相关错误"""
    pass

class PropagationError(OpticalSimulationError):
    """传输相关错误"""
    pass

class RayTracingError(OpticalSimulationError):
    """光线追迹相关错误"""
    pass
```

### 错误信息
- 使用中文错误信息
- 提供足够的上下文信息
- 建议可能的解决方案

## 常用物理常量和单位

### 单位约定
- 长度：默认使用毫米（mm），微米（μm）用于波长
- 角度：默认使用弧度（rad），度（°）用于用户接口
- 波长：微米（μm）



## 相关 Steering 文件

本项目包含以下 steering 文件，请根据具体任务参考：

1. **optical_system_conventions.md** - 光学系统结构定义与坐标约定（全局生效）
   - 坐标系统定义（全局坐标系、局部坐标系）
   - 符号正负号约定（曲率半径、厚度、旋转角度）
   - 倾斜和离轴系统定义方式
   - 元器件入射面与出射面定义

2. **proper_usage.md** - PROPER 物理光学库使用规范（条件触发）
   - 波前初始化和传播操作
   - 与 optiland 接口的相位单位转换

3. **optiland_usage.md** - optiland 几何光学库使用规范（条件触发）
   - 光学系统定义和光线追迹
   - OPD 计算和坐标转换

4. **testing_standards.md** - 测试规范（条件触发）
   - 单元测试、属性基测试、验证测试规范
