---
trigger: model_decision
description: using optiland to trace light
---


**optiland 已通过 pip 安装，必须从安装的包导入，而不是直接引用工作区内的源码！**

```python
# ✅ 正确：从 pip 安装的包导入（注意：需要从子模块导入）
import optiland
from optiland.optic import Optic  # 注意：不是 from optiland import Optic
from optiland.phase import GridPhaseProfile
from optiland.rays import RealRays
from optiland.wavefront import Wavefront
from optiland.samples.objectives import ReverseTelephoto  # 示例系统

# ✅ 另一种正确的导入方式（使用模块别名）
from optiland import optic
from optiland import wavefront
from optiland import distribution

lens = optic.Optic()  # 通过模块访问类
wf = wavefront.Wavefront(lens, 0, 0, num_rays=64)

# ❌ 错误：不要直接引用工作区内的源码
# sys.path.insert(0, 'optiland-master/optiland')

# ❌ 错误：不要从顶层包直接导入类
# from optiland import Optic  # 这样会报错！
```

工作区内的 `optiland-master/` 文件夹仅作为 API 文档和示例参考：
- 查看 `optiland-master/docs/examples/` 了解使用示例
- 查看 `optiland-master/docs/api/` 了解 API 文档

## ⚠️ 关键警告：方形光瞳采样

**在波前到光线转换过程中，相位场默认为方形光瞳（完整的正方形区域），并在整个区域上采样为光线。**

这意味着：
1. 当使用 `GridPhaseProfile` 创建相位面时，相位数据覆盖整个正方形网格
2. 光线采样会在整个方形区域内进行，而不仅仅是圆形光瞳内
3. 如果需要圆形光瞳，必须显式应用光瞳掩模或使用 `is_stop=True` 设置光阑

## ⚠️ 关键警告：相位面 OPD 单位问题（1000倍放大）

**optiland 的 `GridPhaseProfile` 相位面在计算 OPD 时存在单位不一致问题，导致相位引入的 OPD 被放大了 1000 倍！**

### 问题原因
在 `PhaseInteractionModel` 中：
```python
k0 = 2 * np.pi / wavelength  # wavelength 单位是 μm，所以 k0 单位是 1/μm
opd_shift = -phase / k0      # 结果单位应该是 μm
rays.opd = rays.opd + opd_shift  # 但 rays.opd 单位是 mm！
```

### 影响
- 相位 2π 弧度应该引入 1 波长（约 0.55 μm = 0.00055 mm）的 OPD
- 实际引入了 0.55 mm 的 OPD（放大了 1000 倍）

### 解决方案
在使用相位面计算 OPD 时，需要除以 1000 进行修正：

```python
def get_corrected_opd_waves(rays, chief_ray, wavelength_um):
    """获取修正后的 OPD（波长数）
    
    参数:
        rays: 光线追迹结果
        chief_ray: 主光线（作为参考）
        wavelength_um: 波长（μm）
    
    返回:
        相对于主光线的 OPD（波长数）
    """
    import numpy as np
    
    # 计算相对于主光线的 OPD（单位：mm，但被放大了 1000 倍）
    chief_opd = float(np.asarray(chief_ray.opd).item())
    relative_opd_mm = np.asarray(rays.opd) - chief_opd
    
    # 修正 1000 倍放大问题
    relative_opd_corrected_mm = relative_opd_mm / 1000.0
    
    # 转换为波长数
    wavelength_mm = wavelength_um * 1e-3
    opd_waves = relative_opd_corrected_mm / wavelength_mm
    
    return opd_waves
```

```python
# 方形光瞳采样示例
import numpy as np
from optiland.phase import GridPhaseProfile

# 创建方形网格上的相位数据
n = 64
x = np.linspace(-5, 5, n)  # 方形区域 [-5, 5] mm
y = np.linspace(-5, 5, n)
X, Y = np.meshgrid(x, y)

# 相位数据覆盖整个方形区域
phase = some_phase_function(X, Y)


# 创建相位分布 - 注意这是方形区域！
phase_profile = GridPhaseProfile(
    x_coords=x,
    y_coords=y,
    phase_grid=phase,
)

# 如果需要圆形光瞳，需要手动应用掩模
R = np.sqrt(X**2 + Y**2)
pupil_radius = 5.0
circular_mask = R <= pupil_radius
phase_masked = np.where(circular_mask, phase, 0.0)
```