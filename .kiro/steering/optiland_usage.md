<!------------------------------------------------------------------------------------
# optiland 几何光学库使用规范

本文件定义了 optiland 库在混合光学仿真项目中的使用规范。
inclusion: fileMatch
fileMatchPattern: '**/raytracing/**,**/optiland/**,**/*raytrac*,**/*opd*'
-------------------------------------------------------------------------------------> 

## ⚠️ 重要：库的调用方式

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

## optiland 核心概念

### Optic 对象
optiland 使用 `Optic` 类表示完整的光学系统：
```python
from optiland.optic import Optic  # 注意：从子模块导入

lens = Optic()
```

### 表面定义
光学系统由一系列表面组成，每个表面有：
- 曲率半径
- 厚度（到下一表面的距离）
- 材料
- 半口径

## 常用操作参考

### 创建光学系统
```python
from optiland.optic import Optic  # 注意：从子模块导入

# 创建空系统
lens = Optic()

# 设置孔径
lens.set_aperture(aperture_type='EPD', value=25.0)  # 入瞳直径 25mm

# 设置视场
lens.set_field_type('angle')
lens.add_field(y=0)      # 轴上视场
lens.add_field(y=10)     # 10度视场
lens.add_field(y=14)     # 14度视场

# 设置波长
lens.add_wavelength(0.55)   # 550nm
lens.add_wavelength(0.486)  # F线
lens.add_wavelength(0.656)  # C线
lens.set_primary_wavelength(0.55)
```

### 添加表面
```python
# 首先添加物面（index=0）- 这是必须的！
lens.add_surface(index=0, radius=np.inf, thickness=np.inf)

# 添加球面（index=1）
lens.add_surface(
    index=1,
    radius=100.0,      # 曲率半径 (mm)
    thickness=10.0,    # 厚度 (mm)
    material='N-BK7',  # 材料
    is_stop=True       # 是否为光阑
)

# 添加非球面（index=2）
lens.add_surface(
    index=2,
    radius=50.0,
    thickness=5.0,
    conic=-1.0,        # 圆锥常数
    material='air'
)

# 添加像面（最后一个表面）
lens.add_surface(index=3)
```

### 光线追迹
```python
# 追迹单条光线
ray = lens.trace_ray(Hx=0, Hy=0.7, Px=0, Py=1.0, wavelength=0.55)

# 追迹光线束
rays = lens.trace_rays(num_rays=100, field_index=0, wavelength_index=0)
```

### OPD 计算（详细示例）

OPD（光程差）计算是混合仿真的核心功能之一。以下是详细的使用方法：

```python
from optiland import wavefront
from optiland.samples.objectives import ReverseTelephoto
import numpy as np
import matplotlib.pyplot as plt

# 加载示例光学系统
lens = ReverseTelephoto()

# 创建波前对象
# 参数说明：
#   - optic: 光学系统对象
#   - field_index: 视场索引（0 表示第一个视场）
#   - wavelength_index: 波长索引（0 表示第一个波长）
#   - num_rays: 采样光线数量（决定 OPD 网格分辨率）
wf = wavefront.Wavefront(lens, field_index=0, wavelength_index=0, num_rays=64)

# 获取 OPD 数据
opd = wf.opd  # 返回 2D numpy 数组，单位：波长数

# 获取 OPD 统计信息
print(f"OPD RMS: {np.nanstd(opd):.4f} waves")
print(f"OPD PV: {np.nanmax(opd) - np.nanmin(opd):.4f} waves")

# 可视化 OPD
wf.view()  # 内置可视化方法

# 或者手动绘制
plt.figure()
plt.imshow(opd, cmap='RdBu_r', origin='lower')
plt.colorbar(label='OPD (waves)')
plt.title('Optical Path Difference')
plt.show()
```

### Zernike 分解（详细示例）

Zernike 多项式分解用于分析波前像差：

```python
from optiland import wavefront
from optiland.samples.objectives import ReverseTelephoto
import numpy as np

# 加载光学系统
lens = ReverseTelephoto()

# 创建波前对象
wf = wavefront.Wavefront(lens, field_index=0, wavelength_index=0, num_rays=64)

# Zernike 分解
# num_terms: Zernike 项数（标准 Zernike 多项式）
zernike_coeffs = wf.zernike_fit(num_terms=37)

# 查看 Zernike 系数
print("Zernike 系数（单位：波长数）:")
for i, coeff in enumerate(zernike_coeffs):
    if abs(coeff) > 0.001:  # 只显示显著的项
        print(f"  Z{i+1}: {coeff:.6f}")

# 常用 Zernike 项对应关系（Noll 索引）：
# Z1: Piston（活塞）
# Z2: Tilt X（X 方向倾斜）
# Z3: Tilt Y（Y 方向倾斜）
# Z4: Defocus（离焦）
# Z5: Astigmatism 45°（45° 像散）
# Z6: Astigmatism 0°（0° 像散）
# Z7: Coma X（X 方向彗差）
# Z8: Coma Y（Y 方向彗差）
# Z9: Trefoil X
# Z10: Trefoil Y
# Z11: Spherical（球差）

# 从 Zernike 系数重建波前
reconstructed_opd = wf.zernike_to_opd(zernike_coeffs)
```

### 像差分析
```python
from optiland.aberrations import Aberrations

# 计算 Seidel 像差
ab = Aberrations(lens)
seidel = ab.seidel()

# 获取各项像差
spherical = seidel['spherical']
coma = seidel['coma']
astigmatism = seidel['astigmatism']
```

### 使用示例光学系统

optiland 提供了多种预定义的示例光学系统：

```python
# 物镜系统
from optiland.samples.objectives import (
    ReverseTelephoto,    # 反远距镜头
    CookeTriplet,        # Cooke 三片式
    DoubleGauss,         # 双高斯
    Petzval,             # Petzval 镜头
)

# 目镜系统
from optiland.samples.eyepieces import (
    EyepieceErfle,       # Erfle 目镜
    EyepieceKellner,     # Kellner 目镜
)

# 望远镜系统
from optiland.samples.telescopes import (
    Cassegrain,          # 卡塞格林望远镜
    RitcheyChretien,     # RC 望远镜
)

# 使用示例
lens = ReverseTelephoto()
print(lens)  # 打印系统信息
lens.draw()  # 绘制光学系统布局
```

### 光线分布类型

optiland 支持多种光线分布类型：

```python
from optiland import distribution

# 可用的分布类型：
# - 'hexapolar': 六角极坐标分布（默认，推荐用于圆形光瞳）
# - 'rectangular': 矩形网格分布
# - 'random': 随机分布
# - 'cross': 十字分布

# 创建分布
dist = distribution.create_distribution(
    distribution_type='hexapolar',
    num_points=100,
)

# 获取分布点坐标
px, py = dist.get_points()  # 归一化光瞳坐标 [-1, 1]
```

## 与 PROPER 接口的注意事项

### ⚠️ 方形光瞳与圆形光瞳的转换

在混合仿真中，需要特别注意光瞳形状的处理：

```python
import numpy as np
from optiland.wavefront import Wavefront

def get_opd_with_circular_mask(optic, field_index, wavelength_index, grid_size):
    """
    从 optiland 获取 OPD 网格数据，并应用圆形光瞳掩模
    
    注意：optiland 的 Wavefront 默认在方形网格上计算 OPD，
    但实际光瞳通常是圆形的。此函数确保只返回圆形光瞳内的有效数据。
    
    参数:
        optic: optiland Optic 对象
        field_index: 视场索引
        wavelength_index: 波长索引
        grid_size: 输出网格大小
    
    返回:
        opd_grid: OPD 网格数据 (波长数)，圆形光瞳外为 NaN
        pupil_mask: 圆形光瞳掩模（布尔数组）
    """
    # 计算波前
    wf = Wavefront(optic, field_index, wavelength_index, num_rays=grid_size)
    
    # 获取 OPD 数据
    opd_grid = wf.opd
    
    # 创建圆形光瞳掩模
    n = opd_grid.shape[0]
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # 圆形光瞳掩模
    circular_mask = R <= 1.0
    
    # 应用掩模
    opd_grid_masked = np.where(circular_mask, opd_grid, np.nan)
    
    return opd_grid_masked, circular_mask
```

### OPD 数据提取
```python
def get_opd_grid(optic, field_index, wavelength_index, grid_size):
    """
    从 optiland 获取 OPD 网格数据，用于 PROPER
    
    参数:
        optic: optiland Optic 对象
        field_index: 视场索引
        wavelength_index: 波长索引
        grid_size: 输出网格大小
    
    返回:
        opd_grid: OPD 网格数据 (波长数)
        pupil_mask: 光瞳掩模
    """
    from optiland.wavefront import Wavefront
    import numpy as np
    
    # 计算波前
    wf = Wavefront(optic, field_index, wavelength_index, num_rays=grid_size)
    
    # 获取 OPD 数据
    opd_grid = wf.opd
    
    # 创建光瞳掩模
    pupil_mask = ~np.isnan(opd_grid)
    opd_grid = np.nan_to_num(opd_grid, nan=0.0)
    
    return opd_grid, pupil_mask
```

### 坐标系转换
```python
# optiland 使用归一化光瞳坐标 [-1, 1]
# PROPER 使用物理坐标 (米)

def optiland_to_proper_coords(normalized_coords, pupil_radius_m):
    """将 optiland 归一化坐标转换为 PROPER 物理坐标"""
    return normalized_coords * pupil_radius_m
```

### 单位转换
```python
# optiland 默认单位：mm（长度），波长数（OPD）
# PROPER 默认单位：m（长度），弧度（相位）

def opd_waves_to_phase(opd_waves):
    """OPD（波长数）转相位（弧度）"""
    return 2 * np.pi * opd_waves

def mm_to_m(value_mm):
    """毫米转米"""
    return value_mm * 1e-3
```

## 高级功能

### 使用 PyTorch 后端
```python
from optiland.backend import set_backend

# 切换到 PyTorch 后端（支持 GPU 和自动微分）
set_backend('torch')

# 切换回 NumPy 后端
set_backend('numpy')
```

### 自定义表面类型
```python
from optiland.surfaces import Surface

class CustomSurface(Surface):
    """自定义表面类型"""
    
    def sag(self, x, y):
        """计算表面矢高"""
        # 实现自定义矢高计算
        pass
    
    def normal(self, x, y):
        """计算表面法向量"""
        # 实现自定义法向量计算
        pass
```

### 从文件加载系统
```python
# 从 JSON 文件加载
lens = Optic.from_file('system.json')

# 从 Zemax 文件加载
lens = Optic.from_zemax('system.zmx')

# 保存到文件
lens.save('system.json')
```

## 性能优化

### 批量光线追迹
```python
# 使用向量化操作进行批量追迹
rays = lens.trace_rays(num_rays=10000, ...)

# 使用 PyTorch 后端进行 GPU 加速
set_backend('torch')
rays = lens.trace_rays(num_rays=100000, ...)
```

### 缓存机制
```python
# optiland 内部有缓存机制
# 重复计算相同配置时会自动使用缓存
# 修改系统参数后缓存会自动失效
```

## 常见问题

### 光线追迹失败
- 症状：部分光线返回 NaN
- 原因：光线未能到达像面（全反射、渐晕等）
- 解决：检查系统设计，调整光阑大小

### OPD 计算不准确
- 症状：OPD 值异常大或有跳变
- 原因：参考球面选择不当
- 解决：检查参考球面半径设置

### 材料未找到
- 症状：材料数据库查询失败
- 解决：检查材料名称拼写，或使用自定义材料定义

### 方形光瞳导致的边缘效应
- 症状：在波前转光线时，方形区域边角处出现异常光线
- 原因：相位数据在方形网格上定义，但实际光瞳是圆形
- 解决：
  1. 在创建相位面前应用圆形掩模
  2. 使用 `is_stop=True` 设置光阑限制光线范围
  3. 在后处理时过滤掉圆形光瞳外的光线

```python
# 过滤圆形光瞳外的光线示例
def filter_rays_in_circular_pupil(rays, pupil_radius):
    """过滤掉圆形光瞳外的光线"""
    import numpy as np
    
    x = np.asarray(rays.x)
    y = np.asarray(rays.y)
    r = np.sqrt(x**2 + y**2)
    
    # 创建掩模
    mask = r <= pupil_radius
    
    # 返回有效光线的索引
    return np.where(mask)[0]
```

## 完整示例：从光学系统到波前分析

```python
"""
完整示例：创建光学系统并进行波前分析
"""
import numpy as np
import matplotlib.pyplot as plt
from optiland.optic import Optic
from optiland import wavefront

# 1. 创建光学系统（简单单透镜）
lens = Optic()

# 设置系统参数
lens.set_aperture(aperture_type='EPD', value=25.0)  # 入瞳直径 25mm
lens.set_field_type(field_type='angle')
lens.add_field(y=0)      # 轴上视场
lens.add_field(y=5)      # 5度视场
lens.add_wavelength(value=0.55, is_primary=True)  # 550nm

# 添加表面
lens.add_surface(index=0, radius=np.inf, thickness=np.inf)  # 物面
lens.add_surface(index=1, radius=100.0, thickness=10.0, 
                 material='N-BK7', is_stop=True)  # 前表面
lens.add_surface(index=2, radius=-100.0, thickness=80.0)  # 后表面
lens.add_surface(index=3)  # 像面

# 2. 计算波前
wf = wavefront.Wavefront(lens, field_index=0, wavelength_index=0, num_rays=64)

# 3. 获取 OPD
opd = wf.opd
print(f"OPD RMS: {np.nanstd(opd):.4f} waves")

# 4. Zernike 分解
zernike_coeffs = wf.zernike_fit(num_terms=37)
print(f"球差 (Z11): {zernike_coeffs[10]:.6f} waves")

# 5. 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# OPD 图
im = axes[0].imshow(opd, cmap='RdBu_r', origin='lower')
axes[0].set_title('OPD (waves)')
plt.colorbar(im, ax=axes[0])

# Zernike 系数柱状图
axes[1].bar(range(1, 12), zernike_coeffs[:11])
axes[1].set_xlabel('Zernike Term')
axes[1].set_ylabel('Coefficient (waves)')
axes[1].set_title('Zernike Coefficients')

plt.tight_layout()
plt.show()
```
