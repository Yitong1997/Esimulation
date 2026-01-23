# BTS API 文档

## 概述

BTS（Beam Tracing System）是一个混合光学仿真系统，采用 MATLAB 风格的直观 API 设计。它结合了：

- **PROPER**：物理光学波前传播
- **optiland**：几何光线追迹和 OPD 计算

### 设计理念

BTS API 采用"代码块"结构，将复杂逻辑封装在模块内部，主程序只需要简单的函数调用：

1. **简洁性**：主程序代码极简，功能分块直观
2. **完整性**：API 功能强大，足以支撑所有自定义测试
3. **稳定性**：API 参数完整且稳定，不频繁变动

### 主要功能

- 从 ZMX 文件加载光学系统
- 手动定义光学元件（平面镜、球面镜、薄透镜等）
- 定义高斯光源参数
- 执行混合光学仿真
- 可视化和保存仿真结果

---

## 快速入门

### 安装

确保项目的 `src` 目录在 Python 路径中：

```python
import sys
sys.path.insert(0, 'path/to/project/src')
```

### 最简示例

```python
import bts

# 定义光学系统
system = bts.OpticalSystem("My System")
system.add_flat_mirror(z=50, tilt_x=45)  # 45° 折叠镜

# 定义光源
source = bts.GaussianSource(wavelength_um=0.633, w0_mm=5.0)

# 执行仿真
result = bts.simulate(system, source)

# 查看结果
result.summary()
```

### 主程序模板

```python
"""
混合光学仿真主程序模板
"""

# ============================================================
# 1. 导入与初始化
# ============================================================
import bts

# ============================================================
# 2. 定义光学系统（两种方式二选一）
# ============================================================
# 方式 A：从 ZMX 文件导入
system = bts.load_zmx("path/to/system.zmx")

# 方式 B：逐行定义元件
# system = bts.OpticalSystem("My System")
# system.add_flat_mirror(z=50, tilt_x=45, semi_aperture=30)
# system.add_spherical_mirror(z=150, radius=200, semi_aperture=25)

# ============================================================
# 3. 定义光源
# ============================================================
source = bts.GaussianSource(
    wavelength_um=0.633,    # He-Ne 激光波长
    w0_mm=5.0,              # 束腰半径
    grid_size=256,          # 网格大小
)

# ============================================================
# 4. 系统信息展示（仿真前）
# ============================================================
system.print_info()           # 打印系统参数
system.plot_layout()          # 绘制光路图

# ============================================================
# 5. 执行仿真
# ============================================================
result = bts.simulate(system, source)

# ============================================================
# 6. 结果展示与保存
# ============================================================
result.summary()              # 打印结果摘要
result.plot_all()             # 绘制所有结果图
result.save("output/")        # 保存结果
```

---

## 核心函数

### `load_zmx(path)`

从 ZMX 文件加载光学系统。

**签名：**
```python
def load_zmx(path: str) -> OpticalSystem
```

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `path` | `str` | ZMX 文件路径（绝对路径或相对路径） |

**返回值：**
- `OpticalSystem` 对象，包含从 ZMX 文件加载的所有表面定义

**异常：**
- `FileNotFoundError`：文件不存在
- `ParseError`：解析错误（ZMX 文件格式无效或包含不支持的特性）

**示例：**
```python
import bts

# 加载 ZMX 文件
system = bts.load_zmx("path/to/system.zmx")

# 查看系统信息
system.print_info()

# 绘制光路图
system.plot_layout()
```

---

### `simulate(system, source, verbose=True, num_rays=200)`

执行混合光学仿真。

**签名：**
```python
def simulate(
    system: OpticalSystem,
    source: GaussianSource,
    verbose: bool = True,
    num_rays: int = 200,
) -> SimulationResult
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `system` | `OpticalSystem` | - | 光学系统定义 |
| `source` | `GaussianSource` | - | 高斯光源定义 |
| `verbose` | `bool` | `True` | 是否输出详细信息 |
| `num_rays` | `int` | `200` | 光线追迹使用的光线数量 |

**返回值：**
- `SimulationResult` 对象，包含所有表面的波前数据

**异常：**
- `ConfigurationError`：配置不完整（如空系统）
- `SimulationError`：仿真执行失败
- `ValueError`：参数值无效（如负波长）

**示例：**
```python
import bts

system = bts.load_zmx("system.zmx")
source = bts.GaussianSource(wavelength_um=0.633, w0_mm=5.0)

# 执行仿真
result = bts.simulate(system, source)

# 静默模式
result = bts.simulate(system, source, verbose=False)

# 增加光线数量以提高精度
result = bts.simulate(system, source, num_rays=500)
```

---

## 核心类

### `OpticalSystem`

光学系统定义类，支持两种构建方式：
1. 从 ZMX 文件加载：`bts.load_zmx("system.zmx")`
2. 逐行定义元件：`system.add_surface(...)`

#### 构造函数

```python
def __init__(self, name: str = "Unnamed System") -> None
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | `"Unnamed System"` | 系统名称 |

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 系统名称 |
| `num_surfaces` | `int` | 表面数量 |


#### 方法

##### `add_surface(z, radius=inf, conic=0.0, semi_aperture=25.0, is_mirror=False, tilt_x=0.0, tilt_y=0.0, material="")`

添加通用光学表面（支持链式调用）。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `z` | `float` | - | Z 位置 (mm) |
| `radius` | `float` | `inf` | 曲率半径 (mm)，`inf` 表示平面 |
| `conic` | `float` | `0.0` | 圆锥常数（0=球面，-1=抛物面） |
| `semi_aperture` | `float` | `25.0` | 半口径 (mm) |
| `is_mirror` | `bool` | `False` | 是否为反射镜 |
| `tilt_x` | `float` | `0.0` | 绕 X 轴旋转角度（度） |
| `tilt_y` | `float` | `0.0` | 绕 Y 轴旋转角度（度） |
| `material` | `str` | `""` | 材料名称（空字符串表示空气） |

**返回值：**
- `self`（支持链式调用）

**示例：**
```python
system = bts.OpticalSystem()
system.add_surface(z=100, radius=200, is_mirror=True, tilt_x=45)
```

---

##### `add_flat_mirror(z, tilt_x=0.0, tilt_y=0.0, semi_aperture=25.0)`

添加平面反射镜（支持链式调用）。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `z` | `float` | - | Z 位置 (mm) |
| `tilt_x` | `float` | `0.0` | 绕 X 轴旋转角度（度） |
| `tilt_y` | `float` | `0.0` | 绕 Y 轴旋转角度（度） |
| `semi_aperture` | `float` | `25.0` | 半口径 (mm) |

**返回值：**
- `self`（支持链式调用）

**示例：**
```python
system = bts.OpticalSystem()
system.add_flat_mirror(z=50, tilt_x=45)  # 45° 折叠镜
```

---

##### `add_spherical_mirror(z, radius, tilt_x=0.0, tilt_y=0.0, semi_aperture=25.0)`

添加球面反射镜（支持链式调用）。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `z` | `float` | - | Z 位置 (mm) |
| `radius` | `float` | - | 曲率半径 (mm)，正值为凹面镜 |
| `tilt_x` | `float` | `0.0` | 绕 X 轴旋转角度（度） |
| `tilt_y` | `float` | `0.0` | 绕 Y 轴旋转角度（度） |
| `semi_aperture` | `float` | `25.0` | 半口径 (mm) |

**返回值：**
- `self`（支持链式调用）

**示例：**
```python
system = bts.OpticalSystem()
system.add_spherical_mirror(z=100, radius=200)  # 凹面镜，f=100mm
```

---

##### `add_paraxial_lens(z, focal_length, semi_aperture=25.0)`

添加薄透镜（支持链式调用）。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `z` | `float` | - | Z 位置 (mm) |
| `focal_length` | `float` | - | 焦距 (mm) |
| `semi_aperture` | `float` | `25.0` | 半口径 (mm) |

**返回值：**
- `self`（支持链式调用）

**示例：**
```python
system = bts.OpticalSystem()
system.add_paraxial_lens(z=50, focal_length=100)  # f=100mm 薄透镜
```

---

##### `print_info()`

打印系统参数摘要。

显示系统名称、表面数量，以及每个表面的详细参数。

**示例：**
```python
system = bts.OpticalSystem("My System")
system.add_flat_mirror(z=50, tilt_x=45)
system.print_info()
```

**输出示例：**
```
============================================================
光学系统: My System
表面数量: 1
============================================================

表面 0: standard
  位置: z = 50.000 mm
  曲率半径: 无穷大 (平面)
  反射镜: 是
  倾斜: tilt_x = 45.00°, tilt_y = 0.00°
```

---

##### `plot_layout(projection="YZ", num_rays=5, save_path=None, show=True)`

绘制光路图。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `projection` | `str` | `"YZ"` | 投影平面（`'YZ'`、`'XZ'`、`'XY'`） |
| `num_rays` | `int` | `5` | 光线数量 |
| `save_path` | `str` | `None` | 保存路径（可选） |
| `show` | `bool` | `True` | 是否显示图形 |

**返回值：**
- `(fig, ax)` 元组，matplotlib Figure 和 Axes 对象

**示例：**
```python
system = bts.load_zmx("system.zmx")

# 显示光路图
fig, ax = system.plot_layout(projection='YZ', num_rays=5)

# 保存到文件
system.plot_layout(save_path="layout.png", show=False)
```

---

### `GaussianSource`

高斯光源定义类，用于定义入射高斯光束的参数。

#### 构造函数

```python
def __init__(
    self,
    wavelength_um: float,
    w0_mm: float,
    grid_size: int = 256,
    physical_size_mm: Optional[float] = None,
    z0_mm: float = 0.0,
    beam_diam_fraction: Optional[float] = None,
) -> None
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `wavelength_um` | `float` | - | 波长 (μm)，必须为正数 |
| `w0_mm` | `float` | - | 束腰半径 (mm)，必须为正数 |
| `grid_size` | `int` | `256` | 网格大小，必须为正整数 |
| `physical_size_mm` | `float` | `None` | 物理尺寸 (mm)，默认 8 倍束腰 |
| `z0_mm` | `float` | `0.0` | 束腰位置 (mm) |
| `beam_diam_fraction` | `float` | `None` | PROPER beam_diam_fraction 参数（可选） |

**异常：**
- `ValueError`：参数值无效

**示例：**
```python
# 基本用法
source = bts.GaussianSource(wavelength_um=0.633, w0_mm=5.0)

# 指定所有参数
source = bts.GaussianSource(
    wavelength_um=1.064,
    w0_mm=10.0,
    grid_size=512,
    physical_size_mm=100.0,
    z0_mm=0.0,
    beam_diam_fraction=0.5,
)
```

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `wavelength_um` | `float` | 波长 (μm) |
| `w0_mm` | `float` | 束腰半径 (mm) |
| `grid_size` | `int` | 网格大小 |
| `physical_size_mm` | `float` | 物理尺寸 (mm) |
| `z0_mm` | `float` | 束腰位置 (mm) |
| `beam_diam_fraction` | `float` | PROPER beam_diam_fraction 参数 |
| `z_rayleigh_mm` | `float` | 瑞利距离 (mm)，计算公式：z_R = π × w0² / λ |
| `wavelength_mm` | `float` | 波长 (mm)，便于内部计算使用 |

#### 方法

##### `print_info()`

打印光源参数。

**示例：**
```python
source = bts.GaussianSource(wavelength_um=0.633, w0_mm=5.0)
source.print_info()
```

**输出示例：**
```
╔══════════════════════════════════════════════════════════════╗
║                      高斯光源参数                              ║
╠══════════════════════════════════════════════════════════════╣
║  波长:           0.633 μm                                     ║
║  束腰半径:       5.000 mm                                     ║
║  瑞利距离:       124140.05 mm                                 ║
║  网格大小:       256 × 256                                    ║
║  物理尺寸:       40.000 mm × 40.000 mm                        ║
║  束腰位置:       0.000 mm                                     ║
╚══════════════════════════════════════════════════════════════╝
```


---

### `SimulationResult`

仿真结果类，存储完整仿真过程的所有结果，提供便捷的数据访问和可视化接口。

**注意：** 此类从 `hybrid_simulation` 模块重导出，通过 `bts.SimulationResult` 访问。

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `success` | `bool` | 仿真是否成功 |
| `error_message` | `str` | 错误信息（如果失败） |
| `config` | `SimulationConfig` | 仿真配置 |
| `source_params` | `SourceParams` | 光源参数 |
| `surfaces` | `List[SurfaceRecord]` | 表面记录列表 |
| `total_path_length` | `float` | 总光程 (mm) |


#### 方法

##### `summary()`

打印仿真摘要，包括状态、波长、网格大小、表面数量和各表面的残差信息。

**示例：**
```python
result = bts.simulate(system, source)
result.summary()
```

**输出示例：**
```
============================================================
混合光学仿真结果摘要
============================================================
状态: 成功
波长: 0.633 μm
网格: 256 × 256
表面数量: 1
总光程: 100.00 mm
------------------------------------------------------------
  [0] Surface_0: standard
       出射相位残差: RMS=0.000123 waves, PV=0.0012 waves
============================================================
```

---

##### `get_surface(index_or_name)`

通过索引或名称获取表面记录。

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `index_or_name` | `int` 或 `str` | 表面索引或名称 |

**返回值：**
- `SurfaceRecord` 对象

**异常：**
- `KeyError`：未找到指定表面

---

##### `get_final_wavefront()`

获取最终表面的出射波前数据。

**返回值：**
- `WavefrontData` 对象

**异常：**
- `ValueError`：没有表面或最终表面没有出射波前数据

**示例：**
```python
final_wf = result.get_final_wavefront()
rms = final_wf.get_residual_rms_waves()
print(f"残差 RMS: {rms*1000:.3f} milli-waves")
```

---

##### `get_entrance_wavefront(surface_index)`

获取指定表面的入射波前数据。

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `surface_index` | `int` | 表面索引 |

**返回值：**
- `WavefrontData` 对象

---

##### `get_exit_wavefront(surface_index)`

获取指定表面的出射波前数据。

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `surface_index` | `int` | 表面索引 |

**返回值：**
- `WavefrontData` 对象


---

##### `plot_all(save_path=None, show=True)`

绘制所有表面的振幅和相位（2D）。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_path` | `str` | `None` | 保存路径（可选） |
| `show` | `bool` | `True` | 是否显示图表 |

---

##### `plot_all_extended(save_path=None, show=True)`

绘制所有表面的扩展概览图（2D），包含振幅、残差相位、Pilot Beam 振幅、振幅残差。

---

##### `plot_surface(index, save_path=None, show=True)`

绘制指定表面的详细图表（2D）。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `index` | `int` | - | 表面索引 |
| `save_path` | `str` | `None` | 保存路径（可选） |
| `show` | `bool` | `True` | 是否显示图表 |

---

##### `plot_surface_3d(index, plot_type="residual_phase", ...)`

绘制指定表面的 3D 图表。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `index` | `int` | - | 表面索引 |
| `plot_type` | `str` | `"residual_phase"` | 绘图类型 |
| `save_path` | `str` | `None` | 保存路径 |
| `show` | `bool` | `True` | 是否显示 |
| `elevation` | `float` | `30` | 3D 视角仰角（度） |
| `azimuth` | `float` | `-60` | 3D 视角方位角（度） |

**plot_type 可选值：**
- `"amplitude"`：振幅
- `"phase"`：相位
- `"residual_phase"`：残差相位
- `"pilot_amplitude"`：Pilot Beam 振幅
- `"residual_amplitude"`：振幅残差

---

##### `save(path)`

保存结果到目录。

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `path` | `str` | 保存目录路径 |

**示例：**
```python
result.save("output/my_result")
```

---

##### `load(path)` (类方法)

从目录加载结果。

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `path` | `str` | 目录路径 |

**返回值：**
- `SimulationResult` 对象

**示例：**
```python
loaded = bts.SimulationResult.load("output/my_result")
```


---

### `WavefrontData`

波前数据类，封装单个位置的波前数据，提供便捷的计算方法。

**注意：** 此类从 `hybrid_simulation` 模块重导出。

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `amplitude` | `NDArray` | 振幅网格（实数，非负） |
| `phase` | `NDArray` | 相位网格（实数，非折叠，弧度） |
| `pilot_beam` | `PilotBeamParams` | Pilot Beam 参数 |
| `grid` | `GridSampling` | 网格采样信息 |
| `wavelength_um` | `float` | 波长 (μm) |

#### 方法

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `get_intensity()` | `NDArray` | 计算光强分布 (amplitude²) |
| `get_complex_amplitude()` | `NDArray` | 获取复振幅 |
| `get_pilot_beam_phase()` | `NDArray` | 计算 Pilot Beam 参考相位 |
| `get_pilot_beam_amplitude()` | `NDArray` | 计算 Pilot Beam 参考振幅 |
| `get_residual_phase()` | `NDArray` | 计算残差相位 |
| `get_residual_amplitude()` | `NDArray` | 计算振幅残差 |
| `get_residual_rms_waves()` | `float` | 计算残差相位 RMS（波长数） |
| `get_residual_pv_waves()` | `float` | 计算残差相位 PV（波长数） |
| `get_amplitude_residual_rms()` | `float` | 计算振幅残差 RMS |

---

### `SurfaceRecord`

表面记录类，存储单个表面的完整数据。

**注意：** 此类从 `hybrid_simulation` 模块重导出。

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `index` | `int` | 表面索引 |
| `name` | `str` | 表面名称 |
| `surface_type` | `str` | 表面类型（如 `'standard'`、`'paraxial'`） |
| `geometry` | `SurfaceGeometry` | 表面几何信息 |
| `entrance` | `WavefrontData` | 入射面波前数据 |
| `exit` | `WavefrontData` | 出射面波前数据 |
| `optical_axis` | `OpticalAxisInfo` | 光轴状态信息 |


---

## 异常类

### `ParseError`

解析错误，当 ZMX 文件或其他配置文件解析失败时抛出。

**属性：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `message` | `str` | 错误描述 |
| `file_path` | `str` | 相关文件路径（可选） |
| `line_number` | `int` | 错误发生的行号（可选） |

**触发场景：**
- ZMX 文件格式无效
- 缺少必要的表面定义
- 参数值格式错误
- 不支持的 ZMX 特性

**示例：**
```python
try:
    system = bts.load_zmx("invalid.zmx")
except bts.ParseError as e:
    print(f"解析错误: {e.message}")
    if e.file_path:
        print(f"文件: {e.file_path}")
```

---

### `ConfigurationError`

配置错误，当仿真配置不完整或无效时抛出。

**触发场景：**
- 光学系统为空（没有表面）
- 缺少必要的配置参数
- 配置参数之间存在冲突

**示例：**
```python
try:
    system = bts.OpticalSystem()  # 空系统
    result = bts.simulate(system, source)
except bts.ConfigurationError as e:
    print(f"配置错误: {e}")
```

---

### `SimulationError`

仿真错误，当仿真执行过程中发生错误时抛出。

**触发场景：**
- 光线追迹失败
- 波前传播计算错误
- 数值计算异常

**示例：**
```python
try:
    result = bts.simulate(system, source)
except bts.SimulationError as e:
    print(f"仿真失败: {e}")
```


---

## 使用示例

### 示例 1：简单折叠镜

演示 bts API 的基本用法，代码极简。

```python
"""
简单折叠镜测试示例
"""

# ============================================================
# 1. 导入与初始化
# ============================================================
import bts

# ============================================================
# 2. 定义光学系统
# ============================================================
# 创建一个简单的光学系统，包含一个 45° 折叠镜
system = bts.OpticalSystem("Simple Fold Mirror")
system.add_flat_mirror(z=50.0, tilt_x=45.0)  # 45° 折叠镜

# ============================================================
# 3. 定义光源
# ============================================================
# 定义 He-Ne 激光高斯光源
source = bts.GaussianSource(wavelength_um=0.633, w0_mm=5.0, grid_size=256)

# ============================================================
# 4. 执行仿真
# ============================================================
result = bts.simulate(system, source)

# ============================================================
# 5. 查看结果
# ============================================================
# 打印结果摘要
result.summary()

# 绘制所有表面的概览图并保存
result.plot_all(save_path='fold_mirror_overview.png', show=False)

# 保存完整结果到目录
result.save('output/fold_mirror_result')

# 验证波前数据
final_wf = result.get_final_wavefront()
print(f"残差相位 RMS: {final_wf.get_residual_rms_waves()*1000:.3f} milli-waves")
print(f"残差相位 PV: {final_wf.get_residual_pv_waves():.4f} waves")
```


---

### 示例 2：ZMX 文件仿真

演示如何从 ZMX 文件加载光学系统并执行仿真。

```python
"""
ZMX 文件混合光学仿真示例
"""

# ============================================================
# 1. 导入与初始化
# ============================================================
import bts

# ============================================================
# 2. 加载 ZMX 文件
# ============================================================
system = bts.load_zmx("complicated_fold_mirrors_setup_v2.zmx")

# ============================================================
# 3. 系统信息展示
# ============================================================
# 打印系统参数摘要
system.print_info()

# 绘制光路图
system.plot_layout(
    projection='YZ', 
    num_rays=5, 
    save_path="output/zmx_layout.png",
    show=False
)

# ============================================================
# 4. 定义光源
# ============================================================
source = bts.GaussianSource(
    wavelength_um=0.633,    # He-Ne 激光波长 (μm)
    w0_mm=5.0,              # 束腰半径 (mm)
    grid_size=256,          # 网格大小
)

# 打印光源信息
source.print_info()

# ============================================================
# 5. 执行仿真
# ============================================================
result = bts.simulate(system, source)

# ============================================================
# 6. 结果展示与保存
# ============================================================
# 打印结果摘要
result.summary()

# 绘制所有表面的概览图
result.plot_all(save_path="output/zmx_simulation_overview.png", show=False)

# 保存完整结果到目录
result.save("output/zmx_result_data")

# 验证加载功能
loaded = bts.SimulationResult.load("output/zmx_result_data")
print(f"加载验证: success={loaded.success}, 表面数量={len(loaded.surfaces)}")
```


---

### 示例 3：链式调用

演示使用链式调用构建复杂光学系统。

```python
"""
链式调用示例 - 构建多元件光学系统
"""

import bts

# 使用链式调用构建光学系统
system = (bts.OpticalSystem("Multi-Element System")
    .add_flat_mirror(z=50.0, tilt_x=45.0, semi_aperture=30.0)
    .add_spherical_mirror(z=150.0, radius=200.0, semi_aperture=25.0)
    .add_paraxial_lens(z=250.0, focal_length=100.0)
)

# 查看系统信息
system.print_info()

# 定义光源
source = bts.GaussianSource(
    wavelength_um=1.064,    # Nd:YAG 激光波长
    w0_mm=3.0,
    grid_size=512,
)

# 执行仿真
result = bts.simulate(system, source, verbose=True)

# 分析各表面波前
for i in range(len(system)):
    try:
        exit_wf = result.get_exit_wavefront(i)
        rms = exit_wf.get_residual_rms_waves()
        print(f"表面 {i} 出射波前残差 RMS: {rms*1000:.3f} milli-waves")
    except ValueError:
        print(f"表面 {i} 无出射波前数据")
```

---

### 示例 4：近场倾斜平面镜仿真

详细的近场仿真示例，包含参数计算和结果分析。

```python
"""
近场高斯光束入射倾斜平面镜仿真

测试条件：
- 波长: 0.633 μm (He-Ne)
- 束腰半径: 5.0 mm
- 传输距离: 50 mm（近场）
- 平面镜倾斜角: 45°
"""

import bts
import numpy as np

# ============================================================
# 仿真参数
# ============================================================
wavelength_um = 0.633    # He-Ne 激光波长
w0_mm = 5.0              # 束腰半径
grid_size = 256          # 网格大小
mirror_z_mm = 50.0       # 镜面位置
tilt_angle_deg = 45.0    # 倾斜角度

# 计算瑞利长度
wavelength_mm = wavelength_um * 1e-3
z_R = np.pi * w0_mm**2 / wavelength_mm
print(f"瑞利长度: z_R = {z_R:.2f} mm")
print(f"z/z_R = {mirror_z_mm/z_R:.3f} (近场条件: < 1)")

# ============================================================
# 定义光学系统
# ============================================================
system = bts.OpticalSystem("Near Field Tilted Mirror")
system.add_flat_mirror(z=mirror_z_mm, tilt_x=tilt_angle_deg, semi_aperture=30.0)

# ============================================================
# 定义光源
# ============================================================
source = bts.GaussianSource(
    wavelength_um=wavelength_um,
    w0_mm=w0_mm,
    grid_size=grid_size,
    physical_size_mm=8 * w0_mm,
)

# ============================================================
# 系统信息展示
# ============================================================
system.print_info()
system.plot_layout(save_path="output/layout.png", show=False)

# ============================================================
# 执行仿真
# ============================================================
result = bts.simulate(system, source)

# ============================================================
# 结果分析
# ============================================================
result.summary()

# 获取最终波前数据
final_wf = result.get_final_wavefront()
rms_waves = final_wf.get_residual_rms_waves()
pv_waves = final_wf.get_residual_pv_waves()

print(f"\n最终波前误差:")
print(f"  相位残差 RMS: {rms_waves*1000:.3f} milli-waves")
print(f"  相位残差 PV:  {pv_waves:.4f} waves")

# ============================================================
# 结果保存
# ============================================================
result.plot_all(save_path="output/overview.png", show=False)
result.save("output/result_data")
```


---

## 单位约定

BTS API 使用以下单位约定：

| 量 | 单位 | 说明 |
|----|------|------|
| 长度 | mm | 所有空间尺寸 |
| 波长 | μm | 光源波长 |
| 角度 | 度 | 倾斜角度 |
| 相位 | rad | 内部计算 |
| OPD | waves | 波前误差 |

---

## 曲率半径符号约定

| 曲率半径 | 曲率中心位置 | 表面类型 |
|----------|--------------|----------|
| R > 0 | +Z 方向（当前坐标系） | 凹面 |
| R < 0 | -Z 方向（当前坐标系） | 凸面 |
| R = ∞ | 无穷远 | 平面 |

---

## 圆锥常数

| k 值 | 表面类型 |
|------|----------|
| k = 0 | 球面 |
| k = -1 | 抛物面 |
| k < -1 | 双曲面 |
| -1 < k < 0 | 扁椭球面 |
| k > 0 | 长椭球面 |

---

## 常见问题

### Q: 如何选择合适的网格大小？

**A:** 网格大小影响仿真精度和计算时间：
- `128`：快速预览，精度较低
- `256`：标准仿真，推荐默认值
- `512`：高精度仿真
- `1024`：超高精度，计算时间较长

### Q: 如何判断是近场还是远场？

**A:** 比较传播距离 z 与瑞利距离 z_R：
- 近场：z < z_R
- 远场：z > z_R
- 瑞利距离计算：z_R = π × w0² / λ

### Q: 为什么仿真结果中没有整体倾斜？

**A:** BTS 系统自动追踪光轴方向变化，OPD 相对于主光线计算，倾斜相位被自动补偿。结果中只包含真实像差，不包含整体倾斜。

### Q: 如何保存和加载仿真结果？

**A:** 使用 `save()` 和 `load()` 方法：
```python
# 保存
result.save("output/my_result")

# 加载
loaded = bts.SimulationResult.load("output/my_result")
```

---

## 版本信息

- **API 版本**：1.0
- **兼容性**：保留现有 `HybridSimulator` 类作为备选接口
