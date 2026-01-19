# 设计文档：混合传播复振幅重建优化

## 概述

本文档描述混合传播复振幅重建优化的技术设计。核心目标是复用 optiland 的光线→复振幅重建技术，同时重建振幅和相位，解决当前实现只处理相位、忽略振幅的问题。

## 核心数据流

```
PROPER 波前 → 采样光线 → ElementRaytracer 光线追迹 → 获取输入/输出位置 + OPD 
→ 雅可比矩阵计算振幅 → RayToWavefrontReconstructor 重建复振幅 → 加回理论曲率相位 → 更新 PROPER 波前
```

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SequentialOpticalSystem                              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   _apply_element_hybrid()                        │   │
│  │                                                                  │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │   │
│  │  │ PROPER 波前   │───▶│ElementRaytracer│───▶│RayToWavefront   │  │   │
│  │  │ 采样         │    │ 光线追迹      │    │Reconstructor    │  │   │
│  │  └──────────────┘    └──────────────┘    └──────────────────┘  │   │
│  │         │                   │                     │             │   │
│  │         ▼                   ▼                     ▼             │   │
│  │    位置 (x, y)         OPD + 位置变化       复振幅网格          │   │
│  │                                                   │             │   │
│  │                              ┌────────────────────┘             │   │
│  │                              ▼                                  │   │
│  │                    ┌──────────────────┐                        │   │
│  │                    │ 加回理论曲率相位  │                        │   │
│  │                    └──────────────────┘                        │   │
│  │                              │                                  │   │
│  │                              ▼                                  │   │
│  │                    ┌──────────────────┐                        │   │
│  │                    │ 更新 PROPER 波前  │                        │   │
│  │                    └──────────────────┘                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 核心组件

#### 1. RayToWavefrontReconstructor（新增）

封装光线→复振幅重建逻辑，复用 optiland 的技术。

**对应需求**: 需求 2

```python
class RayToWavefrontReconstructor:
    """光线到波前复振幅重建器
    
    使用雅可比矩阵方法（网格变形）计算振幅，基于能量守恒原理。
    将稀疏光线数据（位置、OPD）重建为规则网格上的复振幅。
    """
    
    def __init__(
        self,
        grid_size: int,
        sampling_mm: float,
        wavelength_um: float,
    ):
        """初始化重建器
        
        参数:
            grid_size: 输出网格大小（与 PROPER 网格一致）
            sampling_mm: 网格采样间隔 (mm/pixel)
            wavelength_um: 波长 (μm)
        """
        self.grid_size = grid_size
        self.sampling_mm = sampling_mm
        self.wavelength_um = wavelength_um
    
    def reconstruct(
        self,
        ray_x_in: np.ndarray,
        ray_y_in: np.ndarray,
        ray_x_out: np.ndarray,
        ray_y_out: np.ndarray,
        opd_waves: np.ndarray,
        valid_mask: np.ndarray,
        check_phase_discontinuity: bool = True,
    ) -> np.ndarray:
        """重建复振幅（使用雅可比矩阵方法）
        
        参数:
            ray_x_in: 输入面光线 x 坐标 (mm)
            ray_y_in: 输入面光线 y 坐标 (mm)
            ray_x_out: 输出面光线 x 坐标 (mm)
            ray_y_out: 输出面光线 y 坐标 (mm)
            opd_waves: OPD (波长数)
            valid_mask: 有效光线掩模
            check_phase_discontinuity: 是否检测相位突变（默认 True）
        
        返回:
            复振幅网格 (grid_size × grid_size 复数数组)
        """
        # 1. 使用雅可比矩阵计算振幅和相位
        amplitude, phase = self._compute_amplitude_phase_jacobian(
            ray_x_in, ray_y_in, ray_x_out, ray_y_out, opd_waves, valid_mask
        )
        
        # 2. 重采样到 PROPER 网格（使用输出位置）
        amp_grid, phase_grid = self._resample_to_grid_separate(
            ray_x_out, ray_y_out, amplitude, phase, valid_mask
        )
        
        # 3. 检测相位突变（重采样后、加回理想相位之前）（需求 6.2）
        if check_phase_discontinuity:
            self._check_phase_discontinuity_on_grid(phase_grid, amp_grid)
        
        # 4. 组合为复振幅
        complex_amplitude = amp_grid * np.exp(1j * phase_grid)
        
        return complex_amplitude
```



#### 2. ElementRaytracer（修改）

保存输入光线位置以支持雅可比矩阵计算。

**对应需求**: 需求 1

```python
class ElementRaytracer:
    def __init__(self, ...):
        # ... 现有代码 ...
        self.input_rays = None   # 新增：保存输入光线
        self.output_rays = None
    
    def trace(self, rays_in) -> RealRays:
        """执行光线追迹
        
        新增：保存输入光线位置，用于雅可比矩阵计算
        """
        # 保存输入光线（用于雅可比矩阵计算）
        self.input_rays = rays_in
        
        # ... 现有追迹代码 ...
        
        self.output_rays = rays_out
        return rays_out
    
    def get_input_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取输入光线位置（用于雅可比矩阵计算）
        
        返回:
            (x, y) 元组，输入面光线位置 (mm)
        
        异常:
            RuntimeError: 如果 trace() 方法尚未调用
        """
        if self.input_rays is None:
            raise RuntimeError("尚未执行光线追迹。请先调用 trace() 方法。")
        
        return np.asarray(self.input_rays.x), np.asarray(self.input_rays.y)
    
    def get_output_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取输出光线位置（用于雅可比矩阵计算）
        
        返回:
            (x, y) 元组，输出面光线位置 (mm)
        
        异常:
            RuntimeError: 如果 trace() 方法尚未调用
        """
        if self.output_rays is None:
            raise RuntimeError("尚未执行光线追迹。请先调用 trace() 方法。")
        
        return np.asarray(self.output_rays.x), np.asarray(self.output_rays.y)
```

#### 3. _apply_element_hybrid()（重构）

重构后的方法使用新的复振幅重建流程。

**对应需求**: 需求 3, 4, 5

```python
def _apply_element_hybrid(self, wfo, element) -> None:
    """使用混合传播模式应用光学元件效果（重构版）
    
    核心改进：
    1. 同时重建振幅和相位（而不是只处理相位）
    2. 使用雅可比矩阵方法计算振幅（基于能量守恒）
    3. 正确处理理论曲率相位
    """
    # 1. 获取 SurfaceDefinition
    surface_def = element.get_surface_definition()
    if surface_def is None:
        # 向后兼容：使用 prop_lens
        if not np.isinf(element.focal_length):
            proper.prop_lens(wfo, element.focal_length * 1e-3)
        return
    
    # 2. 更新高斯光束参数（不修改 wfarr）
    if not np.isinf(element.focal_length):
        self._update_gaussian_params_only(wfo, element.focal_length * 1e-3)
    
    # 3. 创建采样光线并追迹
    rays_in = self._create_sampling_rays_from_proper(wfo)
    raytracer = ElementRaytracer(
        surfaces=[surface_def], 
        wavelength=wavelength_um
    )
    rays_out = raytracer.trace(rays_in)
    
    # 4. 获取输入/输出位置和 OPD（需求 1）
    ray_x_in, ray_y_in = raytracer.get_input_positions()
    ray_x_out, ray_y_out = raytracer.get_output_positions()
    opd_waves = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    # 5. 计算像差 OPD
    aberration_opd = self._compute_aberration_opd(
        opd_waves, ray_x_out, ray_y_out, element
    )
    
    # 6. 使用 RayToWavefrontReconstructor 重建复振幅（需求 2）
    reconstructor = RayToWavefrontReconstructor(
        grid_size=n,
        sampling_mm=sampling_mm,
        wavelength_um=wavelength_um,
    )
    
    # 使用雅可比矩阵方法重建复振幅
    aberration_field = reconstructor.reconstruct(
        ray_x_in, ray_y_in,
        ray_x_out, ray_y_out,
        aberration_opd, valid_mask
    )
    
    # 7. 加回理论曲率相位（需求 3）
    theory_phase = self._compute_theory_curvature_phase(wfo, element)
    full_field = aberration_field * np.exp(1j * theory_phase)
    
    # 8. 更新 PROPER 波前（需求 4）
    full_field_fft = proper.prop_shift_center(full_field)
    wfo.wfarr = full_field_fft
```

## 详细设计

### 1. 复振幅计算算法（雅可比矩阵方法）

基于能量守恒原理，通过计算光线从输入面到输出面的位置变化（网格变形），使用雅可比矩阵计算局部面积变化，从而得到光强变化。

**对应需求**: 需求 2

**物理原理**：
- 能量守恒：`I_in × dA_in = I_out × dA_out`
- 因此：`I_out / I_in = dA_in / dA_out = 1 / |J|`
- 其中 `|J|` 是雅可比行列式，表示局部面积放大率

**与光线追迹的关系**：
- 雅可比矩阵计算**复用已有的光线追迹结果**，不需要额外的追迹
- 输入数据：`ray_x_in, ray_y_in`（输入面光线位置）和 `ray_x_out, ray_y_out`（输出面光线位置）
- 这些数据在 `ElementRaytracer.trace()` 过程中已经获得

**数据流**：
```
ElementRaytracer.trace()
    ├── 输入光线位置 (ray_x_in, ray_y_in) ──┐
    │                                        ├──▶ 雅可比矩阵计算 ──▶ 振幅
    └── 输出光线位置 (ray_x_out, ray_y_out) ─┘
```

```python
def _compute_amplitude_phase_jacobian(
    self,
    ray_x_in: np.ndarray,
    ray_y_in: np.ndarray,
    ray_x_out: np.ndarray,
    ray_y_out: np.ndarray,
    opd_waves: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算复振幅的振幅和相位分量（基于雅可比矩阵）
    
    通过计算输入面到输出面的网格变形，使用雅可比矩阵计算光强变化。
    
    注意：此方法复用已有的光线追迹结果，不需要额外的追迹。
    输入/输出光线位置在 ElementRaytracer.trace() 过程中已经获得。
    
    物理原理：
    - 能量守恒：I_in × dA_in = I_out × dA_out
    - 振幅比：A_out / A_in = sqrt(I_out / I_in) = 1 / sqrt(|J|)
    - 其中 |J| 是雅可比行列式（局部面积放大率）
    
    参数:
        ray_x_in: 输入面光线 x 坐标 (mm)，来自采样光线
        ray_y_in: 输入面光线 y 坐标 (mm)，来自采样光线
        ray_x_out: 输出面光线 x 坐标 (mm)，来自追迹结果
        ray_y_out: 输出面光线 y 坐标 (mm)，来自追迹结果
        opd_waves: OPD (波长数)
        valid_mask: 有效光线掩模
    
    返回:
        (amplitude, phase) 元组
    """
    from scipy.interpolate import RBFInterpolator
    
    # 只使用有效光线
    valid_x_in = ray_x_in[valid_mask]
    valid_y_in = ray_y_in[valid_mask]
    valid_x_out = ray_x_out[valid_mask]
    valid_y_out = ray_y_out[valid_mask]
    
    if len(valid_x_in) < 4:
        raise ValueError(f"有效光线数量不足：{len(valid_x_in)} < 4")
    
    # 创建从输入坐标到输出坐标的映射函数
    points_in = np.column_stack([valid_x_in, valid_y_in])
    
    # 使用 RBF 插值创建平滑的映射函数
    interp_x = RBFInterpolator(points_in, valid_x_out, kernel='thin_plate_spline')
    interp_y = RBFInterpolator(points_in, valid_y_out, kernel='thin_plate_spline')
    
    # 计算雅可比矩阵的各分量（使用数值微分）
    eps = 1e-6  # 微分步长 (mm)
    
    # 在每个有效光线位置计算雅可比矩阵
    jacobian_det = np.zeros(len(valid_x_in))
    
    for i in range(len(valid_x_in)):
        x0, y0 = valid_x_in[i], valid_y_in[i]
        
        # 计算 ∂x_out/∂x_in
        dx_out_dx_in = (interp_x([[x0 + eps, y0]])[0] - interp_x([[x0 - eps, y0]])[0]) / (2 * eps)
        # 计算 ∂x_out/∂y_in
        dx_out_dy_in = (interp_x([[x0, y0 + eps]])[0] - interp_x([[x0, y0 - eps]])[0]) / (2 * eps)
        # 计算 ∂y_out/∂x_in
        dy_out_dx_in = (interp_y([[x0 + eps, y0]])[0] - interp_y([[x0 - eps, y0]])[0]) / (2 * eps)
        # 计算 ∂y_out/∂y_in
        dy_out_dy_in = (interp_y([[x0, y0 + eps]])[0] - interp_y([[x0, y0 - eps]])[0]) / (2 * eps)
        
        # 雅可比行列式 = ∂x_out/∂x_in × ∂y_out/∂y_in - ∂x_out/∂y_in × ∂y_out/∂x_in
        jacobian_det[i] = abs(dx_out_dx_in * dy_out_dy_in - dx_out_dy_in * dy_out_dx_in)
    
    # 避免除零
    jacobian_det = np.maximum(jacobian_det, 1e-10)
    
    # 振幅 = 1 / sqrt(|J|)（能量守恒）
    amplitude_valid = 1.0 / np.sqrt(jacobian_det)
    
    # 归一化振幅
    mean_amplitude = np.mean(amplitude_valid)
    amplitude_valid = amplitude_valid / mean_amplitude
    
    # 构建完整的振幅数组
    amplitude = np.zeros_like(ray_x_in)
    amplitude[valid_mask] = amplitude_valid
    
    # 相位 = -2π × OPD
    phase = -2 * np.pi * opd_waves
    
    return amplitude, phase
```

### 2. 相位突变检测（重采样后）

在重采样完成后、加回理想相位之前，检测网格上相邻像素之间的相位差值是否过大。

**对应需求**: 需求 6.2

**检测时机**：在 `_resample_to_grid()` 完成后，对重采样得到的相位网格进行检测。

```python
def _check_phase_discontinuity_on_grid(
    self,
    phase_grid: np.ndarray,
    amplitude_grid: np.ndarray,
    threshold_rad: float = np.pi,
) -> bool:
    """检测重采样后网格上的相位突变
    
    原理：
    - 相邻像素之间的相位差不应超过 π
    - 如果超过，可能导致相位混叠（aliasing）
    - 这是 Nyquist 采样准则的要求
    
    检测时机：
    - 在重采样完成后、加回理想相位之前
    - 只检测像差相位（已减除理想相位的部分）
    
    参数:
        phase_grid: 重采样后的相位网格 (弧度)
        amplitude_grid: 重采样后的振幅网格（用于确定有效区域）
        threshold_rad: 相位差阈值（弧度），默认 π
    
    返回:
        True 如果检测到相位突变，False 否则
    
    副作用:
        如果检测到相位突变，发出 UserWarning
    """
    import warnings
    
    # 创建有效区域掩模（振幅 > 0 的区域）
    valid_mask = amplitude_grid > 1e-10
    
    # 计算 x 方向相邻像素的相位差
    phase_diff_x = np.abs(np.diff(phase_grid, axis=1))
    # 计算 y 方向相邻像素的相位差
    phase_diff_y = np.abs(np.diff(phase_grid, axis=0))
    
    # 只检查有效区域内的相位差
    # x 方向：两个相邻像素都必须有效
    valid_x = valid_mask[:, :-1] & valid_mask[:, 1:]
    # y 方向：两个相邻像素都必须有效
    valid_y = valid_mask[:-1, :] & valid_mask[1:, :]
    
    # 获取有效区域内的最大相位差
    max_diff_x = np.max(phase_diff_x[valid_x]) if np.any(valid_x) else 0
    max_diff_y = np.max(phase_diff_y[valid_y]) if np.any(valid_y) else 0
    max_phase_diff = max(max_diff_x, max_diff_y)
    
    # 检查是否超过阈值
    has_discontinuity = max_phase_diff > threshold_rad
    
    if has_discontinuity:
        max_opd_waves = max_phase_diff / (2 * np.pi)
        warnings.warn(
            f"检测到相位突变：重采样后网格上相邻像素最大相位差为 {max_phase_diff:.3f} 弧度 "
            f"({max_opd_waves:.3f} 波长)，超过阈值 {threshold_rad:.3f} 弧度（π）。"
            f"这可能导致相位混叠，建议增加采样密度。",
            UserWarning
        )
    
    return has_discontinuity
```

### 3. 重采样/插值算法

将稀疏光线数据插值到规则 PROPER 网格。

**对应需求**: 需求 2.5, 2.6, 5.4

```python
def _resample_to_grid_separate(
    self,
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    amplitude: np.ndarray,
    phase: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """将光线数据重采样到规则网格（分别返回振幅和相位）
    
    关键设计决策：
    1. 分别插值振幅和相位
       - 避免直接插值复数导致的相位跳变问题
    2. 使用三次插值（cubic）提高精度
    3. 采样范围外的区域设为 0（需求 5.4）
    4. 分别返回振幅和相位网格，便于后续相位突变检测
    """
    from scipy.interpolate import griddata
    
    # 只使用有效光线
    valid_x = ray_x[valid_mask]
    valid_y = ray_y[valid_mask]
    valid_amp = amplitude[valid_mask]
    valid_phase = phase[valid_mask]
    
    if len(valid_x) < 4:
        raise ValueError(f"有效光线数量不足：{len(valid_x)} < 4")
    
    # 创建目标网格坐标
    half_size = self.sampling_mm * self.grid_size / 2
    coords = np.linspace(-half_size, half_size, self.grid_size)
    X_grid, Y_grid = np.meshgrid(coords, coords)
    
    # 插值点
    points = np.column_stack([valid_x, valid_y])
    
    # 分别插值振幅和相位
    amp_grid = griddata(
        points, valid_amp, (X_grid, Y_grid),
        method='cubic', fill_value=0.0
    )
    
    phase_grid = griddata(
        points, valid_phase, (X_grid, Y_grid),
        method='cubic', fill_value=0.0
    )
    
    # 处理 NaN 值
    amp_grid = np.nan_to_num(amp_grid, nan=0.0)
    phase_grid = np.nan_to_num(phase_grid, nan=0.0)
    
    return amp_grid, phase_grid
```



### 3. 理论曲率相位计算

计算元件的理论聚焦相位（不含倾斜）。

**对应需求**: 需求 3

```python
def _compute_theory_curvature_phase(
    self,
    wfo,
    element,
) -> np.ndarray:
    """计算理论曲率相位
    
    公式（需求 3.1）：
        φ_theory = -k × r² / (2f)
        
    其中：
        k = 2π/λ 为波数
        r² = x² + y² 为到光轴的距离平方
        f 为焦距
    
    注意：
    1. 此相位不包含倾斜分量（需求 3.3）
    2. 对于平面镜（f = ∞），理论相位为 0（需求 3.2）
    """
    import proper
    
    n = proper.prop_get_gridsize(wfo)
    sampling_m = proper.prop_get_sampling(wfo)
    wavelength_m = wfo.lamda
    
    # 创建坐标网格（单位：m）
    half_size_m = sampling_m * n / 2
    coords_m = np.linspace(-half_size_m, half_size_m, n)
    X_m, Y_m = np.meshgrid(coords_m, coords_m)
    r_sq_m = X_m**2 + Y_m**2
    
    # 检查是否为平面镜（需求 3.2）
    if np.isinf(element.focal_length):
        return np.zeros((n, n))
    
    # 计算理论相位（需求 3.1）
    focal_length_m = element.focal_length * 1e-3  # mm → m
    k = 2 * np.pi / wavelength_m
    
    # φ = -k × r² / (2f)
    theory_phase = -k * r_sq_m / (2 * focal_length_m)
    
    return theory_phase
```

### 4. PROPER 波前更新

**对应需求**: 需求 4

```python
def _update_proper_wavefront(
    self,
    wfo,
    complex_amplitude: np.ndarray,
) -> None:
    """更新 PROPER 波前
    
    处理步骤：
    1. 转换到 FFT 坐标系（需求 4.2）
    2. 更新波前数组 wfarr
    """
    import proper
    
    # 转换到 FFT 坐标系（需求 4.2）
    complex_amplitude_fft = proper.prop_shift_center(complex_amplitude)
    
    # 更新波前数组
    wfo.wfarr = complex_amplitude_fft
```

## 坐标系统映射

**对应需求**: 需求 5

### 光线坐标与 PROPER 网格坐标

```
光线坐标系（ElementRaytracer）:
- 原点：入射面中心
- 单位：mm
- 范围：由采样策略决定

PROPER 网格坐标系:
- 原点：网格中心
- 单位：m（内部），mm（接口）
- 范围：[-n/2 * sampling, n/2 * sampling]

映射关系（需求 5.1, 5.2）：
- 光线坐标直接对应 PROPER 网格坐标（同一物理位置）
- 单位转换：mm ↔ m
```

## 与 optiland 代码的对应关系

本项目使用雅可比矩阵方法计算振幅，不依赖 optiland 的 intensity 属性。

| 本项目组件 | 说明 |
|-----------|------|
| 振幅计算 | 基于雅可比矩阵（网格变形），能量守恒原理 |
| 复振幅公式 | `A × exp(-j × 2π × OPD)` |
| 相位计算 | 使用 optiland 追迹得到的 OPD |

### 提前减掉参考相位的影响

**结论：提前减掉理想相位（参考相位）不影响复振幅重建的正确性。**

原因：
1. 复振幅公式 `A × exp(-j × 2π × OPD)` 中，OPD 的绝对值不影响振幅
2. 相位的绝对偏移（piston）在后续传播中会被正确处理
3. 我们在重建后加回理论曲率相位，恢复完整的相位信息

## 错误处理

**对应需求**: 需求 6

### 异常类型

```python
class ReconstructionError(Exception):
    """复振幅重建错误"""
    pass

class InsufficientRaysError(ReconstructionError):
    """有效光线数量不足（需求 6.1）"""
    pass
```

### 错误处理策略

| 情况 | 处理方式 | 对应需求 |
|------|----------|----------|
| 有效光线 < 4 | 抛出 `InsufficientRaysError` | 需求 6.1 |
| 重采样后相邻像素相位差 > π | 发出 `UserWarning` | 需求 6.2, 6.3 |
| 雅可比行列式接近零 | 使用最小值限制，避免除零 | 需求 2 |

### 相位突变检测原理

相位突变检测基于 Nyquist 采样准则：

1. **采样准则**：相邻采样点之间的相位差不应超过 π
2. **检测时机**：在重采样完成后、加回理想相位之前
3. **检测对象**：像差相位网格（已减除理想相位的部分）
4. **检测方法**：计算网格上 x 和 y 方向相邻像素的相位差
5. **处理方式**：发出警告而非抛出异常，因为：
   - 用户可能有意使用高像差系统
   - 警告提供诊断信息，不中断仿真流程
   - 建议用户增加采样密度

## 测试计划

### 单元测试

| 测试文件 | 测试内容 | 对应需求 |
|----------|----------|----------|
| `test_ray_to_wavefront_reconstructor.py` | 复振幅重建器功能 | 需求 2 |
| `test_jacobian_amplitude.py` | 雅可比矩阵振幅计算 | 需求 2 |
| `test_theory_curvature_phase.py` | 理论曲率相位计算 | 需求 3 |

### 集成测试

| 测试文件 | 测试内容 | 对应需求 |
|----------|----------|----------|
| `test_galilean_oap_amplitude.py` | 伽利略 OAP 扩束器振幅变化 | 需求 7 |
| `test_hybrid_vs_proper.py` | 与纯 PROPER 模式对比 | 需求 7 |
| `test_gaussian_beam_amplitude.py` | 高斯光束振幅验证 | 需求 2 |

### 高斯光束振幅验证测试

**测试目的**：验证雅可比矩阵方法对高斯光束的准确性。

**测试场景**：
1. 高斯光束通过理想透镜聚焦
2. 高斯光束通过 OAP 反射镜
3. 高斯光束扩束器

**理论参考**：
- 高斯光束在自由空间传播的解析解
- ABCD 矩阵法计算的光束参数变化

**对比指标**：
- 振幅分布的 RMS 误差
- 光束半径的相对误差
- 峰值位置的偏差

## 正确性属性（Property-Based Testing）

### 属性 1：能量守恒

```python
@given(
    intensity=st.lists(st.floats(min_value=0.1, max_value=2.0), min_size=10)
)
def test_energy_conservation(intensity):
    """
    **Validates: 需求 2.4 - 振幅归一化**
    
    重建后的总能量应与输入能量成比例。
    """
    # 归一化后，mean(amplitude²) ≈ 1
    pass
```

### 属性 2：相位连续性

```python
@given(
    opd_rms=st.floats(min_value=0.0, max_value=0.5)
)
def test_phase_continuity(opd_rms):
    """
    **Validates: 需求 6.2 - 相位梯度检查**
    
    相邻点相位差不应超过 π。
    """
    pass
```

## 风险与缓解

| 风险 | 影响 | 缓解措施 | 对应需求 |
|------|------|----------|----------|
| 插值精度不足 | 相位误差 | 使用三次插值，增加采样密度 | 需求 2 |
| 相位混叠 | 波前失真 | 在重采样后检测相邻像素相位差，发出警告 | 需求 6 |
| 能量不守恒 | 物理不正确 | 验证归一化，添加能量检查 | 需求 2 |
| 向后兼容性 | 现有测试失败 | 保持接口不变 | 需求 7 |
| 采样密度不足 | 相位突变警告 | 提供清晰的警告信息，建议增加采样密度 | 需求 6 |
| 雅可比矩阵数值不稳定 | 振幅计算错误 | 使用 RBF 插值平滑映射，避免除零 | 需求 2 |
