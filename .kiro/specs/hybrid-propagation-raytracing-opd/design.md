# 设计文档：混合传播模式光线追迹 OPD 重构

## 概述

本设计文档描述如何重构混合光学传播模式，使其完全使用真实的几何光线追迹计算 OPD，而不依赖 PROPER 的 `prop_lens` 函数进行相位计算。

## 架构

### 核心设计原则

1. **OPD 计算完全依赖 ElementRaytracer**：不使用 `prop_lens` 计算相位，而是使用几何光线追迹计算完整的 OPD
2. **高斯光束参数更新可使用 prop_lens 逻辑**：复用 `prop_lens` 的参数更新算法，但不使用其相位计算
3. **正确处理参考面变换**：将光线追迹的绝对 OPD 转换为相对于 PROPER 参考面的相位偏差

### 组件关系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SequentialOpticalSystem                               │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  _apply_element_hybrid() 方法（重构后）                             │ │
│  │                                                                     │ │
│  │  1. 更新高斯光束参数（使用 prop_lens 逻辑，不修改 wfarr）          │ │
│  │  2. 使用 ElementRaytracer 计算完整 OPD                             │ │
│  │  3. 计算参考面相位（基于更新后的 z_w0）                            │ │
│  │  4. 计算残差相位 = OPD 相位 - 参考面相位                           │ │
│  │  5. 将残差相位应用到波前                                           │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  _update_gaussian_params_only() 方法（新增）                        │ │
│  │                                                                     │ │
│  │  复用 prop_lens 的参数更新逻辑：                                   │ │
│  │  - 更新 z_w0, w0, z_Rayleigh                                       │ │
│  │  - 更新 beam_type_old, reference_surface                           │ │
│  │  - 不修改 wfarr                                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       ElementRaytracer                                   │
│                                                                          │
│  - trace(): 执行光线追迹                                                │
│  - get_relative_opd_waves(): 获取相对于主光线的 OPD（波长数）          │
│  - get_valid_ray_mask(): 获取有效光线掩模                               │
└─────────────────────────────────────────────────────────────────────────┘
```

## 数据流

```
输入波前 (wfo)
      │
      ▼
┌─────────────────────────────────────┐
│ 1. 更新高斯光束参数                 │
│    - 计算新的 z_w0, w0, z_Rayleigh  │
│    - 更新 reference_surface         │
│    - 不修改 wfarr                   │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ 2. 采样光线                         │
│    - 在整个采样面上创建采样点       │
│    - 采样半径 = PROPER 网格半尺寸   │
│    - 创建平行光入射光线             │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ 3. ElementRaytracer 光线追迹        │
│    - 计算每条光线的 OPD             │
│    - 返回相对于主光线的 OPD         │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ 4. 计算参考面相位                   │
│    - 使用更新后的 z_w0              │
│    - R_ref = z - z_w0               │
│    - φ_ref = -k * r² / (2 * R_ref)  │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ 5. 计算残差相位                     │
│    - φ_opd = 2π * OPD_waves         │
│    - φ_residual = φ_opd - φ_ref     │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ 6. 插值到网格并应用                 │
│    - 将残差相位插值到 PROPER 网格   │
│    - wfarr *= exp(i * φ_residual)   │
└─────────────────────────────────────┘
      │
      ▼
输出波前 (wfo)
```

## 详细设计

### 1. PROPER 参考面机制

PROPER 使用参考球面跟踪机制来减少相位存储需求：

- **PLANAR 参考面**：当光束在瑞利距离内（`|z - z_w0| < rayleigh_factor * z_Rayleigh`）时使用
- **SPHERI 参考面**：当光束在瑞利距离外时使用，参考球面曲率半径 `R_ref = z - z_w0`

存储的相位是相对于参考面的偏差：
```
φ_stored = φ_actual - φ_reference
```

对于理想高斯光束，`φ_stored ≈ 0`。

### 2. 高斯光束参数更新

复用 `prop_lens` 的参数更新逻辑，但不修改 `wfarr`：

```python
def _update_gaussian_params_only(self, wfo, focal_length_m: float) -> None:
    """更新 PROPER 的高斯光束跟踪参数（不修改 wfarr）
    
    复用 prop_lens 的参数更新算法：
    1. 计算当前表面处的束腰半径
    2. 使用高斯光束 ABCD 变换计算新的曲率半径
    3. 更新 z_w0, w0, z_Rayleigh
    4. 更新 beam_type_old, reference_surface, propagator_type
    
    参数:
        wfo: PROPER 波前对象
        focal_length_m: 透镜焦距（单位：m）
    """
    import proper
    
    rayleigh_factor = proper.rayleigh_factor
    
    # 计算当前表面处的束腰半径
    wfo.z_Rayleigh = np.pi * wfo.w0**2 / wfo.lamda
    w_at_surface = wfo.w0 * np.sqrt(1.0 + ((wfo.z - wfo.z_w0) / wfo.z_Rayleigh)**2)
    
    # 计算高斯光束曲率半径变换
    if (wfo.z - wfo.z_w0) != 0.0:
        gR_beam_old = (wfo.z - wfo.z_w0) + wfo.z_Rayleigh**2 / (wfo.z - wfo.z_w0)
        
        if gR_beam_old != focal_length_m:
            gR_beam = 1.0 / (1.0 / gR_beam_old - 1.0 / focal_length_m)
            gR_beam_inf = 0
        else:
            gR_beam_inf = 1
    else:
        gR_beam = -focal_length_m
        gR_beam_inf = 0
    
    # 更新束腰位置和束腰半径
    if not gR_beam_inf:
        wfo.z_w0 = -gR_beam / (1.0 + (wfo.lamda * gR_beam / (np.pi * w_at_surface**2))**2) + wfo.z
        wfo.w0 = w_at_surface / np.sqrt(1.0 + (np.pi * w_at_surface**2 / (wfo.lamda * gR_beam))**2)
    else:
        wfo.z_w0 = wfo.z
        wfo.w0 = w_at_surface
    
    # 更新瑞利距离
    wfo.z_Rayleigh = np.pi * wfo.w0**2 / wfo.lamda
    
    # 确定新的光束类型
    if np.abs(wfo.z_w0 - wfo.z) < rayleigh_factor * wfo.z_Rayleigh:
        beam_type_new = "INSIDE_"
    else:
        beam_type_new = "OUTSIDE"
    
    # 更新传播器类型
    wfo.propagator_type = wfo.beam_type_old + "_to_" + beam_type_new
    
    # 更新参考面类型
    if beam_type_new == "INSIDE_":
        wfo.reference_surface = "PLANAR"
    else:
        wfo.reference_surface = "SPHERI"
    
    # 更新光束类型
    wfo.beam_type_old = beam_type_new
    
    # 更新当前 F 数
    wfo.current_fratio = np.abs(wfo.z_w0 - wfo.z) / (2.0 * w_at_surface)
```

### 3. 参考面相位计算

根据 PROPER 的参考面类型计算参考相位：

```python
def _compute_reference_phase(
    self,
    wfo,
    x_mm: NDArray,
    y_mm: NDArray,
) -> NDArray:
    """计算参考面相位
    
    参数:
        wfo: PROPER 波前对象
        x_mm: X 坐标数组（单位：mm）
        y_mm: Y 坐标数组（单位：mm）
    
    返回:
        参考面相位（弧度）
    """
    wavelength_m = wfo.lamda
    k = 2 * np.pi / wavelength_m  # 波数（1/m）
    
    # 转换坐标到米
    x_m = x_mm * 1e-3
    y_m = y_mm * 1e-3
    r_sq_m = x_m**2 + y_m**2
    
    if wfo.reference_surface == "PLANAR":
        # 平面参考面，参考相位为零
        return np.zeros_like(r_sq_m)
    else:
        # 球面参考面
        R_ref_m = wfo.z - wfo.z_w0
        
        if abs(R_ref_m) < 1e-10:
            # 参考球面曲率半径接近零，视为平面
            return np.zeros_like(r_sq_m)
        
        # 参考球面相位：φ_ref = -k * r² / (2 * R_ref)
        phase_ref = -k * r_sq_m / (2 * R_ref_m)
        return phase_ref
```

### 4. 重构后的 _apply_element_hybrid 方法

```python
def _apply_element_hybrid(self, wfo, element) -> None:
    """使用混合传播模式应用光学元件效果（重构版）
    
    核心设计：
    - OPD 计算完全依赖 ElementRaytracer
    - 高斯光束参数更新使用 prop_lens 逻辑
    - 正确处理参考面变换
    
    流程：
    1. 更新高斯光束参数（不修改 wfarr）
    2. 使用 ElementRaytracer 计算完整 OPD
    3. 计算参考面相位
    4. 计算残差相位并应用到波前
    
    参数:
        wfo: PROPER 波前对象
        element: 光学元件对象
    """
    import proper
    
    # 处理平面镜（焦距无穷大）
    if np.isinf(element.focal_length):
        is_fold = getattr(element, 'is_fold', True)
        if not is_fold and (element.tilt_x != 0 or element.tilt_y != 0):
            # 失调倾斜，添加倾斜相位
            self._apply_tilt_phase(wfo, element)
        return
    
    # 获取 SurfaceDefinition
    surface_def = element.get_surface_definition()
    if surface_def is None:
        # 没有 SurfaceDefinition（如 ThinLens），使用 prop_lens
        proper.prop_lens(wfo, element.focal_length * 1e-3)
        return
    
    # =========================================================================
    # 步骤 1: 更新高斯光束参数（不修改 wfarr）
    # =========================================================================
    focal_length_m = element.focal_length * 1e-3
    self._update_gaussian_params_only(wfo, focal_length_m)
    
    # =========================================================================
    # 步骤 2: 使用 ElementRaytracer 计算完整 OPD
    # =========================================================================
    # 获取波前参数
    n = proper.prop_get_gridsize(wfo)
    sampling_m = proper.prop_get_sampling(wfo)
    sampling_mm = sampling_m * 1e3
    wavelength_um = wfo.lamda * 1e6
    
    # 获取采样面半尺寸（使用完整网格尺寸，避免面积收缩）
    half_size_mm = self._get_sampling_half_size_mm(wfo)
    
    # 创建采样光线（在整个采样面上）
    ray_x, ray_y = self._create_sampling_rays(half_size_mm)
    n_rays = len(ray_x)
    
    if n_rays == 0:
        return
    
    # 创建平行光入射光线
    from optiland.rays import RealRays
    rays_in = RealRays(
        x=ray_x,
        y=ray_y,
        z=np.zeros(n_rays),
        L=np.zeros(n_rays),
        M=np.zeros(n_rays),
        N=np.ones(n_rays),
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, wavelength_um),
    )
    
    # 光线追迹
    from wavefront_to_rays.element_raytracer import ElementRaytracer
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength_um,
    )
    
    rays_out = raytracer.trace(rays_in)
    opd_waves = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    # 修正 OPD 符号（ElementRaytracer 与 PROPER 符号相反）
    opd_waves = -opd_waves
    
    # =========================================================================
    # 步骤 3: 计算参考面相位
    # =========================================================================
    phase_ref = self._compute_reference_phase(wfo, ray_x, ray_y)
    
    # =========================================================================
    # 步骤 4: 计算残差相位
    # =========================================================================
    # OPD 相位
    phase_opd = 2 * np.pi * opd_waves
    
    # 残差相位 = OPD 相位 - 参考面相位
    phase_residual = phase_opd - phase_ref
    
    # 将无效光线的相位设为 0
    phase_residual = np.where(valid_mask, phase_residual, 0.0)
    
    # =========================================================================
    # 步骤 5: 插值到网格并应用
    # =========================================================================
    from scipy.interpolate import griddata
    
    # 只使用有效光线进行插值
    valid_x = ray_x[valid_mask]
    valid_y = ray_y[valid_mask]
    valid_phase = phase_residual[valid_mask]
    
    if len(valid_x) > 3:
        # 创建网格坐标
        half_size_mm = sampling_mm * n / 2
        coords_mm = np.linspace(-half_size_mm, half_size_mm, n)
        X_mm, Y_mm = np.meshgrid(coords_mm, coords_mm)
        
        # 插值
        points = np.column_stack([valid_x, valid_y])
        phase_grid = griddata(
            points,
            valid_phase,
            (X_mm, Y_mm),
            method='cubic',
            fill_value=0.0,
        )
        
        # 处理 NaN 值
        phase_grid = np.nan_to_num(phase_grid, nan=0.0)
        
        # 应用相位
        phase_field = np.exp(1j * phase_grid)
        phase_field_fft = proper.prop_shift_center(phase_field)
        wfo.wfarr = wfo.wfarr * phase_field_fft
```

### 5. 辅助方法

```python
def _get_sampling_half_size_mm(self, wfo) -> float:
    """获取采样面半尺寸
    
    使用 PROPER 网格的完整尺寸作为采样范围，
    避免基于光束强度计算导致的面积收缩问题。
    
    参数:
        wfo: PROPER 波前对象
    
    返回:
        采样面半尺寸（mm）
    """
    import proper
    
    n = proper.prop_get_gridsize(wfo)
    sampling_m = proper.prop_get_sampling(wfo)
    sampling_mm = sampling_m * 1e3
    
    # 使用完整网格尺寸
    half_size_mm = sampling_mm * n / 2
    
    return half_size_mm


def _create_sampling_rays(
    self,
    half_size_mm: float,
) -> Tuple[NDArray, NDArray]:
    """创建采样光线
    
    在整个采样面上创建均匀分布的采样点。
    不基于光束强度限制采样范围，避免面积收缩。
    
    参数:
        half_size_mm: 采样面半尺寸（mm）
    
    返回:
        (ray_x, ray_y): 光线位置数组（mm）
    """
    n_rays_1d = int(np.sqrt(self._hybrid_num_rays))
    ray_coords = np.linspace(-half_size_mm, half_size_mm, n_rays_1d)
    ray_X, ray_Y = np.meshgrid(ray_coords, ray_coords)
    ray_x = ray_X.flatten()
    ray_y = ray_Y.flatten()
    
    return ray_x, ray_y


def _apply_tilt_phase(self, wfo, element) -> None:
    """应用失调倾斜相位
    
    参数:
        wfo: PROPER 波前对象
        element: 光学元件对象
    """
    import proper
    
    n = proper.prop_get_gridsize(wfo)
    sampling_mm = proper.prop_get_sampling(wfo) * 1e3
    wavelength_m = wfo.lamda
    
    half_size = sampling_mm * n / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    
    # 倾斜引入的 OPD
    tilt_opd = (X * np.sin(element.tilt_y) + Y * np.sin(element.tilt_x))
    if element.is_reflective:
        tilt_opd *= 2  # 反射镜 OPD 加倍
    
    k = 2 * np.pi / (wavelength_m * 1e3)  # 1/mm
    tilt_phase = k * tilt_opd
    
    tilt_field = np.exp(1j * tilt_phase)
    tilt_field_fft = proper.prop_shift_center(tilt_field)
    wfo.wfarr = wfo.wfarr * tilt_field_fft
```

## 正确性属性

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*



### Property 1: OPD 符号与参考面变换正确性

*For any* 光学元件和入射光束，光线追迹计算的 OPD 经过符号取反和参考面变换后，应用到波前的残差相位应正确反映元件引入的波前变化。

**Validates: Requirements 1.3, 2.1, 2.2, 2.3, 2.4**

### Property 2: 高斯光束参数更新正确性

*For any* 有效的焦距值和初始高斯光束参数，`_update_gaussian_params_only` 方法更新后的参数（z_w0, w0, z_Rayleigh, beam_type_old, reference_surface）应与 `prop_lens` 的更新结果一致。

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

### Property 3: 抛物面镜 OPD 常数性

*For any* 平行光入射理想抛物面镜，ElementRaytracer 计算的相对 OPD 应为常数（RMS < 0.01 波），因为抛物面镜对轴上平行光无像差。

**Validates: Requirements 4.2**

### Property 4: 凹面镜 OPD 解析验证

*For any* 平行光入射凹面镜，ElementRaytracer 计算的 OPD 应与解析公式一致（相对误差 < 0.1%）。对于球面凹面镜，OPD 应包含正确的球差，与解析公式 `SA = r⁴ / (8 * R³)` 一致。

**Validates: Requirements 4.1, 4.3**

### Property 5: 平面镜 OPD 常数性

*For any* 平行光入射平面镜，ElementRaytracer 计算的相对 OPD 应为常数（RMS < 0.001 波），因为平面镜不改变波前形状。

**Validates: Requirements 4.4**

### Property 6: 45° 折叠镜坐标变换正确性

*For any* 光线入射 45° 折叠镜，ElementRaytracer 应正确处理坐标变换，出射光线方向应与反射定律一致，OPD 分布应保持对称性。

**Validates: Requirements 4.5**

### Property 7: 单元件集成验证

*For any* 高斯光束通过单个聚焦元件（凹面镜、抛物面镜），混合传播模式计算的输出光束半径应与 ABCD 理论一致（相对误差 < 1%）。

**Validates: Requirements 5.1, 5.2**

### Property 8: 多元件系统验证

*For any* 多元件光学系统（如伽利略式扩束镜），混合传播模式计算的放大倍率应与设计值一致（误差 < 1%），各采样面的 WFE RMS 应小于 0.1 波。

**Validates: Requirements 6.1, 6.2, 6.3, 6.4**

### Property 9: 相位采样检查正确性

*For any* 应用到波前的相位网格，当相邻像素间相位差超过 π 时，系统应发出警告并建议增加网格大小或减小光束尺寸。

**Validates: Requirements 7.1, 7.2, 7.3**

## 错误处理

### 相位采样检查

在应用相位到波前之前，检查相位梯度是否过大：

```python
def _check_phase_sampling(self, phase_grid: NDArray, sampling_mm: float) -> None:
    """检查相位采样是否充足
    
    如果相邻像素间相位差超过 π，发出警告。
    
    参数:
        phase_grid: 相位网格（弧度）
        sampling_mm: 采样间隔（mm/pixel）
    """
    import warnings
    
    # 计算相位梯度
    grad_x = np.diff(phase_grid, axis=1)
    grad_y = np.diff(phase_grid, axis=0)
    
    max_grad_x = np.nanmax(np.abs(grad_x))
    max_grad_y = np.nanmax(np.abs(grad_y))
    max_grad = max(max_grad_x, max_grad_y)
    
    if max_grad > np.pi:
        warnings.warn(
            f"相位采样不足：相邻像素间最大相位差为 {max_grad:.2f} 弧度 "
            f"（超过 π = {np.pi:.2f}）。\n"
            f"建议：增加网格大小或减小光束尺寸。",
            UserWarning,
        )
```

### 无效光线处理

当光线追迹返回无效光线时，将其相位设为 0：

```python
# 将无效光线的相位设为 0
phase_residual = np.where(valid_mask, phase_residual, 0.0)
```

### 插值失败处理

当有效光线数量不足以进行插值时，跳过相位应用：

```python
if len(valid_x) <= 3:
    warnings.warn(
        f"有效光线数量不足（{len(valid_x)} 条），无法进行相位插值。",
        UserWarning,
    )
    return
```

## 测试策略

### 单元测试

1. **`_update_gaussian_params_only` 方法测试**
   - 测试与 `prop_lens` 参数更新结果的一致性
   - 测试不同焦距值（正、负、无穷大）
   - 测试不同初始光束状态（INSIDE_, OUTSIDE）

2. **`_compute_reference_phase` 方法测试**
   - 测试 PLANAR 参考面（应返回零）
   - 测试 SPHERI 参考面（应返回二次相位）
   - 测试不同 z_w0 值

3. **`_check_phase_sampling` 方法测试**
   - 测试正常相位梯度（不应警告）
   - 测试过大相位梯度（应警告）

### 属性测试

使用 hypothesis 库验证上述正确性属性：

```python
from hypothesis import given, strategies as st

@given(
    focal_length=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False).filter(lambda x: abs(x) > 10),
    w0=st.floats(min_value=0.1, max_value=10.0),
    wavelength=st.floats(min_value=0.4, max_value=1.0),
)
def test_gaussian_params_update_consistency(focal_length, w0, wavelength):
    """
    **Feature: hybrid-propagation-raytracing-opd, Property 2: 高斯光束参数更新正确性**
    
    验证 _update_gaussian_params_only 与 prop_lens 的参数更新结果一致
    """
    # 创建两个相同的波前对象
    # 一个使用 prop_lens，一个使用 _update_gaussian_params_only
    # 比较更新后的参数
    pass
```

### 集成测试

1. **单元件测试**
   - 凹面镜：验证光束半径与 ABCD 一致
   - 抛物面镜：验证 WFE RMS < 0.1 波
   - 平面镜：验证 WFE RMS < 0.01 波
   - 45° 折叠镜：验证光束方向和 WFE

2. **多元件测试**
   - `galilean_oap_expander.py`：验证放大倍率和 WFE
   - 多折叠光路：验证各采样面的 WFE

### 验证测试

与 ABCD 理论结果对比：

```python
def test_hybrid_vs_abcd_single_mirror():
    """验证混合模式与 ABCD 理论的一致性"""
    # 创建简单的单镜系统
    # 运行混合传播模式
    # 计算 ABCD 理论结果
    # 比较光束半径，误差应 < 1%
    pass
```

## 实现注意事项

### 1. OPD 符号约定

ElementRaytracer 的 OPD 符号与 PROPER 相反：
- ElementRaytracer：正 OPD 表示光程增加
- PROPER：正相位表示光程减少

因此需要取反：`opd_waves = -raytracer.get_relative_opd_waves()`

### 2. 坐标系统

- ElementRaytracer 使用 mm 单位
- PROPER 使用 m 单位
- 需要正确进行单位转换

### 3. 参考面更新时机

高斯光束参数必须在计算参考面相位之前更新，因为参考面相位依赖于更新后的 z_w0。

### 4. 插值方法

使用 `scipy.interpolate.griddata` 的 `cubic` 方法进行插值，`fill_value=0.0` 处理边界外的点。

### 5. 采样策略：整个采样面 vs 有效光束区域

**设计决策**：在整个 PROPER 网格范围内进行光线采样，而不是仅在有效光束区域内采样。

**原因**：
- 如果基于当前光束强度分布计算"有效光束区域"，在多元件系统中会导致采样面积逐渐收缩
- 高斯光束经过聚焦元件后，光束半径会变化，但相位信息需要在整个网格上正确计算
- 使用固定的网格尺寸确保了相位计算的一致性和完整性
- 光束边缘以外的区域振幅接近零，相位误差不会影响最终结果

**实现方式**：
- 采样半径 = PROPER 网格半尺寸 = `sampling_mm * n / 2`
- 在整个方形网格上均匀采样，不进行圆形裁剪
- 无效光线（如被遮挡）的相位设为 0

## 与现有代码的关系

### 复用的模块

- `ElementRaytracer`：完全复用，不需要修改
- `_update_proper_gaussian_params`：可以复用其逻辑，但需要分离参数更新和相位计算

### 修改的模块

- `SequentialOpticalSystem._apply_element_hybrid`：重构核心逻辑

### 新增的方法

- `_update_gaussian_params_only`：仅更新参数，不修改 wfarr
- `_compute_reference_phase`：计算参考面相位
- `_check_phase_sampling`：检查相位采样
- `_get_sampling_half_size_mm`：获取采样面半尺寸
- `_create_sampling_rays`：在整个采样面上创建采样光线
