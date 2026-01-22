<!------------------------------------------------------------------------------------
# PROPER 库使用规范

inclusion: fileMatch
fileMatchPattern: '**/propagation/**,**/proper/**,**/*propagat*'
------------------------------------------------------------------------------------>

## 常用函数

```python
import proper

# 初始化
wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_ratio)

# 传播（distance 单位：m）
proper.prop_propagate(wfo, distance)

# 光学元件
proper.prop_lens(wfo, focal_length)
proper.prop_circular_aperture(wfo, radius)

# 获取结果
amplitude = proper.prop_get_amplitude(wfo)
phase = proper.prop_get_phase(wfo)
sampling = proper.prop_get_sampling(wfo)
(psf, sampling) = proper.prop_end(wfo)
```

---

## 波前初始化

### prop_begin 参数说明

```python
wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_ratio)
```

| 参数 | 说明 | 单位 |
|------|------|------|
| `beam_diameter` | 光束直径（1/e² 强度直径） | m |
| `wavelength` | 波长 | m |
| `grid_size` | 网格大小（应为 2 的幂次） | pixels |
| `beam_ratio` | 光束直径占网格宽度的比例（默认 0.5） | - |

### ⚠️ 关键：prop_begin 初始化的是均匀平面波

**`prop_begin` 创建的波前是均匀振幅（=1）、零相位的平面波，不是高斯光束。**

- PROPER 内部追踪一个理想高斯光束的参考球面（Pilot Beam）
- 但 `wfarr` 数组初始值是均匀的 `1+0j`
- 如需高斯振幅分布，必须手动添加

### 创建高斯光束

```python
import proper
import numpy as np

wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, 0.5)

# 创建高斯振幅分布
sampling = proper.prop_get_sampling(wfo)
n = grid_size
x = (np.arange(n) - n // 2) * sampling
X, Y = np.meshgrid(x, x)
w0 = beam_diameter / 2  # 束腰半径

gaussian_amplitude = np.exp(-(X**2 + Y**2) / w0**2)
proper.prop_multiply(wfo, gaussian_amplitude)
```

---

## 直接操作 wfarr 属性

### ⚠️ 关键：可以直接赋值，但必须使用 prop_shift_center

**`wfo.wfarr` 属性支持直接赋值，但 PROPER 内部使用 FFT-shifted 格式存储。**

```python
# ✅ 正确：使用 prop_shift_center
wfo.wfarr = proper.prop_shift_center(my_complex_array)

# ❌ 错误：直接赋值会导致后续传播结果错误
wfo.wfarr = my_complex_array
```

### 存储格式说明

| 格式 | 数组中心位置 | 使用场景 |
|------|-------------|----------|
| 用户空间 | `(n//2, n//2)` | 构造、显示、分析 |
| PROPER 内部 | `(0, 0)` | `wfarr` 存储、FFT 计算 |

`prop_shift_center()` 在两种格式之间转换。

### 完整示例：自定义复振幅

```python
import proper
import numpy as np

# 初始化
wfo = proper.prop_begin(0.01, 632.8e-9, 512, 0.5)
sampling = proper.prop_get_sampling(wfo)
n = 512

# 创建坐标（用户空间，中心在 n//2）
x = (np.arange(n) - n // 2) * sampling
X, Y = np.meshgrid(x, x)
w0 = 0.005  # 束腰半径

# 构造复振幅（用户空间）
amplitude = np.exp(-(X**2 + Y**2) / w0**2)
phase_error_m = np.random.randn(n, n) * 10e-9  # 10 nm RMS
phase_rad = 2 * np.pi / 632.8e-9 * phase_error_m

complex_amplitude = amplitude * np.exp(1j * phase_rad)

# 赋值（必须 shift_center）
wfo.wfarr = proper.prop_shift_center(complex_amplitude.astype(np.complex128))
```

### 乘法操作的两种方式

```python
# 方式 1：使用函数（自动处理 shift）
proper.prop_multiply(wfo, my_array)

# 方式 2：直接乘法（需要手动 shift）
wfo.wfarr *= proper.prop_shift_center(my_array)
```

---

## 添加误差的方法

### 相位误差

```python
# 方式 1：prop_add_phase（单位：米）
proper.prop_add_phase(wfo, phase_error_meters)

# 方式 2：直接乘法
phase_rad = 2 * np.pi / wavelength * phase_error_meters
wfo.wfarr *= proper.prop_shift_center(np.exp(1j * phase_rad))
```

### Zernike 像差

```python
# 索引使用 Noll 顺序，系数单位：米（RMS）
proper.prop_zernikes(
    wfo,
    np.array([4, 5, 11]),           # Defocus, 45° Astig, Spherical
    np.array([50e-9, 20e-9, 30e-9])  # RMS 系数
)
```

### PSD 误差

```python
# amp: 低频功率 (m^4), b: 相关长度 (cycles/m), c: 高频衰减指数
proper.prop_psd_errormap(wfo, amp=1e-22, b=100, c=3)
```

### 从文件加载

```python
# 镜面误差（自动乘以 -2）
proper.prop_errormap(wfo, 'mirror.fits', SAMPLING=1e-4, MIRROR_SURFACE=True)

# 波前误差
proper.prop_errormap(wfo, 'wavefront.fits', SAMPLING=1e-4, WAVEFRONT=True)
```

---

## 注意事项

- 网格采样需满足 Nyquist 准则（相邻像素相位差 < π）
- 网格大小应为 2 的幂次以优化 FFT
- **不要向波前添加整体倾斜相位**
- 直接赋值 `wfarr` 时必须使用 `prop_shift_center()`
- 数据类型必须是 `np.complex128`
