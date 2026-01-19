<!------------------------------------------------------------------------------------
# PROPER 物理光学库使用规范

本文件定义了 PROPER 库在混合光学仿真项目中的使用规范。
inclusion: fileMatch
fileMatchPattern: '**/propagation/**,**/proper/**,**/*propagat*'
-------------------------------------------------------------------------------------> 

## ⚠️ 重要：库的调用方式

**PROPER (PyPROPER3) 已通过 pip 安装，必须从安装的包导入，而不是直接引用工作区内的源码！**

```python
# ✅ 正确：从 pip 安装的包导入
import proper

# 使用 PROPER 函数
wfo = proper.prop_begin(beam_diameter, wavelength, grid_size)
proper.prop_propagate(wfo, distance)
psf, sampling = proper.prop_end(wfo)

# ❌ 错误：不要直接引用工作区内的源码
# sys.path.insert(0, 'proper_v3.3.4_python/proper')
```

工作区内的 `proper_v3.3.4_python/` 文件夹仅作为 API 文档参考：
- 查看 `proper_v3.3.4_python/PROPER_manual_v3.3.4.pdf` 了解完整文档
- 查看 `proper_v3.3.4_python/proper/examples/` 了解使用示例

## PROPER 核心概念

### 波前对象 (Wavefront Object)
PROPER 使用复数数组表示波前，包含振幅和相位信息：
```python
# 波前 = 振幅 × exp(i × 相位)
# 存储为复数数组：wfo.wfarr
```

### 传播模式
- **近场传播 (Near-field)**：Fresnel 传播，适用于短距离
- **远场传播 (Far-field)**：Fraunhofer 传播，适用于焦平面
- **自动选择**：`prop_propagate` 会根据距离自动选择合适的算法

## 常用函数参考

### 初始化
```python
import proper

# 初始化波前
wfo = proper.prop_begin(
    beam_diameter,    # 光束直径 (m)
    wavelength,       # 波长 (m)
    grid_size,        # 网格大小 (像素)
    beam_ratio=0.5    # 光束直径与网格的比例
)
```

### 传播
```python
# 传播到指定距离
proper.prop_propagate(wfo, distance)  # distance 单位：m

# 传播到焦点
proper.prop_propagate(wfo, focal_length, TO_PLANE=True)
```

### 光学元件
```python
# 理想透镜
proper.prop_lens(wfo, focal_length)

# 圆形光阑
proper.prop_circular_aperture(wfo, radius)

# 圆形遮挡
proper.prop_circular_obscuration(wfo, radius)

# 矩形光阑
proper.prop_rectangular_aperture(wfo, width, height)
```

### 相位操作
```python
# 添加 Zernike 像差
proper.prop_zernikes(wfo, zernike_coeffs)

# 添加自定义相位
proper.prop_add_phase(wfo, phase_array)

# 从文件读取相位误差图
proper.prop_errormap(wfo, filename)
```

### 获取结果
```python
# 获取振幅
amplitude = proper.prop_get_amplitude(wfo)

# 获取相位
phase = proper.prop_get_phase(wfo)

# 获取完整波前
wavefront = proper.prop_get_wavefront(wfo)

# 获取采样间隔
sampling = proper.prop_get_sampling(wfo)

# 结束并获取 PSF
(psf, sampling) = proper.prop_end(wfo)
```

## 与 optiland 接口的注意事项

### 相位单位转换
```python
# PROPER 相位单位：弧度
# optiland OPD 单位：波长数或长度

def opd_to_phase(opd_waves, wavelength):
    """将 OPD（波长数）转换为相位（弧度）"""
    return 2 * np.pi * opd_waves

def opd_meters_to_phase(opd_meters, wavelength):
    """将 OPD（米）转换为相位（弧度）"""
    return 2 * np.pi * opd_meters / wavelength
```

### 网格对齐
```python
# 确保 optiland 计算的 OPD 网格与 PROPER 波前网格对齐
# 可能需要插值重采样

from scipy.interpolate import RectBivariateSpline

def resample_opd(opd, old_grid, new_grid):
    """重采样 OPD 到新网格"""
    interp = RectBivariateSpline(old_grid, old_grid, opd)
    return interp(new_grid, new_grid)
```

### 光瞳坐标
```python
# PROPER 使用物理坐标（米）
# 需要将 optiland 的归一化光瞳坐标转换为物理坐标

def normalized_to_physical(normalized_coords, pupil_radius):
    """归一化坐标转物理坐标"""
    return normalized_coords * pupil_radius
```

## 性能优化

### 使用 FFTW
```python
# 启用 FFTW 加速（需要安装 pyfftw）
proper.prop_use_fftw()

# 加载预计算的 FFTW wisdom
proper.prop_load_fftw_wisdom(filename)
```

### 网格大小选择
- 网格大小应为 2 的幂次（如 512, 1024, 2048）以优化 FFT
- 确保采样满足 Nyquist 准则
- 较大的网格提供更高精度但计算更慢

## 常见问题

### 采样不足
- 症状：PSF 出现混叠
- 解决：增加网格大小或减小 beam_ratio

### 边界效应
- 症状：波前边缘出现伪影
- 解决：使用更大的网格，确保光束不接触边界

### 相位包裹
- 症状：相位图出现不连续
- 解决：使用相位解包裹算法或减小像差量
