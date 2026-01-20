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

## 注意事项

- 网格采样需满足 Nyquist 准则（相邻像素相位差 < π）
- 网格大小应为 2 的幂次以优化 FFT
- **不要向波前添加整体倾斜相位**
