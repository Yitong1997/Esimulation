---
trigger: model_decision
description: when using PROPER to preform physical optical propagate
---


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
- **自动选择**：`prop_propagate` 会根据距离自动选择合适的算法
