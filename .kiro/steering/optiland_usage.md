<!------------------------------------------------------------------------------------
# optiland 库使用规范

inclusion: fileMatch
fileMatchPattern: '**/raytracing/**,**/optiland/**,**/*raytrac*,**/*opd*'
------------------------------------------------------------------------------------>

## 调用方式

```python
from optiland.optic import Optic      # 注意：从子模块导入
from optiland.wavefront import Wavefront
```

**不要**：`from optiland import Optic`（会报错）

## 核心概念

- `Optic` 类表示完整的光学系统
- `Wavefront` 模块计算 OPD（单位：波长数）

## ⚠️ 已知问题

- `GridPhaseProfile` 覆盖整个正方形网格，需要圆形光瞳时必须显式应用掩模
- `GridPhaseProfile` 计算 OPD 时存在 1000 倍放大问题
