# 设计文档：Pilot Beam 参考相位（简化版）

## 概述

本设计文档描述简化后的 Pilot Beam 参考相位计算方案。

**核心简化**：由于初始高斯光束参数已知，使用 ABCD 矩阵法追踪光束在整个系统中的演变，无需从波前拟合估计光束参数（不需要 BeamEstimator 模块）。

## 设计理念

### 为什么不需要 BeamEstimator？

在原设计中，BeamEstimator 用于从波前振幅和相位分布估计高斯光束参数。但这种方法存在以下问题：

1. **拟合误差**：从实际波前拟合高斯参数会引入误差
2. **非理想光束**：实际波前可能不是完美的高斯分布
3. **计算开销**：拟合过程需要额外的计算

**简化方案**：
- 初始高斯光束参数（w0, z0, wavelength, M²）在系统定义时已知
- 使用 ABCD 矩阵法可以精确计算光束在任意位置的参数
- 在每个元件的入射面和出射面都可以直接获取光束参数

## 架构

### 复用现有模块

| 现有模块 | 位置 | 复用方式 |
|---------|------|---------|
| GaussianBeam | src/gaussian_beam_simulation/gaussian_beam.py | 高斯光束参数定义 |
| ABCDCalculator | src/gaussian_beam_simulation/abcd_calculator.py | ABCD 矩阵计算（已扩展） |
| PilotBeamValidator | src/hybrid_propagation/pilot_beam.py | 验证功能保留 |

### 已删除/不需要的模块

| 模块 | 原因 |
|------|------|
| BeamEstimator | 不需要从波前拟合估计参数 |

### 修改的模块

```
src/gaussian_beam_simulation/
├── abcd_calculator.py      # 扩展：添加 get_beam_at_element() 等方法

src/hybrid_propagation/
├── pilot_beam.py           # 重构：PilotBeamCalculator 使用 ABCD 方法
```

## 数据流

```
初始高斯光束参数 (w0, z0, wavelength, M²)
                │
                ▼
        [ABCDCalculator]
                │
                ├──► 元件 0 入射面光束参数 (w, R, q)
                │           │
                │           ▼
                │    [ABCD 矩阵变换]
                │           │
                │           ▼
                ├──► 元件 0 出射面光束参数 (w', R', q')
                │
                ├──► 元件 1 入射面光束参数
                │           │
                │           ▼
                ...
                │
                ▼
        [参考相位计算]
                │
                ▼
        φ_ref(x, y) = k * r² / (2 * R)
```

## 核心算法

### 1. ABCD 矩阵追踪

ABCDCalculator 已扩展以下方法：

```python
class ABCDCalculator:
    def get_beam_at_element(
        self,
        element_index: int,
        position: str = 'entrance',  # 'entrance' 或 'exit'
    ) -> ABCDResult:
        """获取指定元件入射面或出射面的光束参数"""
        ...
    
    def get_all_element_beam_params(self) -> List[Dict[str, ABCDResult]]:
        """获取所有元件入射面和出射面的光束参数"""
        ...
    
    def compute_reference_phase_at_position(
        self,
        x: NDArray,
        y: NDArray,
        element_index: int,
        position: str = 'exit',
    ) -> NDArray:
        """在指定位置计算高斯光束参考相位"""
        ...
    
    def compute_reference_phase_grid(
        self,
        grid_size: int,
        physical_size: float,
        element_index: int,
        position: str = 'exit',
    ) -> NDArray:
        """在网格上计算高斯光束参考相位"""
        ...
```

### 2. 参考相位计算

高斯光束相位公式：
```
φ(r) = k * r² / (2 * R)
```

其中：
- k = 2π/λ 是波数
- r² = x² + y² 是径向距离平方
- R 是波前曲率半径（由 ABCD 法则计算）

### 3. PilotBeamCalculator 简化接口

```python
class PilotBeamCalculator:
    def __init__(
        self,
        beam: GaussianBeam,           # 初始高斯光束
        elements: List[OpticalElement], # 光学元件列表
        initial_distance: float = 0.0,  # 到第一个元件的距离
        grid_size: int = 64,
        physical_size: float = 20.0,
    ):
        # 内部创建 ABCDCalculator
        self._abcd_calculator = ABCDCalculator(beam, elements, initial_distance)
    
    def compute_reference_phase(
        self,
        x_coords: NDArray,
        y_coords: NDArray,
        element_index: int = 0,
        position: str = 'exit',
    ) -> NDArray:
        """计算参考相位"""
        return self._abcd_calculator.compute_reference_phase_at_position(
            x_coords, y_coords, element_index, position
        )
    
    def compute_reference_phase_at_rays(
        self,
        ray_x: NDArray,
        ray_y: NDArray,
        element_index: int = 0,
        position: str = 'exit',
    ) -> NDArray:
        """在光线位置计算参考相位"""
        return self.compute_reference_phase(ray_x, ray_y, element_index, position)
```

## 正确性属性

### Property 18: ABCD 变换正确性

*For any* 有效的输入光束参数和光学元件，ABCD 变换后的输出参数应满足：
- 能量守恒（|q_out| 与 |q_in| 的关系正确）
- 与解析解一致（对于简单元件如平面镜、球面镜）

**Validates: Requirements 1.2, 1.5**

### Property 19: 参考相位连续性

*For any* 计算得到的参考相位网格，相邻像素间的相位差应 < π（无相位跳变）。

**Validates: Requirements 1.3, 2.3**

## 测试策略

### 单元测试

1. **ABCDCalculator 扩展方法测试**
   - 测试 get_beam_at_element() 入射面和出射面
   - 测试 compute_reference_phase_at_position()
   - 测试 compute_reference_phase_grid()

2. **PilotBeamCalculator 测试**
   - 测试与 ABCDCalculator 的集成
   - 测试参考相位计算

### 属性测试

使用 hypothesis 库验证上述正确性属性。

## 优势

1. **精确性**：无拟合误差，参数精确追踪
2. **简单性**：代码更简洁，无需复杂的拟合算法
3. **性能**：ABCD 矩阵计算非常快
4. **可维护性**：减少了模块数量和代码复杂度

## 限制

1. **假设理想高斯光束**：实际光束可能有像差
2. **不支持非高斯光束**：如平顶光束、涡旋光束等

对于非理想情况，可以在残差相位中体现。
