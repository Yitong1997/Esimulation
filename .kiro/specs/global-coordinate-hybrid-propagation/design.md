# 设计文档：全局坐标系混合传播

## 概述

本设计文档描述了全局坐标系混合传播器（`GlobalElementRaytracer` 和 `HybridElementPropagatorGlobal`）的技术实现方案。

### 设计目标

1. **完全复用现有计算逻辑**：与 `ElementRaytracer` 使用相同的 OPD 计算、抛物面反射修正等核心算法
2. **全局坐标系操作**：避免局部坐标系转换，直接在全局坐标系中定义入射面、出射面和光学表面
3. **正确处理 tilt 参数**：当前 `ElementRaytracer` 对抛物面忽略了 `tilt_x/y`，全局版本应正确处理
4. **简化 optiland 集成**：利用 optiland 的绝对坐标定位能力

### 与现有实现的关系

| 方面 | ElementRaytracer | GlobalElementRaytracer |
|------|------------------|------------------------|
| 坐标系 | 入射面局部坐标系 | 全局坐标系 |
| 表面定义 | 相对于入射面偏移 | 绝对位置 (x, y, z) |
| 入射面 | 隐式定义（原点） | 显式定义（点+法向量） |
| 出射面 | 通过旋转矩阵转换 | 显式定义（点+法向量） |
| OPD 计算 | 带符号 OPD | 带符号 OPD（相同算法） |
| 抛物面处理 | 忽略 tilt 参数 | 正确处理 tilt 参数 |

---

## 架构

### 系统架构图

```mermaid
graph TB
    subgraph "HybridElementPropagatorGlobal"
        A[入射波前] --> B[WavefrontToRaysSampler]
        B --> C[光线坐标转换<br/>局部→全局]
        C --> D[GlobalElementRaytracer]
        D --> E[光线坐标转换<br/>全局→局部]
        E --> F[RayToWavefrontReconstructor]
        F --> G[出射波前]
    end
    
    subgraph "GlobalElementRaytracer"
        D1[全局入射面定义] --> D2[optiland 光学系统<br/>绝对坐标]
        D2 --> D3[带符号 OPD 追迹]
        D3 --> D4[抛物面反射修正]
        D4 --> D5[出射面投影]
    end
```

### 数据流

```
入射波前（PROPER）
    ↓
StateConverter.proper_to_amplitude_phase()
    ↓
WavefrontToRaysSampler（入射面局部坐标系）
    ↓
坐标转换：局部 → 全局
    ↓
GlobalElementRaytracer.trace()（全局坐标系）
    ↓
坐标转换：全局 → 出射面局部
    ↓
计算残差 OPD = 绝对 OPD + Pilot Beam OPD
    ↓
RayToWavefrontReconstructor（出射面局部坐标系）
    ↓
StateConverter.amplitude_phase_to_proper()
    ↓
出射波前（PROPER）
```

---

## 组件与接口

### GlobalElementRaytracer 类

```python
class GlobalElementRaytracer:
    """全局坐标系元件光线追迹器
    
    在全局坐标系中进行光线追迹，避免局部坐标系转换。
    完全复用 ElementRaytracer 的计算逻辑。
    """
    
    def __init__(
        self,
        surfaces: List[GlobalSurfaceDefinition],
        wavelength: float,
        entrance_plane: PlaneDef,
        exit_plane: Optional[PlaneDef] = None,
    ) -> None:
        """初始化全局坐标系光线追迹器
        
        参数:
            surfaces: 光学表面定义列表（全局坐标）
            wavelength: 波长 (μm)
            entrance_plane: 入射面定义（点+法向量）
            exit_plane: 出射面定义（可选，自动计算）
        """
        ...
    
    def trace_chief_ray(self) -> Tuple[float, float, float]:
        """追迹主光线，计算出射方向和交点位置
        
        返回:
            出射主光线方向 (L, M, N)，全局坐标系
        """
        ...
    
    def trace(self, input_rays: RealRays) -> RealRays:
        """执行光线追迹
        
        参数:
            input_rays: 输入光线（全局坐标系）
        
        返回:
            出射光线（全局坐标系）
        """
        ...
```

### PlaneDef 数据类

```python
@dataclass
class PlaneDef:
    """平面定义（全局坐标系）
    
    用于定义入射面和出射面。
    """
    position: Tuple[float, float, float]  # 平面上的点 (x, y, z)
    normal: Tuple[float, float, float]    # 法向量 (nx, ny, nz)，必须归一化
```


### GlobalSurfaceDefinition 数据类

```python
@dataclass
class GlobalSurfaceDefinition:
    """全局坐标系表面定义
    
    与 SurfaceDefinition 类似，但使用全局坐标定义位置和朝向。
    """
    surface_type: str  # 'mirror' 或 'refract'
    radius: float      # 曲率半径 (mm)
    conic: float = 0.0 # 圆锥常数
    material: str = 'mirror'
    
    # 全局坐标定位
    vertex_position: Tuple[float, float, float] = (0, 0, 0)  # 顶点位置
    surface_normal: Tuple[float, float, float] = (0, 0, -1)  # 表面法向量
    
    # 旋转角度（可选，用于倾斜表面）
    tilt_x: float = 0.0  # 绕 X 轴旋转 (rad)
    tilt_y: float = 0.0  # 绕 Y 轴旋转 (rad)
    tilt_z: float = 0.0  # 绕 Z 轴旋转 (rad)
```

### HybridElementPropagatorGlobal 类

```python
class HybridElementPropagatorGlobal:
    """全局坐标系混合元件传播器
    
    使用 GlobalElementRaytracer 进行光线追迹，
    波前采样和重建在各自平面的局部坐标系中进行。
    """
    
    def __init__(
        self,
        wavelength_um: float,
        num_rays: int = 200,
    ) -> None:
        """初始化
        
        参数:
            wavelength_um: 波长 (μm)
            num_rays: 光线采样数量
        """
        ...
    
    def propagate(
        self,
        state: PropagationState,
        surface: GlobalSurfaceDefinition,
        entrance_axis: OpticalAxisState,
        exit_axis: OpticalAxisState,
        target_surface_index: int,
    ) -> PropagationState:
        """执行混合元件传播
        
        参数:
            state: 入射面传播状态
            surface: 表面定义
            entrance_axis: 入射光轴状态
            exit_axis: 出射光轴状态
            target_surface_index: 目标表面索引
        
        返回:
            出射面传播状态
        """
        ...
```

---

## 数据模型

### 坐标系定义

#### 全局坐标系

- **原点**：光学系统的参考原点
- **Z 轴**：初始光轴方向（+Z 为光传播方向）
- **Y 轴**：垂直向上
- **X 轴**：由右手定则确定

#### 入射面局部坐标系

- **原点**：主光线与入射面的交点
- **Z 轴**：入射主光线方向
- **X, Y 轴**：垂直于 Z 轴，保持右手系

#### 出射面局部坐标系

- **原点**：主光线与出射面的交点
- **Z 轴**：出射主光线方向
- **X, Y 轴**：垂直于 Z 轴，保持右手系

### 坐标转换矩阵

从局部坐标系到全局坐标系的转换：

```
v_global = R @ v_local + origin
```

其中 `R` 是 3×3 旋转矩阵，`origin` 是局部坐标系原点在全局坐标系中的位置。

旋转矩阵的列向量为局部坐标轴在全局坐标系中的表示：

```
R = [x_local | y_local | z_local]
```

### OPD 计算模型

#### 带符号 OPD 计算

与 `ElementRaytracer` 相同的算法：

```python
# 符号计算：sign(t) = sign(dz) * sign(N_before)
sign_t = np.sign(dz) * np.sign(N_before)
opd_increment_signed = sign_t * opd_increment_abs
```

#### 残差 OPD 计算

```python
# 残差 OPD = 绝对 OPD + Pilot Beam OPD
# 注意：是加法，因为符号约定
residual_opd_waves = absolute_opd_waves + pilot_opd_waves
```

### 抛物面反射模型

对于抛物面反射镜，反射方向直接指向焦点：

```python
# 焦点位置（全局坐标系）
focus = vertex + np.array([0, 0, focal_length])

# 反射方向
P_to_F = focus - intersection_point
reflection_dir = P_to_F / np.linalg.norm(P_to_F)
```


---

## 正确性属性

*正确性属性是系统在所有有效执行中应保持为真的特征或行为——本质上是关于系统应该做什么的形式化陈述。属性作为人类可读规范和机器可验证正确性保证之间的桥梁。*

### Property 1: 法向量归一化验证

*For any* 输入的法向量，如果其模长不等于 1（在容差范围内），系统应抛出 ValueError；如果模长等于 1，系统应正常接受。

**Validates: Requirements 1.2, 1.3, 10.1**

### Property 2: 方向余弦归一化保持

*For any* 光线经过坐标转换（局部→全局或全局→局部），转换后的方向余弦应满足 L² + M² + N² = 1（在数值精度范围内）。

**Validates: Requirements 5.2, 10.4**

### Property 3: 旋转矩阵正交性

*For any* 计算得到的旋转矩阵 R，应满足：
- R × R^T = I（正交性）
- det(R) = 1（右手系）

**Validates: Requirements 5.3, 6.3**

### Property 4: 坐标转换可逆性

*For any* 光线位置和方向，从局部坐标系转换到全局坐标系再转换回局部坐标系，应得到原始值（在数值精度范围内）。

**Validates: Requirements 5.1, 6.1**


### Property 5: OPD 坐标转换不变性

*For any* 光线的 OPD 值，在坐标转换（局部→全局或全局→局部）过程中应保持不变，因为 OPD 是标量。

**Validates: Requirements 6.2**

### Property 6: 主光线 OPD 为零

*For any* 光线追迹结果，主光线（最接近光轴的光线）的相对 OPD 应为 0。

**Validates: Requirements 7.1**

### Property 7: 残差 OPD 计算公式

*For any* 出射光线，残差 OPD = 绝对 OPD + Pilot Beam OPD。对于理想球面镜，残差 OPD 应接近 0。

**Validates: Requirements 7.2, 7.3**

### Property 8: 抛物面反射方向指向焦点

*For any* 平行于光轴入射到抛物面的光线，反射后的方向应指向焦点。

**Validates: Requirements 4.5, 9.2**

### Property 9: 平面镜传输精度

*For any* 倾斜角度（0°-60°）的平面镜，混合传播后的相位 RMS 误差应小于 1 milli-wave。

**Validates: Requirements 11.5**

### Property 10: 与现有实现结果一致性

*For any* 相同的输入（表面定义、光源参数），GlobalElementRaytracer 和 ElementRaytracer 应产生相同的 OPD 结果（在数值精度范围内）。

**Validates: Requirements 11.3**


---

## 错误处理

### 输入验证错误

| 错误条件 | 异常类型 | 错误信息 |
|----------|----------|----------|
| 法向量未归一化 | ValueError | "法向量未归一化：|n| = {value}，期望为 1.0" |
| 方向余弦未归一化 | ValueError | "方向余弦未归一化：L² + M² + N² = {value}" |
| 表面列表为空 | ValueError | "surfaces 列表不能为空" |
| 波长非正值 | ValueError | "wavelength 必须为正值" |
| 无效表面类型 | ValueError | "无效的表面类型：'{type}'" |

### 运行时错误

| 错误条件 | 异常类型 | 错误信息 |
|----------|----------|----------|
| 所有光线无效 | SimulationError | "所有光线都无效，无法进行追迹" |
| 光线与平面平行 | SimulationError | "光线与平面平行，无法计算交点" |
| 主光线追迹失败 | RuntimeError | "主光线追迹失败" |

### 错误恢复策略

1. **输入验证错误**：立即抛出异常，不进行任何计算
2. **部分光线无效**：标记无效光线，继续处理有效光线
3. **数值问题**：使用容差处理，避免除零错误

---

## 测试策略

### 双重测试方法

本设计采用单元测试和属性测试相结合的方法：

- **单元测试**：验证特定示例、边界情况和错误条件
- **属性测试**：验证所有输入的通用属性

### 属性测试配置

- **测试框架**：hypothesis
- **最小迭代次数**：100 次/属性
- **标签格式**：`Feature: global-coordinate-hybrid-propagation, Property {N}: {property_text}`

### 测试用例分类

#### 单元测试

1. **构造函数测试**
   - 有效参数创建对象
   - 无效参数抛出异常
   - 边界值处理

2. **坐标转换测试**
   - 恒等变换（无旋转、无平移）
   - 90° 旋转
   - 任意旋转和平移组合

3. **光线追迹测试**
   - 平面镜正入射
   - 平面镜倾斜入射
   - 球面镜
   - 抛物面镜（轴上和离轴）


#### 属性测试

1. **Property 1: 法向量归一化验证**
   - 生成随机向量（归一化和非归一化）
   - 验证系统正确接受/拒绝

2. **Property 2: 方向余弦归一化保持**
   - 生成随机归一化方向
   - 应用坐标转换
   - 验证结果仍然归一化

3. **Property 3: 旋转矩阵正交性**
   - 生成随机方向向量
   - 计算旋转矩阵
   - 验证正交性和行列式

4. **Property 4: 坐标转换可逆性**
   - 生成随机位置和方向
   - 应用正向和逆向转换
   - 验证结果与原始值一致

5. **Property 5: OPD 坐标转换不变性**
   - 生成随机 OPD 值
   - 应用坐标转换
   - 验证 OPD 不变

6. **Property 9: 平面镜传输精度**
   - 生成随机倾斜角度（0°-60°）
   - 执行混合传播
   - 验证 RMS < 1 milli-wave

### 集成测试

1. **与现有实现对比**
   - 使用相同输入运行两个实现
   - 比较 OPD 结果
   - 验证差异在数值精度范围内

2. **端到端精度测试**
   - 使用标准测试文件（倾斜平面镜、OAP）
   - 验证精度达到规格要求

### 回归测试

修改以下文件时必须运行回归测试：
- `src/wavefront_to_rays/global_element_raytracer.py`
- `src/hybrid_optical_propagation/hybrid_element_propagator_global.py`

回归测试文件：
- `tests/integration/不同倾斜角度平面镜传输误差标准测试文件.py`
- `tests/integration/离轴抛物面镜传输误差标准测试文件.py`
