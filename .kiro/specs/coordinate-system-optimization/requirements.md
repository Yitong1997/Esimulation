# POP 包重构需求文档

## 项目目标

基于现有 BTS API 的核心逻辑，构建一个全新的独立代码包：**POP（Physical Optical Propagation）**。

POP 包将：
- 完全脱离 BTS API 的依赖
- 直接调用 `optiland-master/` 和 `proper_v3.3.4_python/` 目录下的代码
- 将必要的旧代码重构或复制进 POP 包，实现完全自包含
- 保持物理正确性和数值精度，同时大幅简化代码结构

## 核心设计原则

### 1. 代码简洁性

- **最小化代码量**：每个模块只做一件事，避免过度抽象
- **扁平化结构**：减少嵌套层级，优先使用函数而非深层类继承
- **直接调用**：直接使用 optiland/proper API，不做无意义的封装
- **删除冗余**：移除所有未使用的代码路径和过度设计的接口

### 2. 垂直接入

- **单一入口**：用户只需 `import pop` 即可使用全部功能
- **零配置启动**：合理的默认值，无需复杂初始化
- **渐进式复杂度**：简单场景简单用，复杂场景才需要深入配置
- **自文档化 API**：函数签名即文档，参数名即说明

### 3. 物理正确性优先

- **保留所有物理计算细节**：不因简化代码而丢失精度
- **符号约定一致**：曲率半径、OPD、相位的符号约定全局统一
- **单位显式化**：所有物理量的单位在变量名或文档中明确标注
- **边界条件完备**：无穷大曲率、零厚度、垂直入射等特殊情况正确处理


## 依赖项清理策略

### 直接依赖（保留）

| 依赖 | 路径 | 用途 |
|------|------|------|
| optiland | `optiland-master/optiland/` | 光线追迹、表面定义、坐标系处理 |
| proper | `proper_v3.3.4_python/proper/` | 自由空间衍射传播 |
| numpy | 系统包 | 数值计算 |
| scipy | 系统包 | 插值、优化 |

### 需要迁移的模块

以下模块的核心逻辑需要迁移到 POP 包：

| 原模块 | 迁移内容 | 目标位置 |
|--------|----------|----------|
| `hybrid_optical_propagation/` | 混合传播核心算法 | `pop/propagation/` |
| `wavefront_to_rays/` | 波前采样与重建 | `pop/wavefront/` |
| `sequential_system/zmx_parser.py` | ZMX 文件解析 | `pop/io/` |
| `sequential_system/coordinate_system.py` | 坐标系定义 | `pop/coordinates/` |

### 完全移除的依赖

- `bts/` 包的所有 API 封装
- `hybrid_simulation/` 的冗余抽象层
- `gaussian_beam_simulation/` 的旧实现

## 需求列表

### 需求 1：POP 包结构

**用户故事：** 作为开发者，我希望 POP 包有清晰的目录结构，便于理解和维护。

#### 验收标准

1.1 POP 包 SHALL 采用以下目录结构：
```
pop/
├── __init__.py          # 公共 API 导出
├── core.py              # 核心数据结构（PropagationState, PilotBeam）
├── source.py            # 高斯光源定义
├── propagation/
│   ├── __init__.py
│   ├── free_space.py    # 自由空间传播（调用 proper）
│   ├── element.py       # 元件传播（混合光线追迹）
│   └── pilot_beam.py    # Pilot Beam ABCD 变换
├── wavefront/
│   ├── __init__.py
│   ├── sampler.py       # 波前 → 光线采样
│   └── reconstructor.py # 光线 → 波前重建
├── coordinates/
│   ├── __init__.py
│   └── transforms.py    # 坐标系变换
├── io/
│   ├── __init__.py
│   └── zmx.py           # ZMX 文件解析
└── utils.py             # 通用工具函数
```

1.2 POP 包 SHALL 通过 `__init__.py` 导出所有公共 API

1.3 每个模块 SHALL 不超过 500 行代码（不含注释和空行）


### 需求 2：简洁的公共 API

**用户故事：** 作为用户，我希望用最少的代码完成光学仿真。

#### 验收标准

2.1 基本仿真 SHALL 可用 5 行代码完成：
```python
import pop

system = pop.load_zmx("system.zmx")
source = pop.GaussianSource(wavelength_um=0.633, w0_mm=5.0)
result = pop.propagate(system, source)
result.plot()
```

2.2 手动定义系统 SHALL 支持链式调用：
```python
system = (pop.System()
    .add_mirror(z=100, radius=-200)
    .add_mirror(z=200, tilt_x=45))
```

2.3 所有公共函数 SHALL 有合理的默认参数值

2.4 错误信息 SHALL 明确指出问题和解决方案

### 需求 3：optiland 直接集成

**用户故事：** 作为开发者，我希望直接使用 optiland 的光线追迹能力，避免重复实现。

#### 验收标准

3.1 POP SHALL 直接导入 `optiland-master/optiland/` 下的模块

3.2 光线追迹 SHALL 直接调用 `optiland.Surface.trace()` 方法

3.3 坐标系变换 SHALL 复用 optiland 的 `localize/globalize` 机制

3.4 表面定义 SHALL 直接使用 optiland 的 `Surface` 类，不做额外封装

3.5 POP SHALL 在 `sys.path` 中添加 optiland 路径，确保导入正常

3.6 POP SHALL 以旋转矩阵作为表面姿态主数据，并仅在导入 optiland 时提取欧拉角

### 需求 4：proper 直接集成

**用户故事：** 作为开发者，我希望直接使用 proper 的衍射传播能力。

#### 验收标准

4.1 POP SHALL 直接导入 `proper_v3.3.4_python/proper/` 下的模块

4.2 自由空间传播 SHALL 直接调用 `proper.prop_propagate()` 函数

4.3 POP SHALL 在 `sys.path` 中添加 proper 路径，确保导入正常

4.4 proper 调用 SHALL 保持原有的参数传递方式，不做修改

4.5 自由空间传播 SHALL 根据当前介质折射率进行等效传播距离修正（d_eff = d / n）

4.6 每次 `prop_propagate()` 后 SHALL 同步更新网格采样参数（`wfo.dx`、`wfo.ngrid`）


### 需求 5：混合传播算法迁移

**用户故事：** 作为开发者，我希望将 BTS 的混合传播算法完整迁移到 POP，保持物理正确性。

#### 验收标准

5.1 波前采样算法 SHALL 完整迁移，包括：
- 笛卡尔网格采样（确保中心点对齐）
- 相位梯度计算光线方向
- Pilot Beam 方向叠加

5.2 OPD 计算 SHALL 完整迁移，包括：
- 绝对 OPD 计算（相对于主光线）
- Pilot Beam 理论 OPD
- 残差 OPD（仅去除 Piston 项）

5.3 Pilot Beam 更新 SHALL 完整迁移，包括：
- ABCD 矩阵变换
- OAP 有效曲率半径修正
- 传播方向符号修正

5.4 波前重建 SHALL 完整迁移，包括：
- 雅可比矩阵振幅计算
- 三次插值网格重采样
- 相位突变检测

5.5 所有迁移的算法 SHALL 通过与 BTS 的数值对比测试

5.6 OPD 计算 SHALL 使用带符号的传播距离，并以主光线去除 Piston（保留 Tilt）

5.7 元件传播 SHALL 显式定义与出射光轴对齐的出射面（Exit Plane）

5.8 波前采样 SHALL 保证网格中心与光轴原点重合并强制包含中心光线

### 需求 6：坐标系简化

**用户故事：** 作为开发者，我希望坐标系处理逻辑更加简洁，减少出错可能。

#### 验收标准

6.1 POP SHALL 采用两阶段架构：
- 阶段 1（初始化）：追迹主光线，确定每个表面的入射/出射光轴
- 阶段 2（传播）：在光轴局部坐标系中进行波前采样和重建

6.2 出射光轴 SHALL 直接从 optiland 追迹结果获取，不手动计算反射/折射

6.3 坐标转换代码 SHALL 集中在 `coordinates/transforms.py` 模块

6.4 坐标系符号约定 SHALL 在模块文档中明确定义

6.5 坐标系构建 SHALL 采用最小旋转匹配，避免连续表面之间的坐标轴翻转

6.6 坐标系构建 SHALL 支持使用“参考轴”保持滚转连续性（优先使用上一表面出射坐标系）

6.7 坐标转换 SHALL 通过 optiland 的 `localize/globalize` 完成，且保证右手系

6.8 POP SHALL 在系统构建阶段基于主光线对表面朝向进行规范化，消除 Zemax 的前/后表面翻转效应

6.9 规范化策略 SHALL 保持物理等效：
- 反射面：允许通过翻转局部 Z 轴并同步反转曲率半径来消除翻转
- 折射面：不改变几何表面，仅基于入射方向选择法向用于符号约定


### 需求 7：Code Review 规范

**用户故事：** 作为开发者，我希望在迁移代码时有明确的审查标准，确保不丢失关键逻辑。

#### 验收标准

7.1 迁移代码时 SHALL 逐行对比旧代码，确保：
- 所有物理计算步骤都被保留
- 符号翻转和坐标变换的物理意义被理解
- 看似冗余的代码在删除前确认其用途

7.2 数值计算 SHALL 保持以下细节：
- 除零保护（如 `np.where(R != 0, ..., 0)`）
- 数值稳定性处理（如避免 `sqrt(负数)`）
- 精度敏感的计算顺序

7.3 单位转换 SHALL 显式标注：
- 变量名后缀：`_mm`, `_um`, `_rad`, `_deg`
- 或在注释中说明单位

7.4 边界条件 SHALL 有对应的测试用例：
- 无穷大曲率半径（平面）
- 零传播距离
- 垂直入射（无折射）
- 全反射

### 需求 8：测试与验证

**用户故事：** 作为开发者，我希望 POP 包有完善的测试，确保与 BTS 结果一致。

#### 验收标准

8.1 POP SHALL 包含以下测试类型：
- 单元测试：每个函数的基本功能
- 属性测试：物理不变量（如能量守恒）
- 回归测试：与 BTS 的数值对比

8.2 数值对比测试 SHALL 覆盖：
- 平面镜（不同倾斜角度）
- 球面镜（凹/凸）
- 离轴抛物面镜（OAP）
- 折射面（玻璃平板）

8.3 测试精度要求：
- 相位 RMS 误差 < 0.01 波长
- 振幅相对误差 < 1%
- Pilot Beam 参数相对误差 < 0.1%

8.4 测试 SHALL 使用 pytest 框架，支持 `pytest pop/tests/` 运行


## 参考资料

### 计算方法参考

以下是 BTS 中已验证的核心算法，迁移时应作为参考：

#### 波前采样（来自 `wavefront_to_rays/wavefront_sampler.py`）
- 笛卡尔网格对齐采样
- 相位梯度 → 方向余弦转换
- Pilot Beam 方向解析叠加

#### OPD 计算（来自 `hybrid_optical_propagation/hybrid_element_propagator.py`）
- 绝对 OPD = 光程 - 主光线光程
- Pilot OPD = r² / (2R)
- 残差 OPD = 绝对 OPD - Pilot OPD - Piston

#### Pilot Beam ABCD 变换（来自 `hybrid_optical_propagation/paraxial_propagator.py`）
- 自由空间：`[[1, d], [0, 1]]`
- 球面镜：`[[1, 0], [2/R, 1]]`
- 折射面：`[[1, 0], [(n1-n2)/(n2*R), n1/n2]]`
- OAP 有效曲率：`R_eff = R + d²/R`

#### 波前重建（来自 `wavefront_to_rays/reconstructor.py`）
- 雅可比行列式振幅缩放
- RBF 插值坐标映射
- 三次插值网格重采样

### 符号约定参考

| 物理量 | 正值含义 | 负值含义 |
|--------|----------|----------|
| 曲率半径 R | 曲率中心在 +Z 方向 | 曲率中心在 -Z 方向 |
| Pilot Beam R | 发散波前 | 会聚波前 |
| OPD | 光程比主光线长 | 光程比主光线短 |
| 相位 φ | 相位滞后 | 相位超前 |

## 实施优先级

1. **P0（必须）**：核心传播算法迁移（需求 5）
2. **P0（必须）**：optiland/proper 集成（需求 3, 4）
3. **P1（重要）**：坐标系简化（需求 6）
4. **P1（重要）**：测试框架（需求 8）
5. **P2（期望）**：简洁 API（需求 2）
6. **P2（期望）**：包结构优化（需求 1）

## 风险与注意事项

1. **optiland 版本兼容**：确保使用的 optiland 版本 API 稳定
2. **proper 路径配置**：proper 需要正确的 Python 路径设置
3. **数值精度**：迁移过程中注意浮点数精度问题
4. **向后兼容**：保留 BTS API 作为过渡，逐步迁移用户代码

# POP 包重构需求文档

## 项目目标

基于现有 BTS API 的核心逻辑，构建一个全新的独立代码包：**POP（Physical Optical Propagation）**。

POP 包将：
- 完全脱离 BTS API 的依赖
- 直接调用 `optiland-master/` 和 `proper_v3.3.4_python/` 目录下的代码
- 将必要的旧代码重构或复制进 POP 包，实现完全自包含
- 保持物理正确性和数值精度，同时大幅简化代码结构

## 核心设计原则

### 1. 代码简洁性

- **最小化代码量**：每个模块只做一件事，避免过度抽象
- **扁平化结构**：减少嵌套层级，优先使用函数而非深层类继承
- **直接调用**：直接使用 optiland/proper API，不做无意义的封装
- **删除冗余**：移除所有未使用的代码路径和过度设计的接口

### 2. 垂直接入

- **单一入口**：用户只需 `import pop` 即可使用全部功能
- **零配置启动**：合理的默认值，无需复杂初始化
- **渐进式复杂度**：简单场景简单用，复杂场景才需要深入配置
- **自文档化 API**：函数签名即文档，参数名即说明

### 3. 物理正确性优先

- **保留所有物理计算细节**：不因简化代码而丢失精度
- **符号约定一致**：曲率半径、OPD、相位的符号约定全局统一
- **单位显式化**：所有物理量的单位在变量名或文档中明确标注
- **边界条件完备**：无穷大曲率、零厚度、垂直入射等特殊情况正确处理


## 依赖项清理策略

### 直接依赖（保留）

| 依赖 | 路径 | 用途 |
|------|------|------|
| optiland | `optiland-master/optiland/` | 光线追迹、表面定义、坐标系处理 |
| proper | `proper_v3.3.4_python/proper/` | 自由空间衍射传播 |
| numpy | 系统包 | 数值计算 |
| scipy | 系统包 | 插值、优化 |

### 需要迁移的模块

以下模块的核心逻辑需要迁移到 POP 包：

| 原模块 | 迁移内容 | 目标位置 |
|--------|----------|----------|
| `hybrid_optical_propagation/` | 混合传播核心算法 | `pop/propagation/` |
| `wavefront_to_rays/` | 波前采样与重建 | `pop/wavefront/` |
| `sequential_system/zmx_parser.py` | ZMX 文件解析 | `pop/io/` |
| `sequential_system/coordinate_system.py` | 坐标系定义 | `pop/coordinates/` |

### 完全移除的依赖

- `bts/` 包的所有 API 封装
- `hybrid_simulation/` 的冗余抽象层
- `gaussian_beam_simulation/` 的旧实现

## 需求列表

### 需求 1：POP 包结构

**用户故事：** 作为开发者，我希望 POP 包有清晰的目录结构，便于理解和维护。

#### 验收标准

1.1 POP 包 SHALL 采用以下目录结构：
```
pop/
├── __init__.py          # 公共 API 导出
├── core.py              # 核心数据结构（PropagationState, PilotBeam）
├── source.py            # 高斯光源定义
├── propagation/
│   ├── __init__.py
│   ├── free_space.py    # 自由空间传播（调用 proper）
│   ├── element.py       # 元件传播（混合光线追迹）
│   └── pilot_beam.py    # Pilot Beam ABCD 变换
├── wavefront/
│   ├── __init__.py
│   ├── sampler.py       # 波前 → 光线采样
│   └── reconstructor.py # 光线 → 波前重建
├── coordinates/
│   ├── __init__.py
│   └── transforms.py    # 坐标系变换
├── io/
│   ├── __init__.py
│   └── zmx.py           # ZMX 文件解析
└── utils.py             # 通用工具函数
```

1.2 POP 包 SHALL 通过 `__init__.py` 导出所有公共 API

1.3 每个模块 SHALL 不超过 500 行代码（不含注释和空行）


### 需求 2：简洁的公共 API

**用户故事：** 作为用户，我希望用最少的代码完成光学仿真。

#### 验收标准

2.1 基本仿真 SHALL 可用 5 行代码完成：
```python
import pop

system = pop.load_zmx("system.zmx")
source = pop.GaussianSource(wavelength_um=0.633, w0_mm=5.0)
result = pop.propagate(system, source)
result.plot()
```

2.2 手动定义系统 SHALL 支持链式调用：
```python
system = (pop.System()
    .add_mirror(z=100, radius=-200)
    .add_mirror(z=200, tilt_x=45))
```

2.3 所有公共函数 SHALL 有合理的默认参数值

2.4 错误信息 SHALL 明确指出问题和解决方案

### 需求 3：optiland 直接集成

**用户故事：** 作为开发者，我希望直接使用 optiland 的光线追迹能力，避免重复实现。

#### 验收标准

3.1 POP SHALL 直接导入 `optiland-master/optiland/` 下的模块

3.2 光线追迹 SHALL 直接调用 `optiland.Surface.trace()` 方法

3.3 坐标系变换 SHALL 复用 optiland 的 `localize/globalize` 机制

3.4 表面定义 SHALL 直接使用 optiland 的 `Surface` 类，不做额外封装

3.5 POP SHALL 在 `sys.path` 中添加 optiland 路径，确保导入正常

3.6 POP SHALL 以旋转矩阵作为表面姿态主数据，并仅在导入 optiland 时提取欧拉角

### 需求 4：proper 直接集成

**用户故事：** 作为开发者，我希望直接使用 proper 的衍射传播能力。

#### 验收标准

4.1 POP SHALL 直接导入 `proper_v3.3.4_python/proper/` 下的模块

4.2 自由空间传播 SHALL 直接调用 `proper.prop_propagate()` 函数

4.3 POP SHALL 在 `sys.path` 中添加 proper 路径，确保导入正常

4.4 proper 调用 SHALL 保持原有的参数传递方式，不做修改

4.5 自由空间传播 SHALL 根据当前介质折射率进行等效传播距离修正（d_eff = d / n）

4.6 每次 `prop_propagate()` 后 SHALL 同步更新网格采样参数（`wfo.dx`、`wfo.ngrid`）


### 需求 5：混合传播算法迁移

**用户故事：** 作为开发者，我希望将 BTS 的混合传播算法完整迁移到 POP，保持物理正确性。

#### 验收标准

5.1 波前采样算法 SHALL 完整迁移，包括：
- 笛卡尔网格采样（确保中心点对齐）
- 相位梯度计算光线方向
- Pilot Beam 方向叠加

5.2 OPD 计算 SHALL 完整迁移，包括：
- 绝对 OPD 计算（相对于主光线）
- Pilot Beam 理论 OPD
- 残差 OPD（仅去除 Piston 项）

5.3 Pilot Beam 更新 SHALL 完整迁移，包括：
- ABCD 矩阵变换
- OAP 有效曲率半径修正
- 传播方向符号修正

5.4 波前重建 SHALL 完整迁移，包括：
- 雅可比矩阵振幅计算
- 三次插值网格重采样
- 相位突变检测

5.5 所有迁移的算法 SHALL 通过与 BTS 的数值对比测试

5.6 OPD 计算 SHALL 使用带符号的传播距离，并以主光线去除 Piston（保留 Tilt）

5.7 元件传播 SHALL 显式定义与出射光轴对齐的出射面（Exit Plane）

5.8 波前采样 SHALL 保证网格中心与光轴原点重合并强制包含中心光线

### 需求 6：坐标系简化

**用户故事：** 作为开发者，我希望坐标系处理逻辑更加简洁，减少出错可能。

#### 验收标准

6.1 POP SHALL 采用两阶段架构：
- 阶段 1（初始化）：追迹主光线，确定每个表面的入射/出射光轴
- 阶段 2（传播）：在光轴局部坐标系中进行波前采样和重建

6.2 出射光轴 SHALL 直接从 optiland 追迹结果获取，不手动计算反射/折射

6.3 坐标转换代码 SHALL 集中在 `coordinates/transforms.py` 模块

6.4 坐标系符号约定 SHALL 在模块文档中明确定义

6.5 坐标系构建 SHALL 采用最小旋转匹配，避免连续表面之间的坐标轴翻转

6.6 坐标系构建 SHALL 支持使用“参考轴”保持滚转连续性（优先使用上一表面出射坐标系）

6.7 坐标转换 SHALL 通过 optiland 的 `localize/globalize` 完成，且保证右手系

6.8 POP SHALL 在系统构建阶段基于主光线对表面朝向进行规范化，消除 Zemax 的前/后表面翻转效应

6.9 规范化策略 SHALL 保持物理等效：
- 反射面：允许通过翻转局部 Z 轴并同步反转曲率半径来消除翻转
- 折射面：不改变几何表面，仅基于入射方向选择法向用于符号约定


### 需求 7：Code Review 规范

**用户故事：** 作为开发者，我希望在迁移代码时有明确的审查标准，确保不丢失关键逻辑。

#### 验收标准

7.1 迁移代码时 SHALL 逐行对比旧代码，确保：
- 所有物理计算步骤都被保留
- 符号翻转和坐标变换的物理意义被理解
- 看似冗余的代码在删除前确认其用途

7.2 数值计算 SHALL 保持以下细节：
- 除零保护（如 `np.where(R != 0, ..., 0)`）
- 数值稳定性处理（如避免 `sqrt(负数)`）
- 精度敏感的计算顺序

7.3 单位转换 SHALL 显式标注：
- 变量名后缀：`_mm`, `_um`, `_rad`, `_deg`
- 或在注释中说明单位

7.4 边界条件 SHALL 有对应的测试用例：
- 无穷大曲率半径（平面）
- 零传播距离
- 垂直入射（无折射）
- 全反射

### 需求 8：测试与验证

**用户故事：** 作为开发者，我希望 POP 包有完善的测试，确保与 BTS 结果一致。

#### 验收标准

8.1 POP SHALL 包含以下测试类型：
- 单元测试：每个函数的基本功能
- 属性测试：物理不变量（如能量守恒）
- 回归测试：与 BTS 的数值对比

8.2 数值对比测试 SHALL 覆盖：
- 平面镜（不同倾斜角度）
- 球面镜（凹/凸）
- 离轴抛物面镜（OAP）
- 折射面（玻璃平板）

8.3 测试精度要求：
- 相位 RMS 误差 < 0.01 波长
- 振幅相对误差 < 1%
- Pilot Beam 参数相对误差 < 0.1%

8.4 测试 SHALL 使用 pytest 框架，支持 `pytest pop/tests/` 运行


## 参考资料

### 计算方法参考

以下是 BTS 中已验证的核心算法，迁移时应作为参考：

#### 波前采样（来自 `wavefront_to_rays/wavefront_sampler.py`）
- 笛卡尔网格对齐采样
- 相位梯度 → 方向余弦转换
- Pilot Beam 方向解析叠加

#### OPD 计算（来自 `hybrid_optical_propagation/hybrid_element_propagator.py`）
- 绝对 OPD = 光程 - 主光线光程
- Pilot OPD = r² / (2R)
- 残差 OPD = 绝对 OPD - Pilot OPD - Piston

#### Pilot Beam ABCD 变换（来自 `hybrid_optical_propagation/paraxial_propagator.py`）
- 自由空间：`[[1, d], [0, 1]]`
- 球面镜：`[[1, 0], [2/R, 1]]`
- 折射面：`[[1, 0], [(n1-n2)/(n2*R), n1/n2]]`
- OAP 有效曲率：`R_eff = R + d²/R`

#### 波前重建（来自 `wavefront_to_rays/reconstructor.py`）
- 雅可比行列式振幅缩放
- RBF 插值坐标映射
- 三次插值网格重采样

### 符号约定参考

| 物理量 | 正值含义 | 负值含义 |
|--------|----------|----------|
| 曲率半径 R | 曲率中心在 +Z 方向 | 曲率中心在 -Z 方向 |
| Pilot Beam R | 发散波前 | 会聚波前 |
| OPD | 光程比主光线长 | 光程比主光线短 |
| 相位 φ | 相位滞后 | 相位超前 |

## 实施优先级

1. **P0（必须）**：核心传播算法迁移（需求 5）
2. **P0（必须）**：optiland/proper 集成（需求 3, 4）
3. **P1（重要）**：坐标系简化（需求 6）
4. **P1（重要）**：测试框架（需求 8）
5. **P2（期望）**：简洁 API（需求 2）
6. **P2（期望）**：包结构优化（需求 1）

## 风险与注意事项

1. **optiland 版本兼容**：确保使用的 optiland 版本 API 稳定
2. **proper 路径配置**：proper 需要正确的 Python 路径设置
3. **数值精度**：迁移过程中注意浮点数精度问题
4. **向后兼容**：保留 BTS API 作为过渡，逐步迁移用户代码
