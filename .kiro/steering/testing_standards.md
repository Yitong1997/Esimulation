<!------------------------------------------------------------------------------------
# 测试规范

inclusion: fileMatch
fileMatchPattern: '**/tests/**,**/*test*,**/*spec*'
------------------------------------------------------------------------------------>

## ⚠️ 强制规定：所有测试必须通过主函数 API 进行

**本项目的所有测试（包括调试、验证、精度测试）都必须通过 BTS 主函数 API 进行。**

禁止：
- 直接使用底层模块（如 `ElementRaytracer`、`WavefrontSampler`）进行测试
- 绕过 `bts.simulate()` 函数直接调用内部实现
- 创建不使用 `bts.OpticalSystem` 的测试代码

正确做法：
- 使用 `bts.OpticalSystem` 定义光学系统
- 使用 `bts.GaussianSource` 定义光源
- 使用 `bts.simulate()` 执行仿真
- 通过 `SimulationResult` 获取和分析结果

---

## 主程序编写风格

**采用 MATLAB 风格的"代码块"结构，而非 Python 类封装风格。**

主程序应包含清晰分隔的代码块：
1. 导入与初始化
2. 定义光学系统（ZMX 导入或逐行定义）
3. 定义光源
4. 系统信息展示
5. 执行仿真
6. 结果展示与保存

**核心原则**：
- 代码块分明，一目了然
- API 简洁，复杂逻辑封装在模块内部
- 传参稳定，不频繁变动

---

## ⚠️⚠️⚠️ 绝对坐标定义规范（极其重要）

**逐行定义光学元件时，使用绝对坐标 (x, y, z) 定义表面位置。**

### 🚫 绝对禁止

- **绝对禁止使用 `off_axis_distance` 参数！**
- **绝对禁止使用 `add_oap` 方法！**
- **绝对禁止将离轴距离设置为单独的变量！**

### ✅ 正确做法

离轴量必须直接通过修改元件的位置坐标来实现：

```python
import bts

system = bts.OpticalSystem("OAP Test")

# ✅ 正确：离轴抛物面镜，Y 方向离轴 100mm
system.add_parabolic_mirror(
    x=0,             # X 位置
    y=100,           # Y 位置 = 离轴量 100mm
    z=0,             # Z 位置（抛物面顶点）
    radius=200,      # 曲率半径 R = 2f
    semi_aperture=20,
)

# 🚫 错误：绝对禁止！
# system.add_parabolic_mirror(
#     y=0,
#     z=0,
#     radius=200,
#     off_axis_distance=100,  # 🚫 禁止！
# )
```

---

## 测试框架

- **pytest**：单元测试和集成测试
- **hypothesis**：属性基测试
- **numpy.testing**：数值比较

## 目录结构

```
tests/
├── unit/           # 单元测试
├── integration/    # 集成测试
├── validation/     # 验证测试
├── property/       # 属性基测试
└── conftest.py     # pytest 配置
```

## 命名约定

- 测试文件：`test_<模块名>.py`
- 测试类：`Test<类名>`
- 测试方法：`test_<功能描述>`

---

## 调试规范

### ⚠️ 核心原则

**优先在同一个端到端文件内进行编辑与测试，复用现有模块，尽量不要新建测试文件。**

### 何时可以新建文件

- 需要生成可视化图表或报告
- 作为 `examples/` 目录下的长期示例

### 精度问题调试方法论

**逐步定位，由粗到细：**

1. **定位问题发生的位置**：确定在哪个面或自由空间传输中
2. **定位问题发生的阶段**：入射面处理、光线追迹、还是出射面处理
3. **逐步深入调试**：专注于已定位的局部代码，每次只修改一处

**禁止**：在未定位问题根源的情况下，盲目修改多处代码。

---

## 测试覆盖率目标

- 整体覆盖率 > 80%
- 核心模块覆盖率 > 90%
- 所有公共 API 100% 覆盖

---

## ⚠️ 核心回归测试

**任何对以下模块的修改，必须运行对应的回归测试：**

### 1. 倾斜平面镜传输测试

**触发条件**：修改以下文件时必须运行
- `src/wavefront_to_rays/element_raytracer.py`
- `src/hybrid_optical_propagation/hybrid_element_propagator.py`
- `src/wavefront_to_rays/wavefront_sampler.py`
- `src/wavefront_to_rays/reconstructor.py`

**测试文件**：`tests/integration/不同倾斜角度平面镜传输误差标准测试文件.py`

**通过标准**：
- 所有角度（0°-60°）RMS < 1 milli-wave
- 所有角度结果应一致（平面镜无像差）

**运行命令**：
```bash
python tests/integration/不同倾斜角度平面镜传输误差标准测试文件.py
```

### 2. 离轴抛物面镜测试

**触发条件**：修改以下文件时必须运行
- `src/wavefront_to_rays/element_raytracer.py`
- `src/hybrid_optical_propagation/hybrid_element_propagator.py`
- 任何涉及 OPD 计算的模块

**测试文件**：`tests/integration/离轴抛物面镜传输误差标准测试文件.py`

**通过标准**：
- 相位 RMS < 10 milli-waves
- 振幅 RMS < 1%

**运行命令**：
```bash
python tests/integration/离轴抛物面镜传输误差标准测试文件.py
```
