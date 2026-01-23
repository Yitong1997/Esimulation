<!------------------------------------------------------------------------------------
# 测试规范

本文件定义了混合光学仿真项目的测试标准和规范。
inclusion: fileMatch
fileMatchPattern: '**/tests/**,**/*test*,**/*spec*'
------------------------------------------------------------------------------------>

## ⚠️ 主程序使用规范（最高优先级）

### 唯一官方入口

**`examples/zmx_simulation_main.py` 是混合光学仿真系统的 "the only one and final" 主程序。**

```python
from examples.zmx_simulation_main import ZmxSimulationMain

# 标准用法
main = ZmxSimulationMain("system.zmx", wavelength_um=0.633, w0_mm=5.0)
main.visualize()           # 可视化光路
result = main.simulate()   # 执行仿真
main.show_results()        # 展示结果
main.save_results()        # 保存结果

# 或一步完成
result = main.run_all()
```

### 强制规定

1. **所有测试和验证必须通过主程序完成**
2. **禁止自行定义或修改主程序接口**
3. **禁止绕过主程序直接调用底层模块进行端到端测试**
4. **如需扩展功能，必须通过修改主程序实现**

### 路径配置标准

```python
from pathlib import Path

# 标准路径配置（不要修改）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

# ZMX 文件目录
zmx_dir = project_root / 'optiland-master' / 'tests' / 'zemax_files'
```

---

## 测试框架

- **pytest**：单元测试和集成测试
- **hypothesis**：属性基测试（Property-Based Testing）
- **numpy.testing**：数值比较（`assert_allclose`, `assert_array_equal`）

## 目录结构

```
tests/
├── unit/           # 单元测试
├── integration/    # 集成测试
├── validation/     # 验证测试（与参考软件对比）
├── property/       # 属性基测试
└── conftest.py     # pytest 配置和 fixtures
```

## 命名约定

- 测试文件：`test_<模块名>.py`
- 测试类：`Test<类名>`
- 测试方法：`test_<功能描述>`

## 调试规范

### ⚠️ 核心原则

**测试调试时，优先在同一个端到端文件内进行编辑与测试，复用现有模块，尽量不要新建测试文件。**

### 原因

- 减少文件碎片化
- 保持上下文完整
- 复用测试基础设施

### 何时可以新建文件

- 需要生成可视化图表或报告
- 作为 `examples/` 目录下的长期示例

### 精度问题调试方法论

**逐步定位，由粗到细。**

调试仿真精度问题时，按以下步骤逐层缩小范围：

1. **定位问题发生的位置**
   - 先确定精度错误发生在传输过程的哪个面，还是自由空间传输中
   - 对比各个面的入射/出射数据，找到误差首次出现或显著增大的位置

2. **定位问题发生的阶段**
   - 确定问题在该面的入射面处理、光线追迹过程、还是出射面处理
   - 分别检查各阶段的输入输出数据

3. **逐步深入调试**
   - 专注于已定位的局部代码
   - 不要同时修改多处，避免引入新问题
   - 每次修改后验证是否解决问题

**禁止**：在未定位问题根源的情况下，盲目修改多处代码。

## 测试覆盖率目标

- 整体覆盖率 > 80%
- 核心模块覆盖率 > 90%
- 所有公共 API 100% 覆盖
