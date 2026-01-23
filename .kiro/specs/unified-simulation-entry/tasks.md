# Implementation Tasks

## Task 1: 创建模块结构和数据模型

- [x] 1.1 创建 `src/hybrid_simulation/` 目录结构
- [x] 1.2 实现 `data_models.py` - 定义所有数据类
  - SimulationConfig
  - SourceParams
  - SurfaceGeometry
  - OpticalAxisInfo
  - PilotBeamInfo
  - GridInfo
- [x] 1.3 实现 `exceptions.py` - 定义异常类
  - ConfigurationError
  - SimulationError

## Task 2: 实现 WavefrontData 类

- [x] 2.1 实现 `WavefrontData` 基本结构
  - amplitude, phase, pilot_beam, grid 属性
- [x] 2.2 实现 `get_intensity()` 方法
- [x] 2.3 实现 `get_complex_amplitude()` 方法
- [x] 2.4 实现 `get_pilot_beam_phase()` 方法
- [x] 2.5 实现 `get_residual_phase()` 方法
- [x] 2.6 实现 `get_residual_rms_waves()` 方法

## Task 3: 实现 SurfaceRecord 和 SimulationResult 类

- [x] 3.1 实现 `SurfaceRecord` 类
  - index, name, surface_type, geometry
  - entrance, exit (WavefrontData)
  - optical_axis
- [x] 3.2 实现 `SimulationResult` 基本结构
  - success, error_message, config, source_params
  - surfaces, total_path_length
- [x] 3.3 实现 `get_surface()` 方法（支持索引和名称）
- [x] 3.4 实现 `summary()` 方法

## Task 4: 实现 HybridSimulator 类

- [x] 4.1 实现 `HybridSimulator.__init__()` 和基本结构
- [x] 4.2 实现 `load_zmx()` 方法（复用 load_optical_system_from_zmx）
- [x] 4.3 实现 `add_flat_mirror()` 方法
- [x] 4.4 实现 `add_spherical_mirror()` 方法
- [x] 4.5 实现 `add_paraxial_lens()` 方法
- [x] 4.6 实现 `set_source()` 方法（复用 SourceDefinition）
- [x] 4.7 实现 `run()` 方法（复用 HybridOpticalPropagator）
- [x] 4.8 实现 `_convert_result()` 内部方法

## Task 5: 实现可视化功能

- [x] 5.1 实现 `plot_surface_detail()` 函数
  - 绘制振幅、相位、Pilot Beam 相位、残差相位
  - 标注表面信息和误差指标
- [x] 5.2 实现 `plot_all_surfaces()` 函数
  - 绘制所有表面的概览图
- [x] 5.3 实现 `SimulationResult.plot_all()` 方法
- [x] 5.4 实现 `SimulationResult.plot_surface()` 方法

## Task 6: 实现序列化功能

- [x] 6.1 实现 `save_result()` 函数
  - 保存振幅/相位为 .npy
  - 保存配置/参数为 .json
- [x] 6.2 实现 `load_result()` 函数
- [x] 6.3 实现 `SimulationResult.save()` 方法
- [x] 6.4 实现 `SimulationResult.load()` 类方法

## Task 7: 创建模块入口和导出

- [x] 7.1 实现 `__init__.py` - 导出公共 API
  - HybridSimulator
  - SimulationResult
  - SurfaceRecord
  - WavefrontData

## Task 8: 创建测试示例

- [x] 8.1 创建简单折叠镜测试示例 `examples/simple_fold_mirror_test.py`
  - 使用 HybridSimulator 定义 45° 平面镜
  - 执行仿真
  - 打印摘要
  - 绘制结果
  - 保存结果
- [x] 8.2 验证测试示例运行成功
