# 任务列表

## 任务 1：PilotBeamParams 曲率半径公式审查

- [x] 1.1 审查 `from_gaussian_source` 方法使用严格公式 `R = z * (1 + (z_R / z)**2)`
- [x] 1.2 审查 `from_q_parameter` 方法正确提取曲率半径
- [x] 1.3 审查 `compute_phase_grid` 方法使用 `self.curvature_radius_mm`
- [x] 1.4 创建属性测试验证曲率半径公式正确性

## 任务 2：StateConverter PROPER 参考面相位审查

- [x] 2.1 审查 `compute_proper_reference_phase` 方法检查参考面类型
- [x] 2.2 审查平面参考面处理（返回零相位）
- [x] 2.3 审查球面参考面使用远场近似 `wfo.z - wfo.z_w0`
- [x] 2.4 搜索代码确认无公式混用

## 任务 3：ParaxialPhasePropagator 薄透镜 Pilot Beam 更新审查

- [x] 3.1 审查 `prop_lens` 应用薄透镜效果
- [x] 3.2 审查 `apply_lens` 更新 Pilot Beam 参数
- [x] 3.3 创建测试验证 PROPER 参数与 ABCD 法则计算结果一致

## 任务 4：相位连续性验证器实现

- [x] 4.1 实现 PhaseContinuityValidator 类
- [x] 4.2 实现 validate_phase_grid 方法
- [x] 4.3 创建相位连续性属性测试

## 任务 5：能量守恒验证器实现

- [x] 5.1 实现 EnergyConservationValidator 类
- [x] 5.2 实现 validate_energy_conservation 方法
- [x] 5.3 创建能量守恒属性测试

## 任务 6：近场传播验证

- [x] 6.1 创建近场传播测试用例（z ≈ z_R）
- [x] 6.2 验证 Pilot Beam 使用严格公式
- [x] 6.3 验证近场与远场公式在远场趋于一致
