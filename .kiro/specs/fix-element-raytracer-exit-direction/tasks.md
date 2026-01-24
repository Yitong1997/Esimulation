# 任务列表

## 任务概述

本任务列表用于修复 `ElementRaytracer._compute_exit_chief_direction` 方法在计算离轴抛物面出射方向时的 bug。

## 任务

- [ ] 1. 实现解析几何计算方法
  - [ ] 1.1 在 `ElementRaytracer` 类中添加 `_compute_oap_exit_direction_analytic` 方法
    - 参考 `scripts/oap_debug/step1_chief_ray.py` 中的 `ChiefRayTracer` 实现
    - 实现抛物面交点计算：z = d²/(2R)
    - 实现表面法向量计算：n = (-x/R, -y/R, 1)/|n|
    - 实现反射方向计算：r = i - 2(i·n)n
    - 将结果从入射面局部坐标系转换到全局坐标系
    - **验收标准**：需求 2.1, 2.2, 2.3
  - [ ] 1.2 修改 `_compute_exit_chief_direction` 方法添加条件分支
    - 检测条件：conic == -1.0 且 off_axis_distance != 0.0 且 surface_type == 'mirror'
    - 满足条件时调用 `_compute_oap_exit_direction_analytic`
    - 其他情况保持现有 optiland 追迹逻辑
    - **验收标准**：需求 2.4, 3.1, 3.2, 3.3, 3.4

- [ ] 2. 验证修复正确性
  - [ ] 2.1 运行出射方向验证脚本
    - 执行 `python scripts/oap_debug/step2_coordinate_system.py`
    - 确认所有 4 组参数（长焦距_轴上、长焦距_离轴、超长焦距_轴上、超长焦距_离轴）通过验证
    - **验收标准**：需求 1.1, 1.2, 1.3, 4.1, 4.2

- [ ] 3. 运行回归测试
  - [ ] 3.1 运行倾斜平面镜回归测试
    - 执行 `python tests/integration/不同倾斜角度平面镜传输误差标准测试文件.py`
    - 确认所有角度（0°-60°）RMS < 1 milli-wave
    - **验收标准**：需求 3.1, 3.3, 4.3
  - [ ] 3.2 运行离轴抛物面镜回归测试
    - 执行 `python tests/integration/离轴抛物面镜传输误差标准测试文件.py`
    - 确认相位 RMS < 10 milli-waves
    - **验收标准**：需求 1.1, 1.3, 4.3
