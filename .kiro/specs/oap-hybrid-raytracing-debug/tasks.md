# Implementation Plan: OAP Hybrid Raytracing Debug

## Overview

本实现计划将设计文档中的调试流程转换为可执行的任务。采用渐进式验证方法，每个步骤验证通过后记录状态，后续步骤不再重复验证。

**核心原则**：所有操作都通过 BTS API 进行，不直接使用底层模块。

## Tasks

- [x] 1. 扩展 BTS API 以支持调试数据读取
  - [x] 1.1 在 SimulationResult 中添加光线数据读取接口
    - 添加 `get_surface_rays(surface_index, location)` 方法
    - 添加 `get_chief_ray(surface_index)` 方法
    - 添加 `get_pilot_beam_params(surface_index, location)` 方法
    - 添加 `get_coordinate_system(surface_index, location)` 方法
    - _Requirements: 12.1, 12.2_
  
  - [x] 1.2 创建数据类
    - 创建 `RayData` 数据类（光线位置、方向、OPD）
    - 创建 `ChiefRayData` 数据类（主光线信息）
    - 创建 `CoordinateSystemData` 数据类（坐标系信息）
    - _Requirements: 12.1_

- [~] 2. 创建调试基础设施
  - [x] 2.1 创建 OAP 几何计算器
    - 在 `scripts/oap_debug/` 目录下创建 `geometry.py`
    - 实现 `OAPGeometryCalculator` 类
    - 实现纯几何计算方法（交点、法向量、反射方向、出射面位置、理论 OPD）
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  
  - [~] 2.2 创建 BTS 调试辅助器
    - 创建 `scripts/oap_debug/bts_helper.py`
    - 实现 `BTSDebugHelper` 类
    - 封装 BTS API 调用，提供数据提取功能
    - _Requirements: 12.1, 12.2, 12.3, 12.4_
  
  - [x] 2.3 创建测试参数集和验证状态追踪器
    - 创建 `scripts/oap_debug/parameters.py`
    - 实现 `OAPParameters` 和 `TestParameterSet` 数据类
    - 实现 `VerificationStatusTracker` 类
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 3. 步骤 1：主光线追迹验证
  - [x] 3.1 创建主光线追迹验证脚本
    - 创建 `scripts/oap_debug/step1_chief_ray.py`
    - 使用 BTS API 执行仿真
    - 从 SimulationResult 获取主光线数据
    - 与几何计算器的理论值比较
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  
  - [x] 3.2 运行主光线追迹验证
    - 使用 4 组测试参数运行验证
    - 输出详细的几何计算结果
    - 如果通过，标记步骤为"已验证"
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [~] 4. 步骤 2：坐标系验证
  - [x] 4.1 创建坐标系验证脚本
    - 创建 `scripts/oap_debug/step2_coordinate_system.py`
    - 使用 BTS API 获取坐标系数据
    - 验证入射面垂直于入射主光线
    - 验证出射面垂直于出射主光线
    - 验证旋转矩阵的正交性
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4_
  
  - [~] 4.2 运行坐标系验证
    - 使用 4 组测试参数运行验证
    - 输出坐标变换的详细结果
    - 如果通过，标记步骤为"已验证"
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4_


- [x] 5. 步骤 3：Pilot Beam 参数验证
  - [x] 5.1 创建 Pilot Beam 验证脚本
    - 创建 `scripts/oap_debug/step3_pilot_beam.py`
    - 使用 BTS API 获取 Pilot Beam 参数
    - 验证等效焦距计算 f_eff = sqrt(d² + (f - z)²)
    - 验证等效曲率半径 R_eff = 2 × f_eff
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  
  - [x] 5.2 运行 Pilot Beam 验证
    - 使用 4 组测试参数运行验证
    - 输出入射和出射 Pilot Beam 参数对比
    - 如果通过，标记步骤为"已验证"
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [~] 6. 步骤 4：出射面光线几何验证（核心）
  - [~] 6.1 创建出射面光线验证脚本
    - 创建 `scripts/oap_debug/step4_exit_surface_rays.py`
    - 使用 BTS API 获取出射面光线数据
    - 使用几何计算器计算理论光线位置和 OPD
    - 比较实际与理论分布
    - _Requirements: 5.1, 5.3, 5.4_
  
  - [~] 6.2 实现理论光线位置计算
    - 计算入射光线与抛物面的交点
    - 计算交点处的表面法向量
    - 计算反射方向
    - 计算光线与出射面的交点（出射面局部坐标）
    - _Requirements: 5.1, 5.3_
  
  - [~] 6.3 实现理论 OPD 计算
    - 计算入射光程（入射面到交点）
    - 计算反射后光程（交点到出射面）
    - 计算相对于主光线的 OPD
    - _Requirements: 5.4_
  
  - [~] 6.4 运行出射面光线验证
    - 使用 4 组测试参数运行验证
    - 输出位置误差和 OPD 误差统计
    - 生成理论 vs 实际的对比图
    - 如果通过，标记步骤为"已验证"
    - _Requirements: 5.1, 5.3, 5.4_

- [~] 7. 步骤 5：残差 OPD 和网格重采样验证
  - [~] 7.1 创建残差 OPD 验证脚本
    - 创建 `scripts/oap_debug/step5_residual_opd.py`
    - 使用 BTS API 获取出射面光线 OPD
    - 计算 Pilot Beam OPD
    - 计算残差 OPD = 实际 OPD + Pilot Beam OPD（注意是加法，因为符号约定）
    - 验证残差 OPD 的平滑性
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  
  - [~] 7.2 验证网格重采样
    - 将残差 OPD 重采样到网格
    - 验证插值精度
    - 验证重建波前与理论波前的一致性
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  
  - [~] 7.3 运行残差 OPD 验证
    - 使用 4 组测试参数运行验证
    - 输出残差 OPD 的统计信息
    - 如果通过，标记步骤为"已验证"
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 7.1, 7.2, 7.3, 7.4_

- [~] 8. Checkpoint - 验证所有步骤
  - 确保所有 5 个步骤都已验证
  - 如果有步骤失败，分析失败原因
  - 询问用户是否需要进一步调试

- [~] 9. 创建综合验证脚本
  - [~] 9.1 创建主验证脚本
    - 创建 `scripts/oap_debug/run_all_verifications.py`
    - 按顺序运行所有验证步骤
    - 跳过已验证的步骤
    - 输出综合验证报告
    - _Requirements: 9.1, 9.2, 9.3_
  
  - [~] 9.2 验证轴上情况与球面镜一致性
    - 在综合脚本中添加轴上一致性检查
    - 比较 d=0 时的结果与球面镜结果
    - _Requirements: 8.5_

- [~] 10. Final checkpoint - 确认调试完成
  - 确保所有验证通过
  - 生成最终验证报告
  - 询问用户是否有其他问题

## Notes

### ⚠️⚠️⚠️ 强制禁止事项（每次执行任务前必读）

#### 🚫🚫🚫 绝对禁止的参数和方法

以下参数和方法已被**永久废弃**，**永远不要使用**：

| 禁止项 | 说明 |
|--------|------|
| `off_axis_distance` | 离轴距离参数 |
| `dy` | optiland 表面 Y 方向偏心参数 |
| `dx` | optiland 表面 X 方向偏心参数 |
| `add_oap` | 离轴抛物面添加方法 |
| `semi_aperture` | 半口径参数 |
| `aperture` | 口径参数 |

#### ✅ 正确做法

```python
# ✅ 正确：使用 BTS API 定义离轴抛物面
import bts

system = bts.OpticalSystem("OAP Test")
system.add_parabolic_mirror(
    x=0,             # X 位置
    y=100,           # Y 位置 = 离轴量 100mm
    z=0,             # Z 位置
    radius=200,      # 曲率半径
)

source = bts.GaussianSource(wavelength_um=0.633, w0_mm=5.0)
result = bts.simulate(system, source)

# 🚫 错误：绝对禁止！
# system.add_parabolic_mirror(off_axis_distance=100)  # 禁止！
# optic.add_surface(dy=100)  # 禁止！
```

#### 🚫 禁止直接使用底层模块测试

```python
# 🚫 禁止！
# from src.wavefront_to_rays.element_raytracer import ElementRaytracer

# ✅ 正确：通过 BTS API 测试
import bts
result = bts.simulate(system, source)

# 使用扩展接口获取调试数据
exit_rays = result.get_surface_rays(surface_index=0, location="exit")
chief_ray = result.get_chief_ray(surface_index=0)
```

### 其他注意事项

- 本 spec 的目标是调试而非创建新功能，因此不包含属性测试任务
- 所有验证脚本都通过 BTS 主函数 API 进行测试
- 验证状态持久化到 JSON 文件，支持跨会话继续调试
- 每个步骤验证通过后，后续步骤不再重复验证该步骤
- 如果发现明确的错误，需要修改 API 内部接口时，应先与用户确认
- **OPD 验证面是出射面，不是焦点！**

