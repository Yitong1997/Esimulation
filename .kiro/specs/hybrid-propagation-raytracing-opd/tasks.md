# 任务列表：混合传播模式光线追迹 OPD 重构

## 概述

本任务列表基于需求文档和设计文档，实现混合传播模式的重构，使其完全使用真实的几何光线追迹计算 OPD，而不依赖 PROPER 的 `prop_lens` 函数进行相位计算。

## 任务

### 1. 核心方法实现

- [x] 1.1 实现 `_update_gaussian_params_only` 方法
  - 复用 `prop_lens` 的参数更新逻辑，但不修改 `wfarr`
  - 更新 z_w0, w0, z_Rayleigh, beam_type_old, reference_surface, propagator_type, current_fratio
  - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

- [x] 1.2 实现 `_compute_reference_phase` 方法
  - 根据 PROPER 参考面类型（PLANAR/SPHERI）计算参考相位
  - PLANAR 参考面返回零相位
  - SPHERI 参考面返回二次相位 φ_ref = -k * r² / (2 * R_ref)
  - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [x] 1.3 实现 `_check_phase_sampling` 方法
  - 检查相邻像素间相位差是否超过 π
  - 超过时发出警告并建议增加网格大小或减小光束尺寸
  - **Validates: Requirements 7.1, 7.2, 7.3**

- [x] 1.4 实现 `_get_sampling_half_size_mm` 方法
  - 使用 PROPER 网格的完整尺寸作为采样范围
  - 避免基于光束强度计算导致的面积收缩问题
  - **Validates: Requirements 1.1**

- [x] 1.5 实现 `_create_sampling_rays` 方法
  - 在整个采样面上创建均匀分布的采样点
  - 不基于光束强度限制采样范围
  - **Validates: Requirements 1.1**

### 2. 重构 `_apply_element_hybrid` 方法

- [x] 2.1 重构 `_apply_element_hybrid` 方法的核心逻辑
  - 移除对 `prop_lens` 相位计算的依赖
  - 使用 `_update_gaussian_params_only` 更新高斯光束参数
  - 使用 ElementRaytracer 计算完整 OPD
  - 计算参考面相位并得到残差相位
  - 将残差相位应用到波前
  - **Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4**

- [x] 2.2 处理 OPD 符号修正
  - ElementRaytracer 的 OPD 符号与 PROPER 相反，需要取反
  - **Validates: Requirements 1.3**

- [x] 2.3 处理平面镜特殊情况
  - 焦距为无穷大的平面镜不需要 OPD 计算
  - 失调倾斜（is_fold=False）需要添加倾斜相位
  - **Validates: Requirements 1.1**

- [x] 2.4 处理理想抛物面镜特殊情况
  - 理想抛物面镜（conic=-1）对轴上平行光无像差
  - 可以跳过像差计算
  - **Validates: Requirements 4.2**

### 3. 单元测试

- [x] 3.1 编写 `_update_gaussian_params_only` 方法的单元测试
  - 测试与 `prop_lens` 参数更新结果的一致性
  - 测试不同焦距值（正、负）
  - 测试不同初始光束状态（INSIDE_, OUTSIDE）
  - **Validates: Property 2**

- [x] 3.2 编写 `_compute_reference_phase` 方法的单元测试
  - 测试 PLANAR 参考面（应返回零）
  - 测试 SPHERI 参考面（应返回二次相位）
  - 测试不同 z_w0 值
  - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [x] 3.3 编写 `_check_phase_sampling` 方法的单元测试
  - 测试正常相位梯度（不应警告）
  - 测试过大相位梯度（应警告）
  - **Validates: Requirements 7.1, 7.2, 7.3**

### 4. ElementRaytracer OPD 验证测试

- [x] 4.1 编写平面镜 OPD 常数性测试（属性测试）
  - 平行光入射平面镜，OPD RMS < 0.001 波
  - **Validates: Requirements 4.4, Property 5**

- [x] 4.2 编写抛物面镜 OPD 常数性测试（属性测试）
  - 平行光入射理想抛物面镜，OPD RMS < 0.01 波
  - **Validates: Requirements 4.2, Property 3**

- [x] 4.3 编写凹面镜 OPD 解析验证测试（属性测试）
  - 平行光入射球面凹面镜，OPD 与解析公式一致（相对误差 < 0.1%）
  - 验证球差公式 SA = r⁴ / (8 * R³)
  - **Validates: Requirements 4.1, 4.3, Property 4**

- [x] 4.4 编写 45° 折叠镜坐标变换测试
  - 验证出射光线方向与反射定律一致
  - 验证 OPD 分布保持对称性
  - **Validates: Requirements 4.5, Property 6**

### 5. 集成测试 - 简单光路

- [x] 5.1 编写单凹面镜集成测试
  - 高斯光束通过单个凹面镜
  - 混合模式与 ABCD 理论的光束半径误差 < 1%
  - **Validates: Requirements 5.1, Property 7**

- [x] 5.2 编写单抛物面镜集成测试
  - 高斯光束通过单个抛物面镜
  - WFE RMS < 0.1 波
  - **Validates: Requirements 5.2, Property 7**

- [x] 5.3 编写单平面镜集成测试
  - 高斯光束通过单个平面镜
  - WFE RMS < 0.01 波
  - **Validates: Requirements 5.3**

- [x] 5.4 编写 45° 折叠镜集成测试
  - 高斯光束通过 45° 折叠镜
  - 输出光束方向正确改变
  - WFE RMS < 0.01 波
  - **Validates: Requirements 5.4**

### 6. 集成测试 - 复杂光路

- [x] 6.1 编写伽利略式扩束镜集成测试
  - 使用 galilean_oap_expander.py 示例验证
  - 混合模式与 ABCD 理论的光束半径误差 < 1%
  - 放大倍率与设计值一致（误差 < 1%）
  - **Validates: Requirements 6.1, 6.2, Property 8**

- [x] 6.2 编写多元件折叠光路集成测试
  - 包含多个折叠镜的系统
  - 各采样面的 WFE RMS < 0.1 波
  - **Validates: Requirements 6.3**

- [x] 6.3 编写多 OAP 系统集成测试
  - 包含多个 OAP 的系统
  - 输出光束参数与 ABCD 理论一致
  - **Validates: Requirements 6.4, Property 8**

### 7. 属性测试（Property-Based Testing）

- [x] 7.1 编写 OPD 符号与参考面变换正确性属性测试
  - 验证残差相位正确反映元件引入的波前变化
  - **Validates: Property 1**

- [x] 7.2 编写高斯光束参数更新正确性属性测试
  - 验证 `_update_gaussian_params_only` 与 `prop_lens` 的参数更新结果一致
  - 使用 hypothesis 库生成随机焦距和初始参数
  - **Validates: Property 2**

- [x] 7.3 编写相位采样检查正确性属性测试
  - 验证相位梯度超过 π 时发出警告
  - **Validates: Property 9**

### 8. 文档和清理

- [x] 8.1 更新 `_apply_element_hybrid` 方法的文档字符串
  - 描述新的实现逻辑
  - 说明与旧实现的区别
  - **Validates: NFR-4**

- [x] 8.2 添加中文注释说明关键算法
  - 参考面变换算法
  - OPD 符号约定
  - 高斯光束参数更新算法
  - **Validates: NFR-4**

- [ ]* 8.3 清理旧的调试文件（可选）
  - 移除或归档 tests/debug_*.py 文件
  - **Validates: NFR-4**

## 依赖关系

```
1.1 ─┬─> 2.1
1.2 ─┤
1.3 ─┤
1.4 ─┤
1.5 ─┘

2.1 ─┬─> 3.1, 3.2, 3.3
2.2 ─┤
2.3 ─┤
2.4 ─┘

3.x ──> 4.x ──> 5.x ──> 6.x ──> 7.x ──> 8.x
```

## 注意事项

1. **OPD 符号约定**：ElementRaytracer 的 OPD 符号与 PROPER 相反，需要取反
2. **坐标系统**：ElementRaytracer 使用 mm 单位，PROPER 使用 m 单位
3. **参考面更新时机**：高斯光束参数必须在计算参考面相位之前更新
4. **大 OPD 值**：对于 f=-50mm 的抛物面镜，r=10mm 处的 OPD 约为 1580 波，需要正确处理
5. **采样策略**：在整个 PROPER 网格范围内进行光线采样，避免面积收缩
