# 混合光线追迹重构任务清单

## 阶段 1: 核心方法实现

### Task 1.1: 实现方向到旋转角度转换
- [x] 1.1 实现 `_direction_to_rotation_angles()` 方法

**目标**: 将方向向量转换为旋转角度 (rx, ry)

**实现步骤**:
1. 添加 `_direction_to_rotation_angles()` 方法到 `ElementRaytracer` 类
2. 实现公式：`rx = -arcsin(M)`, `ry = arctan2(L, N)`
3. 处理边界情况（M 接近 ±1）

**验收标准**:
- (0, 0, 1) → (0, 0)
- (0, -1, 0) → (π/2, 0)
- (1, 0, 0) → (0, π/2)

**文件**: `src/wavefront_to_rays/element_raytracer.py`

---

### Task 1.2: 实现出射主光线方向计算
- [x] 1.2 实现 `_compute_exit_chief_direction()` 方法

**目标**: 计算出射主光线方向（反射定律）

**实现步骤**:
1. 添加 `_compute_exit_chief_direction()` 方法
2. 对于反射镜：使用反射公式 `r = d - 2(d·n)n`
3. 考虑表面倾斜对法向量的影响
4. 返回归一化的方向余弦

**验收标准**:
- 正入射平面镜：出射方向 (0, 0, -1)
- 45° 折叠镜（tilt_x=π/4）：出射方向 (0, -1, 0)

**文件**: `src/wavefront_to_rays/element_raytracer.py`

---

## 阶段 2: 修改现有方法

### Task 2.1: 修改 _create_optic() - 添加倾斜出射面
- [x] 2.1 修改 `_create_optic()` 方法，添加倾斜的透明平面作为出射面

**目标**: 在 optiland 光学系统中添加倾斜的出射面

**实现步骤**:
1. 在添加元件表面后，调用 `_compute_exit_chief_direction()`
2. 调用 `_direction_to_rotation_angles()` 计算旋转角度
3. 添加倾斜的透明平面：`material='air'`, `rx=exit_rx`, `ry=exit_ry`
4. 保存 `exit_chief_direction` 和 `exit_rotation_matrix` 属性

**验收标准**:
- optiland 光学系统包含 3 个表面（物面、元件、出射面）
- 出射面的旋转角度正确

**文件**: `src/wavefront_to_rays/element_raytracer.py`

---

### Task 2.2: 简化 trace() 方法
- [x] 2.2 简化 `trace()` 方法，直接使用 optiland 追迹

**目标**: 移除手动传播逻辑，让 optiland 完成整个追迹

**实现步骤**:
1. 坐标变换：入射面局部 → 全局
2. 调用 `surface_group.trace(rays, skip=1)`
3. 坐标变换：全局 → 出射面局部（使用 `exit_rotation_matrix`）

**验收标准**:
- 45° 折叠镜：输出光线方向正确
- 输出光线 z≈0（在出射面局部坐标系中）
- OPD 计算正确

**文件**: `src/wavefront_to_rays/element_raytracer.py`

---

### Task 2.3: 更新 get_exit_chief_ray_direction() 方法
- [x] 2.3 更新 `get_exit_chief_ray_direction()` 方法

**目标**: 返回预先计算的出射主光线方向

**实现步骤**:
1. 直接返回 `self.exit_chief_direction` 属性
2. 添加 `get_exit_rotation_matrix()` 方法

**验收标准**:
- 返回值与 `_compute_exit_chief_direction()` 一致

**文件**: `src/wavefront_to_rays/element_raytracer.py`

---

## 阶段 3: 单元测试

### Task 3.1: 方向到旋转角度转换测试
- [ ] 3.1 编写 `_direction_to_rotation_angles()` 的单元测试

**测试用例**:
1. (0, 0, 1) → (0, 0)
2. (0, -1, 0) → (π/2, 0)
3. (1, 0, 0) → (0, π/2)
4. 45° 方向测试

**文件**: `tests/test_element_raytracer_rotation.py`

---

### Task 3.2: 出射主光线方向计算测试
- [ ] 3.2 编写 `_compute_exit_chief_direction()` 的单元测试

**测试用例**:
1. 正入射平面镜：出射方向 (0, 0, -1)
2. 45° 折叠镜（tilt_x=π/4）：出射方向 (0, -1, 0)
3. 凹面镜正入射：出射方向 (0, 0, -1)

**文件**: `tests/test_element_raytracer_exit_direction.py`

---

## 阶段 4: 集成测试

### Task 4.1: 45° 折叠镜集成测试
- [ ] 4.1 编写 45° 折叠镜的集成测试

**测试内容**:
1. 输入平面波
2. 经 45° 折叠镜反射
3. 验证输出仍为平面波
4. 验证光束半径不变
5. 验证 WFE RMS < 0.01 波长

**文件**: `tests/test_folding_mirror_integration.py`

---

### Task 4.2: 伽利略式 OAP 扩束镜集成测试
- [ ] 4.2 更新伽利略式 OAP 扩束镜测试

**测试内容**:
1. 输入 10mm 束腰高斯光束
2. 经 OAP1、折叠镜、OAP2
3. 验证输出 30mm 束腰
4. 验证各采样面光束半径误差 < 5%
5. 验证 WFE RMS < 0.1 波长

**文件**: `tests/test_galilean_oap_phase_validation.py`

---

## 阶段 5: 属性基测试

### Task 5.1: 方向余弦归一化属性测试
- [ ] 5.1 编写方向余弦归一化的属性基测试

**属性**: 出射光线的方向余弦满足 L² + M² + N² = 1

**Validates: Requirements REQ-2.3**

**文件**: `tests/test_element_raytracer_properties.py`

---

### Task 5.2: 出射面位置属性测试
- [ ] 5.2 编写出射面位置的属性基测试

**属性**: 出射光线在出射面局部坐标系中 z ≈ 0

**Validates: Requirements REQ-2.3**

**文件**: `tests/test_element_raytracer_properties.py`

---

## 依赖关系

```
Task 1.1 ──┐
           ├──→ Task 2.1 ──→ Task 2.2 ──→ Task 4.1
Task 1.2 ──┘                    │          Task 4.2
                                │
                                └──→ Task 2.3

Task 3.1 ──┐
Task 3.2 ──┴──→ Task 4.1, Task 4.2

Task 5.1 ──┐
Task 5.2 ──┴──→ 验证
```

## 优先级

1. **高优先级**（阻塞其他任务）:
   - Task 1.1 - 1.2: 核心方法实现
   - Task 2.1 - 2.2: 修改现有方法

2. **中优先级**:
   - Task 2.3: 更新出射面旋转矩阵
   - Task 3.1 - 3.2: 单元测试

3. **低优先级**（可并行）:
   - Task 4.1 - 4.2: 集成测试
   - Task 5.1 - 5.2: 属性基测试

## 与原方案对比

### 移除的任务（不再需要）
- ~~Task 1.3: 实现虚拟出射面几何创建~~
- ~~Task 1.4: 实现交点距离计算~~
- ~~Task 1.5: 实现光线传播到出射面~~

### 简化原因
新方案通过在 optiland 中添加倾斜的透明平面，让 optiland 自动完成光线传播和 OPD 计算，无需手动实现。
