# 需求文档

## 简介

元件光线追迹模块（Element Raytracer）是混合光学仿真系统的核心组件之一，负责在光学元件处执行几何光线追迹。该模块接收来自 `wavefront_sampler` 模块采样得到的光线（RealRays 对象），通过一个或多个光学表面进行追迹，并输出出射光束的光线数据。

本模块不使用偏振光线追迹，专注于标量光线追迹功能。

## 术语表

- **Element_Raytracer**：元件光线追迹器，负责将输入光线通过光学表面进行追迹
- **RealRays**：optiland 库中的光线数据结构，包含光线位置 (x, y, z)、方向余弦 (L, M, N)、OPD 等信息
- **Optical_Surface**：光学表面，定义光学元件的几何形状和材料属性
- **Entrance_Plane**：入射面，光线进入光学元件的参考平面，位于 z=0 位置
- **Exit_Plane**：出射面，光线离开光学元件的参考平面，位于最后一个元件的顶点位置
- **OPD**：光程差（Optical Path Difference），单位为波长数或毫米
- **Concave_Mirror**：凹面反射镜，曲率中心在反射面的 +Z 方向
- **Spherical_Wave**：球面波，从一点发出的波前，等相位面为球面
- **Plane_Wave**：平面波，等相位面为平面的波前

## 需求

### 需求 1：光线输入接口

**用户故事：** 作为开发者，我希望元件光线追迹模块能够接收 wavefront_sampler 模块输出的光线数据，以便实现模块间的无缝衔接。

#### 验收标准

1. WHEN 输入 RealRays 对象时，THE Element_Raytracer SHALL 验证光线数据的有效性（位置、方向余弦、OPD 等字段存在且非空）
2. WHEN 输入光线的方向余弦不满足归一化条件（L² + M² + N² ≠ 1）时，THE Element_Raytracer SHALL 抛出 ValueError 异常并提供描述性错误信息
3. THE Element_Raytracer SHALL 支持任意数量的输入光线
4. WHEN 输入光线数量为零时，THE Element_Raytracer SHALL 返回空的输出光线集合而不抛出异常

### 需求 2：光学表面定义

**用户故事：** 作为开发者，我希望能够定义一个或多个光学表面用于光线追迹，以便模拟各种光学元件。

#### 验收标准

1. THE Element_Raytracer SHALL 支持定义球面反射镜（通过曲率半径参数）
2. THE Element_Raytracer SHALL 支持定义平面反射镜（曲率半径为无穷大）
3. THE Element_Raytracer SHALL 支持定义球面折射面（通过曲率半径和材料参数）
4. WHEN 定义反射镜时，THE Element_Raytracer SHALL 接受曲率半径参数，正值表示凹面镜（曲率中心在 +Z 方向）
5. THE Element_Raytracer SHALL 支持定义多个连续的光学表面
6. WHEN 定义光学表面时，THE Element_Raytracer SHALL 接受表面半口径参数以限制有效区域

### 需求 3：入射面与出射面位置

**用户故事：** 作为开发者，我希望入射面位于 z=0 位置，与 wavefront_sampler 的输出平面重合，以便简化模块间的坐标转换。

#### 验收标准

1. THE Element_Raytracer SHALL 将入射面定位于 z=0 位置
2. THE Element_Raytracer SHALL 将出射面定位于最后一个光学表面的顶点位置
3. WHEN 只有一个反射镜时，THE Element_Raytracer SHALL 将出射面定位于该反射镜的顶点位置（z=0）
4. WHEN 输入光线的 z 坐标不为零时，THE Element_Raytracer SHALL 接受该光线并从其当前位置开始追迹

### 需求 4：光线追迹执行

**用户故事：** 作为开发者，我希望模块能够执行几何光线追迹，计算光线与光学表面的交点并更新光线方向。

#### 验收标准

1. WHEN 执行光线追迹时，THE Element_Raytracer SHALL 计算每条光线与光学表面的交点
2. WHEN 光线与反射面相交时，THE Element_Raytracer SHALL 根据反射定律更新光线方向
3. WHEN 光线与折射面相交时，THE Element_Raytracer SHALL 根据折射定律（Snell 定律）更新光线方向
4. WHEN 光线追迹完成时，THE Element_Raytracer SHALL 累计计算光线的 OPD
5. WHEN 光线未能到达光学表面（渐晕）时，THE Element_Raytracer SHALL 将该光线标记为无效

### 需求 5：输出光线数据

**用户故事：** 作为开发者，我希望获取出射光束的完整光线数据，包括位置、方向和 OPD，以便进行后续分析。

#### 验收标准

1. THE Element_Raytracer SHALL 输出 RealRays 对象，包含出射光线的位置 (x, y, z)
2. THE Element_Raytracer SHALL 输出出射光线的方向余弦 (L, M, N)
3. THE Element_Raytracer SHALL 输出出射光线的 OPD（相对于主光线，单位：波长数）
4. THE Element_Raytracer SHALL 提供方法获取有效光线的掩模（布尔数组）
5. WHEN 计算相对 OPD 时，THE Element_Raytracer SHALL 使用主光线（Px=0, Py=0）作为参考

### 需求 6：OPD 计算准确性

**用户故事：** 作为开发者，我希望 OPD 计算结果准确，以便验证光学系统的性能。

#### 验收标准

1. WHEN 球面波入射至焦距匹配的凹面反射镜时，THE Element_Raytracer SHALL 输出 OPD 为常数的平面波（OPD 标准差 < 0.01 波长）
2. WHEN 平面波入射至凹面反射镜时，THE Element_Raytracer SHALL 输出 OPD 分布符合球面波特征
3. THE Element_Raytracer SHALL 修正 optiland 相位面 OPD 计算的 1000 倍放大问题
4. FOR ALL 有效光线，THE Element_Raytracer SHALL 保证 OPD 计算的相对误差小于 0.01 波长

### 需求 7：球面波入射凹面镜测试

**用户故事：** 作为开发者，我希望通过球面波入射凹面镜的测试用例验证模块的正确性。

#### 验收标准

1. WHEN 球面波从凹面镜焦点发出并入射至该凹面镜时，THE Element_Raytracer SHALL 输出平面波
2. THE Element_Raytracer SHALL 支持设置反射镜尺寸大于光瞳尺寸
3. WHEN 测试完成时，THE Element_Raytracer SHALL 提供 OPD 可视化功能
4. THE Element_Raytracer SHALL 提供 OPD 与理论值的对比分析功能
5. FOR ALL 测试用例，THE Element_Raytracer SHALL 输出 OPD 统计信息（均值、标准差、峰谷值）

### 需求 8：错误处理

**用户故事：** 作为开发者，我希望模块能够优雅地处理各种错误情况，提供清晰的错误信息。

#### 验收标准

1. WHEN 输入参数类型错误时，THE Element_Raytracer SHALL 抛出 TypeError 异常并提供描述性错误信息
2. WHEN 输入参数值无效时，THE Element_Raytracer SHALL 抛出 ValueError 异常并提供描述性错误信息
3. WHEN 光线追迹过程中发生全反射时，THE Element_Raytracer SHALL 将该光线标记为无效而不抛出异常
4. IF 所有光线都无效，THEN THE Element_Raytracer SHALL 返回空的有效光线集合并记录警告信息
