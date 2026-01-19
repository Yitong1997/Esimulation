# 实现计划：元件光线追迹模块 (Element Raytracer)

## 概述

本实现计划将元件光线追迹模块的设计转换为可执行的编码任务。模块基于 optiland 库实现，支持正入射和倾斜入射两种情况。

## 任务

- [x] 1. 创建核心数据结构和辅助函数
  - [x] 1.1 创建 `element_raytracer.py` 文件，定义 `SurfaceDefinition` 数据类
    - 定义表面类型、曲率半径、厚度、材料、半口径等属性
    - 实现 `to_dict()` 方法用于序列化
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6_
  
  - [x] 1.2 实现坐标转换辅助函数
    - 实现 `compute_rotation_matrix()`: 根据主光线方向计算旋转矩阵
    - 实现 `transform_rays_to_global()`: 将光线从局部坐标系转换到全局坐标系
    - 实现 `transform_rays_to_local()`: 将光线从全局坐标系转换到局部坐标系
    - _Requirements: 3.1, 3.2, 3.4_
  
  - [x] 1.3 实现便捷工厂函数
    - 实现 `create_mirror_surface()`: 创建反射镜表面定义
    - 实现 `create_concave_mirror_for_spherical_wave()`: 创建用于球面波转平面波的凹面镜
    - _Requirements: 2.1, 2.2_

- [x] 2. 实现 ElementRaytracer 核心类
  - [x] 2.1 实现 `__init__` 方法
    - 验证输入参数
    - 计算坐标转换旋转矩阵
    - 创建 optiland Optic 对象并配置光学表面
    - _Requirements: 1.1, 2.5, 8.1, 8.2_
  
  - [x] 2.2 实现 `_create_optic()` 内部方法
    - 根据 SurfaceDefinition 列表创建 optiland 光学系统
    - 配置反射镜（material='mirror'）和折射面
    - 设置表面半口径（aperture）
    - _Requirements: 2.1, 2.2, 2.3, 2.5_
  
  - [x] 2.3 实现 `trace()` 方法
    - 验证输入光线有效性
    - 将输入光线从入射面局部坐标系转换到全局坐标系
    - 调用 optiland 的 `surface_group.trace()` 进行光线追迹
    - 将输出光线从全局坐标系转换到出射面局部坐标系
    - 处理空输入情况
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 4.1, 4.4, 4.5_
  
  - [x] 2.4 实现输出方法
    - 实现 `get_output_rays()`: 返回出射光线
    - 实现 `get_relative_opd_waves()`: 计算相对于主光线的 OPD（波长数）
    - 实现 `get_valid_ray_mask()`: 返回有效光线掩模
    - 实现 `get_exit_chief_ray_direction()`: 返回出射主光线方向
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 3. 检查点 - 核心功能完成
  - 确保所有核心功能实现完成，代码可以运行
  - 如有问题请询问用户

- [x] 4. 实现测试用例
  - [x] 4.1 创建 `test_element_raytracer.py` 测试文件
    - 设置测试框架和 fixtures
    - 导入必要的模块
    - _Requirements: 7.5_
  
  - [x] 4.2 实现单元测试
    - 测试 `SurfaceDefinition` 数据类
    - 测试坐标转换函数
    - 测试空输入处理
    - 测试无效输入处理（方向余弦未归一化）
    - _Requirements: 1.2, 1.4, 8.1, 8.2_
  
  - [x] 4.3 实现球面波入射凹面镜测试（正入射）
    - 创建球面波波前
    - 使用 WavefrontToRaysSampler 采样为光线
    - 创建凹面镜并追迹
    - 验证出射 OPD 标准差 < 0.01 波长
    - 可视化 OPD 分布
    - _Requirements: 6.1, 7.1, 7.2, 7.3_
  
  - [x] 4.4 实现球面波入射倾斜平面镜测试
    - 创建球面波波前
    - 创建 45° 倾斜平面镜
    - 验证出射主光线方向正确
    - 验证出射 OPD 保持球面波特征
    - 验证 OPD 差值标准差 < 0.01 波长
    - _Requirements: 6.4_
  
  - [x] 4.5 实现属性基测试
    - **Property 1: 输入光线数量不变性**
    - **Validates: Requirements 1.3**
  
  - [x] 4.6 实现属性基测试
    - **Property 3: 球面波到平面波转换**
    - **Validates: Requirements 6.1, 7.1**

- [x] 5. 检查点 - 测试完成
  - 确保所有测试通过
  - 如有问题请询问用户

- [x] 6. 更新模块导出
  - [x] 6.1 更新 `src/wavefront_to_rays/__init__.py`
    - 导出 `ElementRaytracer` 类
    - 导出 `SurfaceDefinition` 数据类
    - 导出辅助函数
    - _Requirements: 5.1_

- [x] 7. 最终检查点
  - 确保所有测试通过
  - 确保代码符合项目规范
  - 如有问题请询问用户

## 注意事项

- 每个任务都引用了具体的需求以确保可追溯性
- 检查点用于确保增量验证
- 属性基测试使用 hypothesis 库验证普遍正确性属性
- 所有测试均为必需任务
