# Implementation Plan: Zemax 光轴追踪与坐标转换

## Overview

本实现计划将 Zemax 序列模式光学系统转换为 optiland 全局坐标系。实现采用增量式开发，每个任务都建立在前一个任务的基础上，确保代码可以立即运行和测试。

**核心目标**：
- 实现统一的表面遍历算法，不对连续坐标断点或空气面做特殊处理
- 垂直接入 optiland 库，最小化中间层代码
- 严格遵循 Zemax 序列模式的坐标系演化规则

## Tasks

- [x] 1. 实现 CurrentCoordinateSystem 类
  - [x] 1.1 创建 `src/sequential_system/coordinate_system.py` 文件
    - 定义 `CurrentCoordinateSystem` 数据类（不可变）
    - 实现 `identity()` 类方法创建初始状态（原点 (0,0,0)，轴为单位矩阵）
    - 实现 `origin` 和 `axes` 属性
    - 实现 `x_axis`, `y_axis`, `z_axis` 属性返回列向量
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  
  - [x] 1.2 实现坐标系变换方法
    - 实现 `advance_along_z(thickness)` 方法，返回新实例
    - 实现 `apply_decenter(dx, dy)` 方法，沿当前 X/Y 轴平移
    - 实现 `apply_rotation(tilt_x, tilt_y, tilt_z)` 方法，应用组合旋转
    - _Requirements: 3.1, 3.2, 3.3_
  
  - [x]* 1.3 编写 CurrentCoordinateSystem 属性测试
    - **Property 6: 方向余弦单位向量** - 轴向量始终为单位向量
    - **Property 8: 厚度前进** - origin 沿 Z 轴正确移动
    - **Property 4: 偏心平移正确性** - 沿当前 X/Y 轴平移
    - **Validates: Requirements 1.4, 3.1, 2.4**

- [x] 2. 实现 CoordinateBreakProcessor 类
  - [x] 2.1 创建旋转矩阵辅助函数
    - 在 `coordinate_system.py` 中添加 `CoordinateBreakProcessor` 类
    - 实现 `rotation_matrix_x(angle)` 静态方法
    - 实现 `rotation_matrix_y(angle)` 静态方法
    - 实现 `rotation_matrix_z(angle)` 静态方法
    - 实现 `rotation_matrix_xyz(tilt_x, tilt_y, tilt_z)` 静态方法（R_z × R_y × R_x）
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [x] 2.2 实现坐标断点处理逻辑
    - 实现 `process()` 静态方法
    - 支持 Order=0（先平移后旋转，再厚度）
    - 支持 Order=1（先旋转后平移，再厚度）
    - 角度输入为弧度
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [x]* 2.3 编写坐标断点处理属性测试
    - **Property 1: 坐标断点 Order=0 变换正确性**
    - **Property 2: 坐标断点 Order=1 变换正确性**
    - **Property 3: 旋转顺序 X→Y→Z**
    - **Property 9: 坐标断点厚度处理**
    - **Validates: Requirements 2.1, 2.2, 2.3, 3.4**

- [x] 3. Checkpoint - 验证坐标系变换
  - 运行所有属性测试，确保通过
  - 验证 45° 旋转后 Z 轴方向正确
  - 如有问题请询问用户


- [x] 4. 实现 GlobalSurfaceDefinition 类
  - [x] 4.1 定义 GlobalSurfaceDefinition 数据类
    - 在 `coordinate_system.py` 中添加数据类
    - 包含所有必需字段：index, surface_type, vertex_position, orientation, radius, conic, is_mirror, semi_aperture, material, asphere_coeffs, comment
    - _Requirements: 8.1_
  
  - [x] 4.2 实现计算属性
    - 实现 `surface_normal` 属性（返回 -orientation[:, 2]）
    - 实现 `curvature_center` 属性（vertex + R × z_axis，平面返回 None）
    - _Requirements: 8.2, 8.3_
  
  - [x]* 4.3 编写 GlobalSurfaceDefinition 属性测试
    - **Property 12: 曲率中心计算** - 正确计算曲率中心位置
    - **Property 13: 圆锥常数保持** - conic 值不变
    - **Validates: Requirements 6.1, 6.2, 6.5, 7.1**

- [x] 5. 实现 SurfaceTraversalAlgorithm 类
  - [x] 5.1 创建遍历算法框架
    - 在 `coordinate_system.py` 中添加 `SurfaceTraversalAlgorithm` 类
    - 实现 `__init__(zmx_data)` 初始化方法
    - 实现 `traverse()` 主方法返回 GlobalSurfaceDefinition 列表
    - _Requirements: 5.1_
  
  - [x] 5.2 实现表面处理逻辑
    - 实现 `_is_virtual_surface(surface)` 判断方法
    - 实现 `_process_coordinate_break(surface)` 处理坐标断点
    - 实现 `_process_optical_surface(surface)` 处理光学表面
    - 实现 `_create_global_surface(surface)` 创建全局定义
    - _Requirements: 5.2, 5.3, 5.4, 5.5, 5.6_
  
  - [x]* 5.3 编写表面遍历属性测试
    - **Property 14: 连续坐标断点累积** - 多个坐标断点正确累积
    - **Property 15: 负厚度处理** - 负厚度正确处理
    - **Property 16: 虚拟表面与光学表面分类** - 正确分类表面类型
    - **Validates: Requirements 5.2, 5.3, 5.5, 5.6**

- [x] 6. Checkpoint - 验证表面遍历
  - 使用简单 ZMX 数据测试遍历算法
  - 验证连续坐标断点正确处理
  - 验证光学表面正确生成 GlobalSurfaceDefinition
  - 如有问题请询问用户


- [x] 7. 实现 ZemaxToOptilandConverter 类
  - [x] 7.1 创建转换器框架
    - 在 `coordinate_system.py` 中添加 `ZemaxToOptilandConverter` 类
    - 实现 `__init__(global_surfaces)` 初始化方法
    - 实现 `convert()` 主方法返回 optiland Optic 对象
    - _Requirements: 10.1, 10.5, 10.6_
  
  - [x] 7.2 实现表面添加逻辑
    - 实现 `_add_surface_to_optic(optic, surface)` 方法
    - 支持标准球面（Standard）
    - 支持偶次非球面（Even Asphere）
    - 支持平面（Flat，无穷大半径）
    - 正确设置反射镜材料
    - _Requirements: 10.2, 10.3, 10.4_
  
  - [x]* 7.3 编写 optiland 转换属性测试
    - **Property 17: optiland 表面参数传递** - 半径、conic、材料正确传递
    - **Property 18: 旋转坐标系后的曲率中心计算** - 旋转后曲率中心正确
    - **Validates: Requirements 10.2, 10.3, 10.4, 6.5**

- [x] 8. 集成与端到端测试
  - [x] 8.1 集成 ZmxParser
    - 创建便捷函数 `convert_zmx_to_optiland(zmx_file_path)` 
    - 整合 ZmxParser → SurfaceTraversalAlgorithm → ZemaxToOptilandConverter 流程
    - 添加错误处理和日志记录
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_
  
  - [x] 8.2 编写端到端集成测试
    - 测试 45° 折叠镜系统
    - 测试 Z 形双镜系统
    - 测试离轴抛物面镜（OAP）
    - 验证与 Zemax 参考数据的一致性
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [x] 9. 验证测试与反射镜行为
  - [x]* 9.1 编写反射镜坐标系属性测试
    - **Property 10: 反射镜不改变当前坐标系** - 镜面不自动旋转坐标系
    - **Property 11: 表面顶点和姿态转换** - 顶点和姿态正确记录
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**
  
  - [x] 9.2 创建验证脚本
    - 创建 `scripts/verify_coordinate_conversion.py`
    - 加载测试 ZMX 文件
    - 输出转换后的全局坐标表面定义
    - 与 Zemax 导出数据对比
    - _Requirements: 12.5_

- [x] 10. Final Checkpoint - 完整功能验证
  - 运行所有单元测试和属性测试
  - 运行端到端集成测试
  - 验证代码覆盖率 > 80%
  - 确认所有 Requirements 都有对应测试覆盖
  - 如有问题请询问用户
