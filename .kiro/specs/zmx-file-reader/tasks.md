# Implementation Plan: ZMX File Reader

## Overview

本实现计划将 ZMX 文件读取功能分解为可执行的编码任务。实现将使用 Python，遵循项目现有的代码风格和模块组织。

## Tasks

- [x] 1. 创建数据模型和基础结构
  - [x] 1.1 创建 `src/sequential_system/zmx_parser.py` 文件，定义 `ZmxSurfaceData` 和 `ZmxDataModel` 数据类
    - 定义 ZmxSurfaceData 包含所有表面参数（index, surface_type, radius, thickness, conic, material, is_mirror, is_stop, semi_diameter, decenter_x/y, tilt_x/y/z_deg, asphere_coeffs, comment）
    - 定义 ZmxDataModel 包含 surfaces 字典、wavelengths 列表、primary_wavelength_index、entrance_pupil_diameter
    - 添加辅助方法 get_surface() 和 get_mirror_surfaces()
    - _Requirements: 4.1, 4.2_

  - [x] 1.2 定义异常类 `ZmxParseError`, `ZmxUnsupportedError`, `ZmxConversionError`
    - ZmxParseError 包含 line_number 和 line_content 属性
    - 实现 _format_message() 方法生成详细错误信息
    - _Requirements: 8.1, 8.3, 8.4_

- [x] 2. 实现 ZMX 文件解析器
  - [x] 2.1 实现 `ZmxParser` 类的文件读取功能
    - 实现 _try_read_file() 方法，尝试 UTF-16、UTF-8、ISO-8859-1 编码
    - 处理文件不存在错误
    - _Requirements: 1.1, 1.2, 1.5_

  - [x] 2.2 实现操作符解析分发机制
    - 创建操作符到解析方法的映射字典
    - 实现 parse() 主方法和 _parse_line() 方法
    - 实现 _parse_mode() 验证序列模式
    - _Requirements: 1.3, 1.4_

  - [x] 2.3 实现表面相关操作符解析
    - 实现 _parse_surface() 处理 SURF 操作符
    - 实现 _parse_type() 处理 TYPE 操作符
    - 实现 _parse_curv() 处理 CURV 操作符（曲率转半径）
    - 实现 _parse_disz() 处理 DISZ 操作符
    - 实现 _parse_coni() 处理 CONI 操作符
    - 实现 _parse_diam() 处理 DIAM 操作符
    - 实现 _parse_stop() 处理 STOP 操作符
    - 实现 _parse_comm() 处理 COMM 操作符
    - _Requirements: 2.1, 2.5, 2.6_

  - [x] 2.4 实现材料和波长解析
    - 实现 _parse_glas() 处理 GLAS 操作符，识别 MIRROR 材料
    - 实现 _parse_wavm() 处理 WAVM 操作符
    - 实现 _parse_enpd() 处理 ENPD 操作符
    - _Requirements: 2.3, 2.4, 4.3, 4.4_

  - [x] 2.5 实现坐标断点解析
    - 实现 _parse_parm() 处理 PARM 操作符
    - 对于 COORDBRK 类型，提取 decenter_x/y 和 tilt_x/y/z
    - 将角度从度转换为弧度存储
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 2.6 编写 ZmxParser 单元测试
    - 测试各种编码的文件读取
    - 测试标准表面解析
    - 测试坐标断点解析
    - 测试错误处理
    - _Requirements: 1.1-1.5, 2.1-2.6, 3.1-3.3_

- [x] 3. Checkpoint - 验证解析器功能
  - 确保所有解析器测试通过，使用 one_mirror_up_45deg.zmx 验证基本功能

- [x] 4. 实现元件转换器
  - [x] 4.1 创建 `src/sequential_system/zmx_converter.py` 文件，定义 `CoordinateTransform` 和 `ConvertedElement` 数据类
    - CoordinateTransform 包含累积的偏心和倾斜
    - 实现 apply_coordinate_break() 和 reset() 方法
    - ConvertedElement 包含元件、ZMX 索引、注释、is_fold_mirror、fold_angle_deg
    - _Requirements: 3.4, 6.2_

  - [x] 4.2 实现 `ElementConverter` 类的基础结构
    - 初始化方法接收 ZmxDataModel
    - 定义 FOLD_ANGLE_THRESHOLD = 5.0 度
    - 实现 convert() 主方法框架
    - _Requirements: 7.1_

  - [x] 4.3 实现反射镜元件创建逻辑
    - 实现 _create_mirror_element() 方法
    - 根据 radius 和 conic 判断创建 FlatMirror、ParabolicMirror 或 SphericalMirror
    - 实现 _is_fold_mirror() 判断是否为折叠镜
    - _Requirements: 5.1, 5.2, 5.3, 5.7, 5.8_

  - [x] 4.4 实现坐标断点处理和折叠镜序列检测
    - 实现 _process_coordinate_break() 累积坐标变换
    - 实现 _process_mirror_surface() 处理反射镜
    - 实现 _calculate_thickness_after_reflection() 计算反射后传播距离
    - 处理负厚度作为反射方向传播
    - _Requirements: 3.4, 3.5, 3.6, 6.1, 6.3, 6.4, 6.5_

  - [x] 4.5 实现表面遍历和元件生成
    - 实现 _process_surfaces() 遍历所有表面
    - 跳过物面（index=0）和像面（最后一个）
    - 正确设置每个元件的 thickness、semi_aperture、tilt_x、tilt_y、decenter_x、decenter_y、is_fold
    - _Requirements: 7.2, 7.4, 7.5_

  - [x] 4.6 编写 ElementConverter 单元测试
    - 测试反射镜类型分类
    - 测试折叠镜检测
    - 测试坐标变换累积
    - 测试厚度计算
    - _Requirements: 5.1-5.8, 6.1-6.5, 7.1-7.5_

- [x] 5. Checkpoint - 验证转换器功能
  - 确保所有转换器测试通过，使用 complicated_fold_mirrors_setup_v2.zmx 验证复杂折叠光路

- [x] 6. 实现代码生成器
  - [x] 6.1 实现 `CodeGenerator` 类
    - 定义 INDENT = "    " 常量
    - 实现 generate() 主方法
    - 实现 _generate_imports() 生成 import 语句
    - _Requirements: 9.1, 9.7_

  - [x] 6.2 实现元件代码生成
    - 实现 _generate_element_code() 为每个元件生成代码
    - 包含所有参数（thickness, semi_aperture, tilt_x, tilt_y, decenter_x, decenter_y, is_fold）
    - 添加注释标明原始 ZMX 表面索引
    - 对折叠镜添加折叠角度注释
    - 实现 _format_float() 格式化浮点数
    - _Requirements: 9.2, 9.3, 9.4, 9.5, 9.6_

  - [x] 6.3 在 ElementConverter 中集成代码生成
    - 实现 ElementConverter.generate_code() 方法
    - 调用 CodeGenerator 生成代码
    - _Requirements: 9.1_

  - [x] 6.4 编写代码生成器测试
    - 测试生成的代码可执行
    - 测试参数完整性
    - 测试注释正确性
    - 测试格式正确性
    - _Requirements: 9.1-9.7_

- [x] 7. 实现便捷函数和模块导出
  - [x] 7.1 实现便捷函数 `load_zmx_file()` 和 `load_zmx_and_generate_code()`
    - load_zmx_file() 返回 List[OpticalElement]
    - load_zmx_and_generate_code() 返回 Tuple[List[OpticalElement], str]
    - _Requirements: 7.1_

  - [x] 7.2 更新 `src/sequential_system/__init__.py` 导出新模块
    - 导出 ZmxParser, ZmxDataModel, ZmxSurfaceData
    - 导出 ElementConverter, ConvertedElement
    - 导出 load_zmx_file, load_zmx_and_generate_code
    - 导出异常类
    - _Requirements: 7.1_

- [x] 8. 集成测试和验证
  - [x] 8.1 编写 complicated_fold_mirrors_setup_v2.zmx 集成测试
    - 验证所有反射镜被正确识别
    - 验证所有坐标断点被正确提取
    - 验证折叠镜序列的 is_fold 标志正确
    - 验证光程长度计算正确
    - _Requirements: 10.1, 10.2, 10.3, 10.6_

  - [x] 8.2 编写 one_mirror_up_45deg.zmx 集成测试
    - 验证单个 45 度折叠镜被正确识别
    - 验证 tilt_x 或 tilt_y 为 π/4
    - _Requirements: 10.4_

  - [x] 8.3 编写所有 ZMX 文件的有效性测试
    - 遍历 optiland-master/tests/zemax_files/ 中的所有 .zmx 文件
    - 验证每个文件都能成功解析
    - 验证生成的元件可以添加到 SequentialOpticalSystem
    - _Requirements: 10.5_

  - [x] 8.4 编写属性基测试
    - **Property 1**: 表面数据提取完整性
    - **Property 2**: 坐标断点参数提取
    - **Property 3**: 反射镜类型分类
    - **Property 4**: 折叠镜检测和配置
    - **Property 7**: 代码生成往返测试
    - _Requirements: 2.1-2.6, 3.1-3.3, 5.1-5.8, 9.1-9.3_

- [x] 9. Final Checkpoint - 完整功能验证
  - 确保所有测试通过
  - 使用 complicated_fold_mirrors_setup_v2.zmx 进行端到端测试
  - 验证生成的代码可以直接复制使用

## Notes

- 每个任务都引用了具体的需求条款以确保可追溯性
- Checkpoint 任务用于验证阶段性成果
- 属性基测试使用 hypothesis 库，每个测试至少运行 100 次迭代
- 所有测试任务都是必需的，确保全面的测试覆盖
