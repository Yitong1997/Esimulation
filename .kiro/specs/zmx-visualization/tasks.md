# 任务列表

## 任务 1：创建 ZMX 可视化模块

- [x] 1.1 创建 `src/sequential_system/zmx_visualization.py` 模块
  - [x] 1.1.1 实现 `ZmxOpticLoader` 类，封装 ZMX → optiland Optic 转换
  - [x] 1.1.2 实现 `visualize_zmx()` 便捷函数
  - [x] 1.1.3 实现 `view_2d()` 和 `view_3d()` 辅助函数

## 任务 2：创建示例脚本

- [x] 2.1 创建 `examples/visualize_zmx_example.py` 示例脚本
  - [x] 2.1.1 演示加载 `complicated_fold_mirrors_setup_v2.zmx`
  - [x] 2.1.2 演示 2D 可视化（YZ 投影）
  - [x] 2.1.3 演示 3D 可视化（可选，需要 VTK）
  - [x] 2.1.4 打印表面信息摘要

## 任务 3：验证测试

- [x] 3.1 使用多个 ZMX 测试文件验证可视化功能
  - [x] 3.1.1 测试 `complicated_fold_mirrors_setup_v2.zmx`
  - [x] 3.1.2 测试 `simple_fold_mirror_up.zmx`
  - [x] 3.1.3 测试 `lens1.zmx` 或 `lens2.zmx`
