# 设计文档：ZMX 文件可视化

## 概述

本模块提供基于 zemax-optical-axis-tracing 规范的 ZMX 文件可视化功能。**核心设计原则是最大程度复用 optiland 的可视化模块**，仅在必要时补充自定义功能。

### 设计目标

1. **最大化复用**：直接使用 optiland 的 `OpticViewer` 和 `OpticViewer3D` 进行可视化
2. **最小化代码**：仅编写必要的胶水代码连接 ZMX 解析与 optiland 可视化
3. **灵活性**：当 optiland 转换失败时，提供轻量级备选方案

### optiland 可视化模块分析

optiland 提供了完整的可视化能力：

| 模块 | 功能 | 复用可行性 |
|------|------|-----------|
| `OpticViewer` | 2D 可视化（YZ/XZ/XY 投影） | ✅ 完全可用 |
| `OpticViewer3D` | 3D VTK 可视化 | ✅ 完全可用 |
| `Surface2D/Surface3D` | 表面渲染 | ✅ 自动处理面形计算 |
| `Rays2D/Rays3D` | 光线追迹可视化 | ✅ 可选使用 |

**结论**：optiland 的可视化模块功能完整，可以直接复用。关键是确保 `ZemaxToOptilandConverter` 能正确转换 ZMX 文件。

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户接口层                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              visualize_zmx() 便捷函数                        ││
│  └──────────────────────────┬──────────────────────────────────┘│
└─────────────────────────────┼───────────────────────────────────┘
                              │
┌─────────────────────────────┼───────────────────────────────────┐
│                             ▼                                    │
│                    ZMX → optiland 转换层                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   ZmxParser     │→ │SurfaceTraversal │→ │OptilandConverter│  │
│  └─────────────────┘  └─────────────────┘  └────────┬────────┘  │
└─────────────────────────────────────────────────────┼───────────┘
                                                      │
┌─────────────────────────────────────────────────────┼───────────┐
│                                                     ▼           │
│                    optiland 可视化层（复用）                     │
│  ┌─────────────────┐  ┌─────────────────┐                       │
│  │  OpticViewer    │  │  OpticViewer3D  │                       │
│  │  (2D 可视化)    │  │  (3D VTK 可视化)│                       │
│  └─────────────────┘  └─────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

## 组件和接口

### 1. 核心便捷函数

提供一站式 ZMX 文件可视化：

```python
def visualize_zmx(
    zmx_file_path: str,
    mode: str = '2d',
    projection: str = 'YZ',
    show_rays: bool = True,
    num_rays: int = 3,
    figsize: tuple = None,
    title: str = None,
    **kwargs
) -> tuple:
    """
    可视化 ZMX 文件定义的光学系统
    
    直接复用 optiland 的 OpticViewer/OpticViewer3D 进行渲染。
    
    参数:
        zmx_file_path: ZMX 文件路径
        mode: '2d' 或 '3d'
        projection: 2D 投影平面 ('YZ', 'XZ', 'XY')，仅 mode='2d' 时有效
        show_rays: 是否显示光线追迹
        num_rays: 光线数量
        figsize: 图形大小
        title: 图形标题
        **kwargs: 传递给 OpticViewer.view() 的其他参数
    
    返回:
        mode='2d': (fig, ax, interaction_manager)
        mode='3d': None (VTK 窗口)
    
    示例:
        >>> # 2D 可视化
        >>> fig, ax, _ = visualize_zmx('system.zmx', mode='2d', projection='YZ')
        >>> plt.show()
        >>> 
        >>> # 3D 可视化
        >>> visualize_zmx('system.zmx', mode='3d')
    """
    pass
```

### 2. ZmxOpticLoader 类

封装 ZMX 到 optiland Optic 的转换：

```python
class ZmxOpticLoader:
    """ZMX 文件到 optiland Optic 的加载器"""
    
    def __init__(self, zmx_file_path: str):
        """
        参数:
            zmx_file_path: ZMX 文件路径
        """
        self.zmx_file_path = zmx_file_path
        self._zmx_data = None
        self._global_surfaces = None
        self._optic = None
    
    def load(self) -> 'Optic':
        """
        加载 ZMX 文件并转换为 optiland Optic 对象
        
        返回:
            optiland Optic 对象
        """
        from sequential_system.zmx_parser import ZmxParser
        from sequential_system.coordinate_system import (
            SurfaceTraversalAlgorithm,
            ZemaxToOptilandConverter,
        )
        
        # 1. 解析 ZMX 文件
        parser = ZmxParser(self.zmx_file_path)
        self._zmx_data = parser.parse()
        
        # 2. 遍历表面，生成全局坐标定义
        traversal = SurfaceTraversalAlgorithm(self._zmx_data)
        self._global_surfaces = traversal.traverse()
        
        # 3. 转换为 optiland Optic
        wavelength = self._zmx_data.wavelengths[0] if self._zmx_data.wavelengths else 0.55
        epd = self._zmx_data.entrance_pupil_diameter or 10.0
        
        converter = ZemaxToOptilandConverter(
            self._global_surfaces,
            wavelength=wavelength,
            entrance_pupil_diameter=epd
        )
        self._optic = converter.convert()
        
        return self._optic
    
    @property
    def zmx_data(self) -> 'ZmxDataModel':
        """获取解析后的 ZMX 数据模型"""
        return self._zmx_data
    
    @property
    def global_surfaces(self) -> list:
        """获取全局坐标表面定义列表"""
        return self._global_surfaces
    
    @property
    def optic(self) -> 'Optic':
        """获取 optiland Optic 对象"""
        return self._optic
```

### 3. 使用 optiland 可视化

直接复用 optiland 的可视化类：

```python
def view_2d(optic: 'Optic', **kwargs):
    """
    使用 optiland OpticViewer 进行 2D 可视化
    
    参数:
        optic: optiland Optic 对象
        **kwargs: 传递给 OpticViewer.view() 的参数
    
    返回:
        (fig, ax, interaction_manager)
    """
    from optiland.visualization.system.optic_viewer import OpticViewer
    
    viewer = OpticViewer(optic)
    return viewer.view(**kwargs)


def view_3d(optic: 'Optic', **kwargs):
    """
    使用 optiland OpticViewer3D 进行 3D 可视化
    
    参数:
        optic: optiland Optic 对象
        **kwargs: 传递给 OpticViewer3D.view() 的参数
    """
    from optiland.visualization.system.optic_viewer_3d import OpticViewer3D
    
    viewer = OpticViewer3D(optic)
    viewer.view(**kwargs)
```

## 数据模型

### 复用现有数据结构

本模块完全复用现有的数据结构，不引入新的数据模型：

| 数据结构 | 来源 | 用途 |
|---------|------|------|
| `ZmxDataModel` | `zmx_parser.py` | ZMX 文件解析结果 |
| `ZmxSurfaceData` | `zmx_parser.py` | 单个表面数据 |
| `GlobalSurfaceDefinition` | `coordinate_system.py` | 全局坐标表面定义 |
| `Optic` | optiland | 光学系统对象 |

## 实现流程

### 2D 可视化流程

```
1. 用户调用 visualize_zmx(path, mode='2d')
2. ZmxOpticLoader 加载并转换 ZMX 文件
3. 创建 optiland OpticViewer
4. 调用 OpticViewer.view() 渲染
5. 返回 (fig, ax, interaction_manager)
```

### 3D 可视化流程

```
1. 用户调用 visualize_zmx(path, mode='3d')
2. ZmxOpticLoader 加载并转换 ZMX 文件
3. 创建 optiland OpticViewer3D
4. 调用 OpticViewer3D.view() 启动 VTK 窗口
```

## 错误处理

| 错误类型 | 触发条件 | 处理方式 |
|---------|---------|---------|
| FileNotFoundError | ZMX 文件不存在 | 抛出描述性错误 |
| ZmxParseError | ZMX 文件格式错误 | 抛出解析错误详情 |
| ZmxUnsupportedError | 不支持的表面类型 | 抛出不支持错误 |
| OptilandConversionError | optiland 转换失败 | 记录警告，尝试部分可视化 |

## 测试策略

### 集成测试

使用 `optiland-master/tests/zemax_files/` 目录下的测试文件：

1. `complicated_fold_mirrors_setup_v2.zmx` - 复杂折叠镜系统
2. `simple_fold_mirror_up.zmx` - 简单折叠镜
3. `one_mirror_up_45deg.zmx` - 单个 45° 镜
4. `lens1.zmx`, `lens2.zmx` - 透镜系统

### 验证标准

- 2D 可视化：表面位置和方向正确
- 3D 可视化：VTK 窗口正常显示
- 光线追迹：光线路径合理

## 示例代码

### 基本使用

```python
from sequential_system.zmx_visualization import visualize_zmx
import matplotlib.pyplot as plt

# 2D 可视化
fig, ax, _ = visualize_zmx(
    'optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx',
    mode='2d',
    projection='YZ',
    title='复杂折叠镜系统'
)
plt.show()

# 3D 可视化
visualize_zmx(
    'optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx',
    mode='3d'
)
```

### 高级使用

```python
from sequential_system.zmx_visualization import ZmxOpticLoader
from optiland.visualization.system.optic_viewer import OpticViewer

# 加载 ZMX 文件
loader = ZmxOpticLoader('system.zmx')
optic = loader.load()

# 打印表面信息
for surface in loader.global_surfaces:
    print(f"表面 {surface.index}: {surface.comment}")
    print(f"  位置: {surface.vertex_position}")
    print(f"  是否反射镜: {surface.is_mirror}")

# 自定义可视化
viewer = OpticViewer(optic)
fig, ax, _ = viewer.view(
    fields='all',
    wavelengths='primary',
    num_rays=5,
    projection='YZ'
)
plt.savefig('system_layout.png', dpi=150)
```

## 文件结构

```
src/sequential_system/
├── zmx_parser.py           # 现有：ZMX 解析器
├── coordinate_system.py    # 现有：坐标转换
└── zmx_visualization.py    # 新增：可视化模块（约 150 行）

examples/
└── visualize_zmx_example.py  # 新增：示例脚本
```

## 依赖关系

- `sequential_system.zmx_parser` - ZMX 文件解析
- `sequential_system.coordinate_system` - 坐标转换
- `optiland.optic` - 光学系统对象
- `optiland.visualization.system.optic_viewer` - 2D 可视化
- `optiland.visualization.system.optic_viewer_3d` - 3D 可视化
- `matplotlib` - 2D 绑定
- `vtk` - 3D 渲染（可选）
