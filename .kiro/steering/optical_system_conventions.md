<!------------------------------------------------------------------------------------
# 光学系统结构定义与坐标约定

本文件定义了混合光学仿真项目中光学系统的结构定义方式、坐标系统约定和符号规范。
基于 optiland 和 PROPER 的约定俗成进行统一。
inclusion: always
-------------------------------------------------------------------------------------> 

## 坐标系统定义

### 全局坐标系 (Global Coordinate System)

采用右手坐标系，与 optiland 保持一致：

```
        Y (向上)
        |
        |
        |________ Z (光轴方向，光传播方向)
       /
      /
     X (指向屏幕内)
```

- **Z 轴**：光轴方向，光线传播的主方向（从物方到像方为正）
- **Y 轴**：垂直向上（子午面内）
- **X 轴**：水平方向，指向屏幕内（弧矢面内）

### 局部坐标系 (Local Coordinate System)

每个光学表面都有自己的局部坐标系：
- 原点位于表面顶点
- 局部 Z 轴垂直于表面顶点处的切平面
- 局部 X、Y 轴与全局坐标系对齐（无旋转时）

### optiland 坐标系类

```python
from optiland.coordinate_system import CoordinateSystem

# 创建坐标系
cs = CoordinateSystem(
    x=0,      # X 方向平移 (mm)
    y=0,      # Y 方向平移 (mm)
    z=0,      # Z 方向平移 (mm)
    rx=0,     # 绕 X 轴旋转 (弧度)
    ry=0,     # 绕 Y 轴旋转 (弧度)
    rz=0,     # 绕 Z 轴旋转 (弧度)
    reference_cs=None  # 参考坐标系
)
```

## 符号正负号约定

### 曲率半径 (Radius of Curvature)

- **正值**：曲率中心在表面顶点的 +Z 方向（凹面朝向物方）
- **负值**：曲率中心在表面顶点的 -Z 方向（凸面朝向物方）
- **无穷大 (np.inf)**：平面

```python
# 示例：双凸透镜
lens.add_surface(index=1, radius=100.0, ...)   # 前表面，曲率中心在后方
lens.add_surface(index=2, radius=-100.0, ...)  # 后表面，曲率中心在前方
```

### 厚度 (Thickness)

- **正值**：沿 +Z 方向的距离
- 表示从当前表面顶点到下一表面顶点的距离

### 旋转角度

- **正值**：按右手定则，拇指指向轴正方向，四指弯曲方向为正旋转
- **rx**：绕 X 轴旋转（俯仰，Pitch）
- **ry**：绕 Y 轴旋转（偏航，Yaw）
- **rz**：绕 Z 轴旋转（滚转，Roll）

### 视场角

- **正值**：Y 方向向上的视场
- **负值**：Y 方向向下的视场

### 光瞳坐标

- **归一化坐标**：范围 [-1, 1]
- **Px, Py**：光瞳上的归一化坐标
- **Hx, Hy**：视场的归一化坐标

## 光学系统结构定义

### 基本结构（使用 optiland 方式）

```python
from optiland import Optic
import numpy as np

# 创建光学系统
lens = Optic()

# 1. 设置系统参数
lens.set_aperture(aperture_type='EPD', value=25.0)  # 入瞳直径
lens.set_field_type(field_type='angle')              # 视场类型
lens.add_field(y=0)                                   # 轴上视场
lens.add_field(y=10)                                  # 10度视场
lens.add_wavelength(value=0.55, is_primary=True)     # 主波长 (μm)

# 2. 添加表面
# 物面（index=0 自动创建）
lens.add_surface(index=1, radius=100.0, thickness=10.0, material='N-BK7')
lens.add_surface(index=2, radius=-100.0, thickness=50.0, material='air')
# 像面（最后一个表面）
```

### 倾斜和离轴系统定义

#### 方法一：使用相对厚度（推荐，优先使用）

使用 thickness 配合旋转参数定义离轴系统，这是推荐的标准方式：

```python
import numpy as np
from optiland import Optic

lens = Optic()
lens.set_aperture(aperture_type='EPD', value=5.0)
lens.set_field_type(field_type='angle')
lens.add_field(y=0)
lens.add_wavelength(value=0.633, is_primary=True)

# 使用 thickness 定义表面间距，配合旋转参数定义倾斜

# 物面（index=0 自动创建）

# 入瞳
lens.add_surface(index=1, thickness=10.0, is_stop=True)

# 透镜前表面
lens.add_surface(index=2, radius=100.0, thickness=5.0, material='bk7')

# 透镜后表面
lens.add_surface(index=3, radius=-100.0, thickness=10.0, material='air')

# 第一个折叠镜（45度倾斜）
# thickness 表示沿当前光轴方向到下一表面的距离
lens.add_surface(
    index=4, 
    thickness=15.0,
    material='mirror', 
    rx=-np.pi/4      # 绕 X 轴旋转 -45 度
)

# 第二个折叠镜
# 折叠后光轴方向改变，thickness 沿新的光轴方向
lens.add_surface(
    index=5, 
    thickness=20.0,
    material='mirror', 
    rx=-np.pi/4
)

# 像面
lens.add_surface(index=6, thickness=0)
```

**thickness 方式的优点：**
- 更直观地表示光学元件间的间距
- 自动处理坐标变换
- 与传统光学设计软件（如 Zemax）的定义方式一致
- 便于进行公差分析和优化

#### 方法二：使用绝对坐标（适用于特殊复杂系统）

当系统结构非常复杂，或需要精确控制每个表面的空间位置时，可以使用 (x, y, z) 绝对坐标：

```python
import numpy as np
from optiland import Optic

lens = Optic()
lens.set_aperture(aperture_type='EPD', value=5.0)
lens.set_field_type(field_type='angle')
lens.add_field(y=0)
lens.add_wavelength(value=0.633, is_primary=True)

# 使用绝对坐标定义表面位置
# 注意：使用 x, y, z 坐标后，不要再使用 thickness 参数

# 物面
lens.add_surface(index=0, radius=np.inf, z=-np.inf)

# 入瞳
lens.add_surface(index=1, z=0)

# 透镜前表面
lens.add_surface(index=2, z=10, material='bk7')

# 透镜后表面
lens.add_surface(index=3, z=15, material='air')

# 第一个折叠镜（45度倾斜）
lens.add_surface(
    index=4, 
    z=25, 
    material='mirror', 
    rx=-np.pi/4,
    is_stop=True
)

# 第二个折叠镜
lens.add_surface(
    index=5, 
    z=25, 
    y=-15,            # Y 方向偏移
    material='mirror', 
    rx=-np.pi/4
)

# 像面
lens.add_surface(index=6, z=45, y=-15)
```

### 重要注意事项

1. **优先使用 thickness 方式**
   - 除非有特殊需求，否则应优先使用 thickness 定义方式
   - thickness 方式更符合光学设计的传统习惯

2. **坐标定义方式不要混用**
   - 一旦使用 (x, y, z) 绝对坐标，后续表面都应使用绝对坐标
   - 不要在同一系统中混用 thickness 和绝对坐标

3. **折叠镜后的光轴方向**
   - 使用 thickness 方式时，折叠镜后的 thickness 沿新的光轴方向计算
   - 系统会自动处理坐标变换

4. **旋转顺序**
   - optiland 使用 XYZ 顺序的欧拉角
   - 先绕 X 轴，再绕 Y 轴，最后绕 Z 轴

## 元器件入射面与出射面定义

### 核心约定

**在本混合仿真系统中，元器件的入射面和出射面都定义在元器件的顶点位置，垂直于主光轴。**

这意味着：
1. 波前传输计算（PROPER）在元器件顶点处进行
2. OPD 计算（optiland）基于从顶点到实际表面的光程差
3. 简化了两个库之间的接口

### 实现方式

```python
class OpticalElement:
    """光学元件基类"""
    
    def __init__(self, position_z: float, optiland_surface_data: dict):
        """
        参数:
            position_z: 元件顶点在主光轴上的位置 (m)
            optiland_surface_data: optiland 表面定义数据
        """
        self.vertex_position = position_z  # 顶点位置
        self.surface_data = optiland_surface_data
        
    @property
    def entrance_plane_z(self) -> float:
        """入射面位置（顶点位置）"""
        return self.vertex_position
    
    @property
    def exit_plane_z(self) -> float:
        """出射面位置（顶点位置）"""
        return self.vertex_position
```

### 波前传输流程

```
1. PROPER 传输波前到元件入射面（顶点位置）
2. optiland 计算该元件引入的 OPD
3. 将 OPD 转换为相位，应用到 PROPER 波前
4. PROPER 从出射面（顶点位置）继续传输
```

### 示例：透镜元件

```python
class LensElement(OpticalElement):
    """透镜元件"""
    
    def __init__(self, position_z: float, front_radius: float, 
                 back_radius: float, thickness: float, material: str):
        """
        参数:
            position_z: 透镜前表面顶点位置 (m)
            front_radius: 前表面曲率半径 (m)
            back_radius: 后表面曲率半径 (m)
            thickness: 中心厚度 (m)
            material: 材料名称
        """
        self.front_vertex_z = position_z
        self.back_vertex_z = position_z + thickness
        self.front_radius = front_radius
        self.back_radius = back_radius
        self.thickness = thickness
        self.material = material
        
    def get_opd(self, optic: 'Optic', field_index: int, 
                wavelength_index: int, grid_size: int) -> np.ndarray:
        """
        计算透镜引入的 OPD
        
        返回:
            opd: OPD 网格数据 (波长数)
        """
        from optiland.wavefront import Wavefront
        wf = Wavefront(optic, field_index, wavelength_index, num_rays=grid_size)
        return wf.opd
```

### 反射镜元件

```python
class MirrorElement(OpticalElement):
    """反射镜元件"""
    
    def __init__(self, position_z: float, radius: float, 
                 tilt_x: float = 0, tilt_y: float = 0):
        """
        参数:
            position_z: 反射镜顶点位置 (m)
            radius: 曲率半径 (m)，平面镜为 np.inf
            tilt_x: 绕 X 轴倾斜角 (弧度)
            tilt_y: 绕 Y 轴倾斜角 (弧度)
        """
        self.vertex_z = position_z
        self.radius = radius
        self.tilt_x = tilt_x
        self.tilt_y = tilt_y
        
    @property
    def entrance_plane_z(self) -> float:
        """入射面位置（顶点位置）"""
        return self.vertex_z
    
    @property
    def exit_plane_z(self) -> float:
        """出射面位置（反射后仍在顶点）"""
        return self.vertex_z  # 反射镜入射和出射在同一位置
```

## PROPER 与 optiland 接口约定

### 单位转换

```python
# optiland 使用 mm，PROPER 使用 m
def mm_to_m(value_mm: float) -> float:
    return value_mm * 1e-3

def m_to_mm(value_m: float) -> float:
    return value_m * 1e3

# OPD 转相位
def opd_waves_to_phase(opd_waves: np.ndarray) -> np.ndarray:
    """OPD（波长数）转相位（弧度）"""
    return 2 * np.pi * opd_waves

def opd_meters_to_phase(opd_m: np.ndarray, wavelength_m: float) -> np.ndarray:
    """OPD（米）转相位（弧度）"""
    return 2 * np.pi * opd_m / wavelength_m
```

### 坐标系对齐

```python
def align_grids(optiland_opd: np.ndarray, proper_grid_size: int,
                optiland_pupil_radius: float, proper_pupil_radius: float) -> np.ndarray:
    """
    将 optiland 计算的 OPD 网格对齐到 PROPER 波前网格
    
    参数:
        optiland_opd: optiland 计算的 OPD 数据
        proper_grid_size: PROPER 波前网格大小
        optiland_pupil_radius: optiland 中的光瞳半径 (mm)
        proper_pupil_radius: PROPER 中的光瞳半径 (m)
    
    返回:
        aligned_opd: 对齐后的 OPD 数据
    """
    from scipy.interpolate import RectBivariateSpline
    
    # optiland 使用归一化坐标 [-1, 1]
    n_optiland = optiland_opd.shape[0]
    x_optiland = np.linspace(-1, 1, n_optiland)
    
    # PROPER 使用物理坐标
    x_proper = np.linspace(-proper_pupil_radius, proper_pupil_radius, proper_grid_size)
    x_proper_normalized = x_proper / proper_pupil_radius
    
    # 插值
    interp = RectBivariateSpline(x_optiland, x_optiland, optiland_opd)
    aligned_opd = interp(x_proper_normalized, x_proper_normalized)
    
    return aligned_opd
```

## 标准光学系统模板

### 简单透镜系统

```python
def create_simple_lens(focal_length: float, aperture: float, 
                       wavelength: float) -> Optic:
    """
    创建简单透镜系统
    
    参数:
        focal_length: 焦距 (mm)
        aperture: 入瞳直径 (mm)
        wavelength: 波长 (μm)
    """
    lens = Optic()
    lens.set_aperture(aperture_type='EPD', value=aperture)
    lens.set_field_type(field_type='angle')
    lens.add_field(y=0)
    lens.add_wavelength(value=wavelength, is_primary=True)
    
    # 简化：使用薄透镜近似
    lens.add_surface(index=0, radius=np.inf, z=-np.inf)
    lens.add_surface(index=1, z=0, is_stop=True)
    lens.add_surface(index=2, z=focal_length)  # 像面
    
    return lens
```

### 折叠光路系统

```python
def create_folded_system() -> Optic:
    """创建带折叠镜的光学系统（使用 thickness 方式）"""
    lens = Optic()
    lens.set_aperture(aperture_type='EPD', value=10.0)
    lens.set_field_type(field_type='angle')
    lens.add_field(y=0)
    lens.add_wavelength(value=0.633, is_primary=True)
    
    # 物面（index=0 自动创建）
    
    # 入瞳
    lens.add_surface(index=1, thickness=100.0, is_stop=True)
    
    # 45度折叠镜
    # thickness 表示沿当前光轴到下一表面的距离
    lens.add_surface(
        index=2, 
        thickness=100.0,
        material='mirror',
        rx=-np.pi/4  # 45度倾斜，光路向 -Y 方向折叠
    )
    
    # 像面（折叠后沿新光轴方向）
    lens.add_surface(index=3, thickness=0)
    
    return lens
```
