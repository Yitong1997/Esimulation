"""全局坐标系元件光线追迹模块

本模块提供在全局坐标系中进行光线追迹的功能，避免局部坐标系转换。

主要组件：
- PlaneDef: 平面定义数据类（位置和法向量）
- GlobalSurfaceDefinition: 全局坐标系表面定义数据类
- GlobalElementRaytracer: 全局坐标系元件光线追迹器类

基于 optiland 库的绝对坐标定位能力实现。

设计目标：
1. 完全复用 ElementRaytracer 的计算逻辑（OPD 计算、抛物面反射修正等）
2. 全局坐标系操作：避免局部坐标系转换
3. 正确处理 tilt 参数：当前 ElementRaytracer 对抛物面忽略了 tilt_x/y
4. 简化 optiland 集成：利用 optiland 的绝对坐标定位能力
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray

from optiland.rays import RealRays


@dataclass
class PlaneDef:
    """平面定义（全局坐标系）
    
    用于定义入射面和出射面。平面由一个点和法向量唯一确定。
    
    参数:
        position: 平面上的点 (x, y, z)，单位：mm
            - 通常为主光线与平面的交点
        normal: 法向量 (nx, ny, nz)，必须归一化
            - 满足 nx² + ny² + nz² = 1
            - 指向光传播方向（入射面）或出射方向（出射面）
    
    示例:
        >>> # 创建垂直于 Z 轴的入射面，原点在 (0, 0, 100)
        >>> entrance_plane = PlaneDef(
        ...     position=(0.0, 0.0, 100.0),
        ...     normal=(0.0, 0.0, 1.0)
        ... )
        
        >>> # 创建倾斜 45° 的出射面
        >>> import math
        >>> exit_plane = PlaneDef(
        ...     position=(0.0, 0.0, 200.0),
        ...     normal=(0.0, math.sin(math.pi/4), math.cos(math.pi/4))
        ... )
    
    注意:
        - 法向量必须归一化，否则在使用时会抛出 ValueError
        - 法向量归一化验证在 GlobalElementRaytracer 中进行
        - 位置坐标使用全局坐标系（Z 轴为初始光轴方向）
    
    Validates:
        - Requirements 1.1: 接受入射面的全局位置 (x, y, z) 和法向量 (nx, ny, nz) 作为参数
        - Requirements 1.2: 验证法向量已归一化（L² + M² + N² = 1）
        - Requirements 1.3: 法向量未归一化时抛出 ValueError 并提供清晰的错误信息
    """
    position: Tuple[float, float, float]  # 平面上的点 (x, y, z)
    normal: Tuple[float, float, float]    # 法向量 (nx, ny, nz)，必须归一化


@dataclass
class GlobalSurfaceDefinition:
    """全局坐标系表面定义
    
    与 SurfaceDefinition 类似，但使用全局坐标定义位置和朝向。
    用于在全局坐标系中定义光学表面，避免局部坐标系转换。
    
    参数:
        surface_type: 表面类型
            - 'mirror': 反射镜
            - 'refract': 折射面
        radius: 曲率半径，单位：mm
            - 正值表示凹面（曲率中心在表面法向量方向）
            - 负值表示凸面（曲率中心在表面法向量反方向）
            - np.inf 表示平面
        conic: 圆锥常数，默认 0.0
            - 0: 球面
            - -1: 抛物面
            - < -1: 双曲面
            - -1 < k < 0: 椭球面（扁椭球）
            - > 0: 椭球面（长椭球）
        material: 材料名称
            - 'mirror': 反射镜
            - 其他字符串: 折射材料名称（如 'N-BK7', 'air' 等）
        vertex_position: 顶点位置 (x, y, z)，单位：mm
            - 表面顶点在全局坐标系中的绝对位置
            - 默认为 (0, 0, 0)
        surface_normal: 表面法向量 (nx, ny, nz)
            - 表面在顶点处的法向量
            - 指向光入射方向（即与入射光方向相反）
            - 默认为 (0, 0, -1)，表示面向 -Z 方向
        tilt_x: 绕 X 轴旋转角度（弧度），默认 0.0
        tilt_y: 绕 Y 轴旋转角度（弧度），默认 0.0
        tilt_z: 绕 Z 轴旋转角度（弧度），默认 0.0
    
    示例:
        >>> # 创建平面反射镜，位于 z=100，面向 -Z 方向
        >>> flat_mirror = GlobalSurfaceDefinition(
        ...     surface_type='mirror',
        ...     radius=np.inf,
        ...     vertex_position=(0.0, 0.0, 100.0),
        ...     surface_normal=(0.0, 0.0, -1.0),
        ... )
        
        >>> # 创建离轴抛物面镜，顶点在 (0, 100, 0)
        >>> oap = GlobalSurfaceDefinition(
        ...     surface_type='mirror',
        ...     radius=200.0,
        ...     conic=-1.0,
        ...     vertex_position=(0.0, 100.0, 0.0),
        ...     surface_normal=(0.0, 0.0, -1.0),
        ... )
    
    注意:
        - 离轴效果通过 vertex_position 的 (x, y) 坐标自然实现
        - 不使用 off_axis_distance、dy、dx 等参数
        - 表面法向量用于确定表面朝向，与 tilt 参数配合使用
    
    Validates:
        - Requirements 4.1: 接受顶点位置 (x, y, z) 作为绝对坐标
        - Requirements 4.2: 接受旋转角度 (rx, ry, rz) 定义表面朝向
        - Requirements 4.4: 正确处理离轴系统（如 OAP）的表面定义
        - Requirements 4.5: 通过顶点位置自然实现离轴效果
    """
    surface_type: str = 'mirror'
    radius: float = field(default_factory=lambda: np.inf)
    conic: float = 0.0
    material: str = 'mirror'
    
    # 全局坐标定位
    vertex_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    surface_normal: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    
    # 旋转角度（可选，用于倾斜表面）
    tilt_x: float = 0.0  # 绕 X 轴旋转 (rad)
    tilt_y: float = 0.0  # 绕 Y 轴旋转 (rad)
    tilt_z: float = 0.0  # 绕 Z 轴旋转 (rad)

    
    def __post_init__(self) -> None:
        """初始化后验证参数有效性"""
        # 验证表面类型
        valid_types = ('mirror', 'refract')
        if self.surface_type not in valid_types:
            raise ValueError(
                f"无效的表面类型：'{self.surface_type}'。"
                f"有效类型为：{valid_types}"
            )
        
        # 验证曲率半径
        if not isinstance(self.radius, (int, float)):
            raise TypeError(
                f"曲率半径类型错误：期望 int 或 float，"
                f"实际为 {type(self.radius).__name__}"
            )
        
        # 验证圆锥常数
        if not isinstance(self.conic, (int, float)):
            raise TypeError(
                f"圆锥常数类型错误：期望 int 或 float，"
                f"实际为 {type(self.conic).__name__}"
            )
        if not np.isfinite(self.conic):
            raise ValueError(
                f"圆锥常数必须为有限值，实际为 {self.conic}"
            )
        
        # 验证材料名称
        if not isinstance(self.material, str):
            raise TypeError(
                f"材料名称类型错误：期望 str，"
                f"实际为 {type(self.material).__name__}"
            )
        if not self.material:
            raise ValueError("材料名称不能为空字符串")
        
        # 验证顶点位置
        if not isinstance(self.vertex_position, (tuple, list)):
            raise TypeError(
                f"vertex_position 类型错误：期望 tuple 或 list，"
                f"实际为 {type(self.vertex_position).__name__}"
            )
        if len(self.vertex_position) != 3:
            raise ValueError(
                f"vertex_position 必须包含 3 个元素 (x, y, z)，"
                f"实际包含 {len(self.vertex_position)} 个元素"
            )
        for i, val in enumerate(self.vertex_position):
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"vertex_position[{i}] 类型错误：期望 int 或 float，"
                    f"实际为 {type(val).__name__}"
                )
            if not np.isfinite(val):
                raise ValueError(
                    f"vertex_position[{i}] 必须为有限值，实际为 {val}"
                )

        
        # 验证表面法向量
        if not isinstance(self.surface_normal, (tuple, list)):
            raise TypeError(
                f"surface_normal 类型错误：期望 tuple 或 list，"
                f"实际为 {type(self.surface_normal).__name__}"
            )
        if len(self.surface_normal) != 3:
            raise ValueError(
                f"surface_normal 必须包含 3 个元素 (nx, ny, nz)，"
                f"实际包含 {len(self.surface_normal)} 个元素"
            )
        for i, val in enumerate(self.surface_normal):
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"surface_normal[{i}] 类型错误：期望 int 或 float，"
                    f"实际为 {type(val).__name__}"
                )
            if not np.isfinite(val):
                raise ValueError(
                    f"surface_normal[{i}] 必须为有限值，实际为 {val}"
                )
        
        # 验证倾斜角度
        for name, value in [('tilt_x', self.tilt_x), 
                           ('tilt_y', self.tilt_y), 
                           ('tilt_z', self.tilt_z)]:
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} 类型错误：期望 int 或 float，"
                    f"实际为 {type(value).__name__}"
                )
            if not np.isfinite(value):
                raise ValueError(
                    f"{name} 必须为有限值，实际为 {value}"
                )
    
    @property
    def is_mirror(self) -> bool:
        """判断是否为反射镜"""
        return self.surface_type == 'mirror'
    
    @property
    def is_plane(self) -> bool:
        """判断是否为平面"""
        return np.isinf(self.radius)
    
    @property
    def is_parabola(self) -> bool:
        """判断是否为抛物面"""
        return np.isclose(self.conic, -1.0)
    
    @property
    def focal_length(self) -> Optional[float]:
        """计算焦距（仅对反射镜有效）
        
        对于反射镜，焦距 f = R/2
        对于平面镜，返回 None
        
        返回:
            焦距（mm），平面镜返回 None
        """
        if self.is_plane:
            return None
        if self.is_mirror:
            return self.radius / 2.0
        return None

    
    def to_dict(self) -> Dict[str, Any]:
        """将表面定义转换为字典
        
        用于序列化和调试输出。
        
        返回:
            包含所有属性的字典
        """
        return {
            'surface_type': self.surface_type,
            'radius': float(self.radius) if np.isfinite(self.radius) else 'inf',
            'conic': self.conic,
            'material': self.material,
            'vertex_position': self.vertex_position,
            'surface_normal': self.surface_normal,
            'tilt_x': self.tilt_x,
            'tilt_y': self.tilt_y,
            'tilt_z': self.tilt_z,
        }
    
    def __repr__(self) -> str:
        """返回表面定义的字符串表示"""
        radius_str = 'inf' if np.isinf(self.radius) else f'{self.radius:.2f}'
        conic_str = f'{self.conic:.2f}' if self.conic != 0.0 else '0'
        pos_str = f'({self.vertex_position[0]:.1f}, {self.vertex_position[1]:.1f}, {self.vertex_position[2]:.1f})'
        
        result = (
            f"GlobalSurfaceDefinition("
            f"type='{self.surface_type}', "
            f"radius={radius_str} mm, "
            f"conic={conic_str}, "
            f"material='{self.material}', "
            f"vertex={pos_str}"
        )
        
        # 添加倾斜信息（如果有）
        if self.tilt_x != 0.0 or self.tilt_y != 0.0 or self.tilt_z != 0.0:
            tilt_x_deg = np.degrees(self.tilt_x)
            tilt_y_deg = np.degrees(self.tilt_y)
            tilt_z_deg = np.degrees(self.tilt_z)
            result += f", tilt=({tilt_x_deg:.1f}°, {tilt_y_deg:.1f}°, {tilt_z_deg:.1f}°)"
        
        result += ")"
        return result



# =============================================================================
# 辅助函数
# =============================================================================

def _validate_normal_vector(
    normal: Tuple[float, float, float],
    name: str = "法向量",
    tolerance: float = 1e-6,
) -> None:
    """验证法向量是否归一化
    
    参数:
        normal: 法向量 (nx, ny, nz)
        name: 参数名称，用于错误信息
        tolerance: 归一化容差
    
    异常:
        ValueError: 如果法向量未归一化
    
    Validates:
        - Requirements 1.2: 验证法向量已归一化（L² + M² + N² = 1）
        - Requirements 1.3: 法向量未归一化时抛出 ValueError 并提供清晰的错误信息
        - Requirements 10.1: 入射面法向量未归一化时抛出 ValueError
    """
    normal_array = np.array(normal, dtype=np.float64)
    norm_squared = np.sum(normal_array ** 2)
    
    if not np.isclose(norm_squared, 1.0, rtol=tolerance):
        norm = np.sqrt(norm_squared)
        raise ValueError(
            f"{name}未归一化：|n| = {norm:.6f}，期望为 1.0。"
            f"请确保 nx² + ny² + nz² = 1。"
        )



# =============================================================================
# GlobalElementRaytracer 类
# =============================================================================

class GlobalElementRaytracer:
    """全局坐标系元件光线追迹器
    
    在全局坐标系中进行光线追迹，避免局部坐标系转换。
    完全复用 ElementRaytracer 的计算逻辑（OPD 计算、抛物面反射修正等）。
    
    与 ElementRaytracer 的主要区别：
    
    | 方面 | ElementRaytracer | GlobalElementRaytracer |
    |------|------------------|------------------------|
    | 坐标系 | 入射面局部坐标系 | 全局坐标系 |
    | 表面定义 | 相对于入射面偏移 | 绝对位置 (x, y, z) |
    | 入射面 | 隐式定义（原点） | 显式定义（点+法向量） |
    | 出射面 | 通过旋转矩阵转换 | 显式定义（点+法向量） |
    | OPD 计算 | 带符号 OPD | 带符号 OPD（相同算法） |
    | 抛物面处理 | 忽略 tilt 参数 | 正确处理 tilt 参数 |
    
    参数:
        surfaces: 光学表面定义列表（全局坐标），至少包含一个表面
        wavelength: 波长，单位：μm，必须为正值
        entrance_plane: 入射面定义（点+法向量）
        exit_plane: 出射面定义（可选，自动计算）
    
    属性:
        surfaces: 光学表面定义列表
        wavelength: 波长（μm）
        entrance_plane: 入射面定义
        exit_plane: 出射面定义（追迹主光线后设置）
        optic: optiland Optic 对象
        output_rays: 出射光线数据
    
    示例:
        >>> # 创建平面反射镜光线追迹器
        >>> mirror = GlobalSurfaceDefinition(
        ...     surface_type='mirror',
        ...     radius=np.inf,
        ...     vertex_position=(0.0, 0.0, 100.0),
        ... )
        >>> entrance = PlaneDef(
        ...     position=(0.0, 0.0, 50.0),
        ...     normal=(0.0, 0.0, 1.0),
        ... )
        >>> raytracer = GlobalElementRaytracer(
        ...     surfaces=[mirror],
        ...     wavelength=0.633,
        ...     entrance_plane=entrance,
        ... )
    
    Validates:
        - Requirements 1.1-1.5: 全局坐标系入射面定义
        - Requirements 3.1-3.6: 全局坐标系光线追迹
        - Requirements 4.1-4.5: 表面定义与全局坐标
        - Requirements 10.1-10.4: 错误处理
    """

    
    def __init__(
        self,
        surfaces: List[GlobalSurfaceDefinition],
        wavelength: float,
        entrance_plane: PlaneDef,
        exit_plane: Optional[PlaneDef] = None,
    ) -> None:
        """初始化全局坐标系光线追迹器
        
        参数:
            surfaces: 光学表面定义列表（全局坐标），至少包含一个表面
            wavelength: 波长，单位：μm，必须为正值
            entrance_plane: 入射面定义（点+法向量）
            exit_plane: 出射面定义（可选，自动计算）
        
        异常:
            TypeError: 如果输入参数类型错误
            ValueError: 如果输入参数值无效
        
        Validates:
            - Requirements 1.1: 接受入射面的全局位置和法向量作为参数
            - Requirements 1.2: 验证法向量已归一化
            - Requirements 1.3: 法向量未归一化时抛出 ValueError
            - Requirements 10.1: 入射面法向量未归一化时抛出 ValueError
            - Requirements 10.2: 表面定义参数无效时抛出 ValueError
            - Requirements 10.4: 光线方向余弦未归一化时抛出 ValueError
        """
        # =====================================================================
        # 验证 surfaces 参数
        # =====================================================================
        
        # 检查类型：必须是列表
        if not isinstance(surfaces, list):
            raise TypeError(
                f"surfaces 参数类型错误：期望 list，"
                f"实际为 {type(surfaces).__name__}"
            )
        
        # 检查非空
        if len(surfaces) == 0:
            raise ValueError("surfaces 列表不能为空，至少需要一个光学表面")
        
        # 检查列表元素类型
        for i, surface in enumerate(surfaces):
            if not isinstance(surface, GlobalSurfaceDefinition):
                raise TypeError(
                    f"surfaces[{i}] 类型错误：期望 GlobalSurfaceDefinition，"
                    f"实际为 {type(surface).__name__}"
                )

        
        # =====================================================================
        # 验证 wavelength 参数
        # =====================================================================
        
        # 检查类型
        if not isinstance(wavelength, (int, float)):
            raise TypeError(
                f"wavelength 参数类型错误：期望 int 或 float，"
                f"实际为 {type(wavelength).__name__}"
            )
        
        # 检查值：必须为正
        if wavelength <= 0:
            raise ValueError(
                f"wavelength 必须为正值，实际为 {wavelength} μm"
            )
        
        # 检查是否为有限值
        if not np.isfinite(wavelength):
            raise ValueError(
                f"wavelength 必须为有限值，实际为 {wavelength}"
            )
        
        # =====================================================================
        # 验证 entrance_plane 参数
        # =====================================================================
        
        # 检查类型
        if not isinstance(entrance_plane, PlaneDef):
            raise TypeError(
                f"entrance_plane 参数类型错误：期望 PlaneDef，"
                f"实际为 {type(entrance_plane).__name__}"
            )
        
        # 验证入射面位置
        if not isinstance(entrance_plane.position, (tuple, list)):
            raise TypeError(
                f"entrance_plane.position 类型错误：期望 tuple 或 list，"
                f"实际为 {type(entrance_plane.position).__name__}"
            )
        if len(entrance_plane.position) != 3:
            raise ValueError(
                f"entrance_plane.position 必须包含 3 个元素 (x, y, z)，"
                f"实际包含 {len(entrance_plane.position)} 个元素"
            )
        for i, val in enumerate(entrance_plane.position):
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"entrance_plane.position[{i}] 类型错误：期望 int 或 float，"
                    f"实际为 {type(val).__name__}"
                )
            if not np.isfinite(val):
                raise ValueError(
                    f"entrance_plane.position[{i}] 必须为有限值，实际为 {val}"
                )

        
        # 验证入射面法向量
        if not isinstance(entrance_plane.normal, (tuple, list)):
            raise TypeError(
                f"entrance_plane.normal 类型错误：期望 tuple 或 list，"
                f"实际为 {type(entrance_plane.normal).__name__}"
            )
        if len(entrance_plane.normal) != 3:
            raise ValueError(
                f"entrance_plane.normal 必须包含 3 个元素 (nx, ny, nz)，"
                f"实际包含 {len(entrance_plane.normal)} 个元素"
            )
        for i, val in enumerate(entrance_plane.normal):
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"entrance_plane.normal[{i}] 类型错误：期望 int 或 float，"
                    f"实际为 {type(val).__name__}"
                )
            if not np.isfinite(val):
                raise ValueError(
                    f"entrance_plane.normal[{i}] 必须为有限值，实际为 {val}"
                )
        
        # 验证入射面法向量归一化
        _validate_normal_vector(
            entrance_plane.normal,
            name="入射面法向量 (entrance_plane.normal)"
        )
        
        # =====================================================================
        # 验证 exit_plane 参数（可选）
        # =====================================================================
        
        if exit_plane is not None:
            # 检查类型
            if not isinstance(exit_plane, PlaneDef):
                raise TypeError(
                    f"exit_plane 参数类型错误：期望 PlaneDef 或 None，"
                    f"实际为 {type(exit_plane).__name__}"
                )
            
            # 验证出射面位置
            if not isinstance(exit_plane.position, (tuple, list)):
                raise TypeError(
                    f"exit_plane.position 类型错误：期望 tuple 或 list，"
                    f"实际为 {type(exit_plane.position).__name__}"
                )
            if len(exit_plane.position) != 3:
                raise ValueError(
                    f"exit_plane.position 必须包含 3 个元素 (x, y, z)，"
                    f"实际包含 {len(exit_plane.position)} 个元素"
                )
            for i, val in enumerate(exit_plane.position):
                if not isinstance(val, (int, float)):
                    raise TypeError(
                        f"exit_plane.position[{i}] 类型错误：期望 int 或 float，"
                        f"实际为 {type(val).__name__}"
                    )
                if not np.isfinite(val):
                    raise ValueError(
                        f"exit_plane.position[{i}] 必须为有限值，实际为 {val}"
                    )

            
            # 验证出射面法向量
            if not isinstance(exit_plane.normal, (tuple, list)):
                raise TypeError(
                    f"exit_plane.normal 类型错误：期望 tuple 或 list，"
                    f"实际为 {type(exit_plane.normal).__name__}"
                )
            if len(exit_plane.normal) != 3:
                raise ValueError(
                    f"exit_plane.normal 必须包含 3 个元素 (nx, ny, nz)，"
                    f"实际包含 {len(exit_plane.normal)} 个元素"
                )
            for i, val in enumerate(exit_plane.normal):
                if not isinstance(val, (int, float)):
                    raise TypeError(
                        f"exit_plane.normal[{i}] 类型错误：期望 int 或 float，"
                        f"实际为 {type(val).__name__}"
                    )
                if not np.isfinite(val):
                    raise ValueError(
                        f"exit_plane.normal[{i}] 必须为有限值，实际为 {val}"
                    )
            
            # 验证出射面法向量归一化
            _validate_normal_vector(
                exit_plane.normal,
                name="出射面法向量 (exit_plane.normal)"
            )
        
        # =====================================================================
        # 存储属性
        # =====================================================================
        
        self.surfaces: List[GlobalSurfaceDefinition] = surfaces
        self.wavelength: float = float(wavelength)
        self.entrance_plane: PlaneDef = entrance_plane
        self.exit_plane: Optional[PlaneDef] = exit_plane
        
        # 初始化其他属性（将由其他方法设置）
        self.optic = None  # 将由 _create_optic_global 方法创建
        self.input_rays: Optional[RealRays] = None   # 将由 trace 方法设置
        self.output_rays: Optional[RealRays] = None  # 将由 trace 方法设置
        self._chief_ray_traced: bool = False  # 标记主光线是否已追迹
        self._chief_ray_data: Optional[Dict[str, Any]] = None  # 主光线追迹数据

    
    # =========================================================================
    # 公共方法（方法签名，具体实现待后续任务完成）
    # =========================================================================
    
    def trace_chief_ray(self) -> Tuple[float, float, float]:
        """追迹主光线，计算出射方向和交点位置
        
        此方法应在混合光线追迹之前单独调用，用于：
        1. 计算出射主光线方向
        2. 存储主光线追迹数据（位置、方向等）
        3. 确定入射面和出射面的原点
        
        返回:
            出射主光线方向 (L, M, N)，全局坐标系，归一化
        
        说明:
            - 主光线从入射面中心出发，沿入射面法向量方向传播
            - 追迹结果会被存储，后续调用 trace() 时使用
            - 如果已提供 exit_plane，则直接使用
        
        Validates:
            - Requirements 1.4: 使用入射主光线方向作为入射面法向量
            - Requirements 1.5: 使用主光线与表面的交点作为入射面原点
            - Requirements 2.1: 根据出射主光线方向定义出射面法向量
            - Requirements 2.2: 使用主光线与表面的交点作为出射面原点
        """
        # 如果已经追迹过，直接返回存储的结果
        if self._chief_ray_traced:
            if self.exit_plane is None:
                raise RuntimeError(
                    "主光线已追迹但出射面未定义。这是一个内部错误。"
                )
            return self.exit_plane.normal
        
        # 如果 optic 尚未创建，先创建
        if self.optic is None:
            self._create_optic_global()
        
        # =====================================================================
        # 创建主光线：从入射面中心出发，沿入射面法向量方向传播
        # =====================================================================
        
        # 入射面中心位置（全局坐标系）
        entrance_x = float(self.entrance_plane.position[0])
        entrance_y = float(self.entrance_plane.position[1])
        entrance_z = float(self.entrance_plane.position[2])
        
        # 入射面法向量（即主光线方向）
        chief_L = float(self.entrance_plane.normal[0])
        chief_M = float(self.entrance_plane.normal[1])
        chief_N = float(self.entrance_plane.normal[2])
        
        # 创建单条主光线
        chief_ray = RealRays(
            x=np.array([entrance_x]),
            y=np.array([entrance_y]),
            z=np.array([entrance_z]),
            L=np.array([chief_L]),
            M=np.array([chief_M]),
            N=np.array([chief_N]),
            intensity=np.array([1.0]),
            wavelength=np.array([self.wavelength]),
        )
        
        # =====================================================================
        # 使用 optiland 追迹主光线
        # =====================================================================
        
        # 逐个表面追迹光线（optiland 会原地修改光线数据）
        surfaces = self.optic.surface_group.surfaces
        for i, surface in enumerate(surfaces):
            if i < 1:  # 跳过物面
                continue
            surface.trace(chief_ray)
        
        # 获取追迹后的光线数据
        x_out = float(np.asarray(chief_ray.x)[0])
        y_out = float(np.asarray(chief_ray.y)[0])
        z_out = float(np.asarray(chief_ray.z)[0])
        L_out = float(np.asarray(chief_ray.L)[0])
        M_out = float(np.asarray(chief_ray.M)[0])
        N_out = float(np.asarray(chief_ray.N)[0])
        
        # 检查追迹结果是否有效
        if not np.isfinite(x_out) or not np.isfinite(y_out) or not np.isfinite(z_out):
            raise RuntimeError(
                f"主光线追迹失败：输出位置包含无效值 "
                f"(x={x_out}, y={y_out}, z={z_out})"
            )
        
        if not np.isfinite(L_out) or not np.isfinite(M_out) or not np.isfinite(N_out):
            raise RuntimeError(
                f"主光线追迹失败：输出方向包含无效值 "
                f"(L={L_out}, M={M_out}, N={N_out})"
            )
        
        # =====================================================================
        # 计算出射主光线方向（归一化）
        # =====================================================================
        
        exit_dir = np.array([L_out, M_out, N_out], dtype=np.float64)
        exit_dir_norm = np.linalg.norm(exit_dir)
        
        if exit_dir_norm < 1e-10:
            raise RuntimeError(
                f"主光线追迹失败：出射方向向量长度为零 "
                f"(L={L_out}, M={M_out}, N={N_out})"
            )
        
        # 归一化
        exit_dir = exit_dir / exit_dir_norm
        exit_L = float(exit_dir[0])
        exit_M = float(exit_dir[1])
        exit_N = float(exit_dir[2])
        
        # =====================================================================
        # 计算主光线与最后一个光学表面的交点
        # =====================================================================
        
        # 对于抛物面，需要特殊处理交点计算
        first_surface = self.surfaces[0]
        is_parabola = np.isclose(first_surface.conic, -1.0, atol=1e-6)
        is_mirror = first_surface.surface_type == 'mirror'
        
        if is_parabola and is_mirror:
            # 对于抛物面反射镜，使用解析计算交点
            intersection_point = self._compute_parabola_intersection()
        else:
            # 对于其他表面，使用 optiland 追迹结果
            # 注意：optiland 追迹后的位置是在像面上，不是在光学表面上
            # 需要从像面位置反推到光学表面的交点
            intersection_point = self._compute_surface_intersection(
                entrance_x, entrance_y, entrance_z,
                chief_L, chief_M, chief_N
            )
        
        # =====================================================================
        # 如果 exit_plane 未提供，根据出射主光线方向和交点位置创建
        # =====================================================================
        
        if self.exit_plane is None:
            # 出射面原点为主光线与表面的交点
            # 出射面法向量为出射主光线方向
            self.exit_plane = PlaneDef(
                position=(
                    float(intersection_point[0]),
                    float(intersection_point[1]),
                    float(intersection_point[2])
                ),
                normal=(exit_L, exit_M, exit_N)
            )
        
        # =====================================================================
        # 存储主光线追迹数据
        # =====================================================================
        
        self._chief_ray_data = {
            # 入射面信息
            'entrance_position': (entrance_x, entrance_y, entrance_z),
            'entrance_direction': (chief_L, chief_M, chief_N),
            # 表面交点信息
            'surface_intersection': (
                float(intersection_point[0]),
                float(intersection_point[1]),
                float(intersection_point[2])
            ),
            # 出射面信息
            'exit_position': self.exit_plane.position,
            'exit_direction': (exit_L, exit_M, exit_N),
            # 追迹后的像面位置（用于调试）
            'image_position': (x_out, y_out, z_out),
        }
        
        # 标记主光线已追迹
        self._chief_ray_traced = True
        
        return (exit_L, exit_M, exit_N)
    
    def _compute_parabola_intersection(self) -> np.ndarray:
        """计算主光线与抛物面的交点（全局坐标系）
        
        使用解析方法计算，因为 optiland 对离轴抛物面的交点计算可能不准确。
        
        返回:
            交点位置 (x, y, z)，全局坐标系，numpy 数组
        """
        first_surface = self.surfaces[0]
        radius = first_surface.radius
        
        # 抛物面顶点位置（全局坐标系）
        vertex = np.array(first_surface.vertex_position, dtype=np.float64)
        
        # 主光线起点和方向（全局坐标系）
        ray_origin = np.array(self.entrance_plane.position, dtype=np.float64)
        ray_dir = np.array(self.entrance_plane.normal, dtype=np.float64)
        
        # 抛物面方程（以顶点为原点，光轴沿 +Z）：
        # z - vertex[2] = ((x - vertex[0])² + (y - vertex[1])²) / (2R)
        #
        # 主光线参数方程：
        # P(t) = ray_origin + t * ray_dir
        #
        # 代入抛物面方程求解 t
        
        # 简化：假设主光线沿 +Z 方向（即 ray_dir = (0, 0, 1)）
        # 这是最常见的情况
        if np.isclose(ray_dir[2], 1.0, atol=1e-6) and \
           np.isclose(ray_dir[0], 0.0, atol=1e-6) and \
           np.isclose(ray_dir[1], 0.0, atol=1e-6):
            # 主光线沿 +Z 方向
            # 交点的 x, y 坐标与起点相同
            x_int = ray_origin[0]
            y_int = ray_origin[1]
            
            # 计算 z 坐标
            # z - vertex[2] = ((x - vertex[0])² + (y - vertex[1])²) / (2R)
            dx = x_int - vertex[0]
            dy = y_int - vertex[1]
            z_int = vertex[2] + (dx**2 + dy**2) / (2 * radius)
            
            return np.array([x_int, y_int, z_int])
        else:
            # 一般情况：需要求解二次方程
            # 这里使用数值方法
            return self._compute_parabola_intersection_general(
                ray_origin, ray_dir, vertex, radius
            )
    
    def _compute_parabola_intersection_general(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        vertex: np.ndarray,
        radius: float
    ) -> np.ndarray:
        """计算主光线与抛物面的交点（一般情况）
        
        使用二次方程求解。
        
        参数:
            ray_origin: 光线起点（全局坐标系）
            ray_dir: 光线方向（归一化）
            vertex: 抛物面顶点位置（全局坐标系）
            radius: 曲率半径
        
        返回:
            交点位置 (x, y, z)，全局坐标系，numpy 数组
        """
        # 抛物面方程（以顶点为原点）：
        # z = (x² + y²) / (2R)
        #
        # 光线参数方程：
        # x = ox + t * dx
        # y = oy + t * dy
        # z = oz + t * dz
        #
        # 其中 (ox, oy, oz) = ray_origin - vertex
        #      (dx, dy, dz) = ray_dir
        #
        # 代入抛物面方程：
        # oz + t * dz = ((ox + t * dx)² + (oy + t * dy)²) / (2R)
        #
        # 展开：
        # 2R * (oz + t * dz) = (ox + t * dx)² + (oy + t * dy)²
        # 2R * oz + 2R * t * dz = ox² + 2*ox*t*dx + t²*dx² + oy² + 2*oy*t*dy + t²*dy²
        #
        # 整理为 at² + bt + c = 0：
        # a = dx² + dy²
        # b = 2*ox*dx + 2*oy*dy - 2R*dz
        # c = ox² + oy² - 2R*oz
        
        o = ray_origin - vertex
        d = ray_dir
        
        a = d[0]**2 + d[1]**2
        b = 2 * o[0] * d[0] + 2 * o[1] * d[1] - 2 * radius * d[2]
        c = o[0]**2 + o[1]**2 - 2 * radius * o[2]
        
        # 求解二次方程
        if np.abs(a) < 1e-10:
            # 线性情况（光线平行于抛物面轴）
            if np.abs(b) < 1e-10:
                raise RuntimeError("主光线与抛物面无交点")
            t = -c / b
        else:
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                raise RuntimeError("主光线与抛物面无交点")
            
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)
            
            # 选择正向的、较近的交点
            if t1 > 0 and t2 > 0:
                t = min(t1, t2)
            elif t1 > 0:
                t = t1
            elif t2 > 0:
                t = t2
            else:
                raise RuntimeError("主光线与抛物面无正向交点")
        
        # 计算交点
        intersection = ray_origin + t * ray_dir
        return intersection
    
    def _compute_surface_intersection(
        self,
        entrance_x: float,
        entrance_y: float,
        entrance_z: float,
        chief_L: float,
        chief_M: float,
        chief_N: float
    ) -> np.ndarray:
        """计算主光线与光学表面的交点（非抛物面情况）
        
        对于平面镜和球面镜，使用解析方法计算。
        
        参数:
            entrance_x, entrance_y, entrance_z: 入射面中心位置
            chief_L, chief_M, chief_N: 主光线方向
        
        返回:
            交点位置 (x, y, z)，全局坐标系，numpy 数组
        """
        first_surface = self.surfaces[0]
        radius = first_surface.radius
        vertex = np.array(first_surface.vertex_position, dtype=np.float64)
        
        ray_origin = np.array([entrance_x, entrance_y, entrance_z])
        ray_dir = np.array([chief_L, chief_M, chief_N])
        
        if np.isinf(radius):
            # 平面镜：计算光线与平面的交点
            # 平面方程：(P - vertex) · surface_normal = 0
            # 假设平面法向量为 (0, 0, -1)（面向 -Z 方向）
            surface_normal = np.array(first_surface.surface_normal, dtype=np.float64)
            
            # 光线与平面的交点
            # t = ((vertex - ray_origin) · surface_normal) / (ray_dir · surface_normal)
            denom = np.dot(ray_dir, surface_normal)
            if np.abs(denom) < 1e-10:
                # 光线与平面平行
                raise RuntimeError("主光线与平面镜平行，无交点")
            
            t = np.dot(vertex - ray_origin, surface_normal) / denom
            intersection = ray_origin + t * ray_dir
            return intersection
        else:
            # 球面镜：计算光线与球面的交点
            # 球面方程：|P - center|² = R²
            # 其中 center = vertex + R * (0, 0, 1)（曲率中心在 +Z 方向）
            center = vertex + np.array([0, 0, radius])
            
            # 光线与球面的交点
            # |ray_origin + t * ray_dir - center|² = R²
            oc = ray_origin - center
            a = np.dot(ray_dir, ray_dir)
            b = 2 * np.dot(oc, ray_dir)
            c = np.dot(oc, oc) - radius**2
            
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                raise RuntimeError("主光线与球面镜无交点")
            
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)
            
            # 选择正向的、较近的交点
            if t1 > 0 and t2 > 0:
                t = min(t1, t2)
            elif t1 > 0:
                t = t1
            elif t2 > 0:
                t = t2
            else:
                raise RuntimeError("主光线与球面镜无正向交点")
            
            intersection = ray_origin + t * ray_dir
            return intersection
    
    def trace(self, input_rays: RealRays) -> RealRays:
        """执行光线追迹
        
        将输入光线通过光学系统进行追迹，输出出射光线数据。
        
        参数:
            input_rays: 输入光线（在全局坐标系中）
                       必须是 RealRays 对象
        
        返回:
            出射光线数据（在全局坐标系中），RealRays 对象
        
        处理流程:
            1. 验证输入光线有效性（方向余弦归一化）
            2. 处理空输入情况
            3. 调用 optiland 进行光线追迹
            4. 计算带符号 OPD
            5. 应用抛物面反射修正（如果需要）
            6. 投影到出射面并计算 OPD 增量
        
        异常:
            TypeError: 如果输入参数类型错误
            ValueError: 如果光线方向余弦未归一化
        
        Validates:
            - Requirements 3.1: 直接使用光线的全局坐标进行追迹
            - Requirements 3.3: 输出光线在全局坐标系中的位置和方向
            - Requirements 3.4: 支持反射镜
            - Requirements 3.5: 支持折射面
            - Requirements 3.6: 正确计算每条光线的累积 OPD
        """
        # =====================================================================
        # 确保主光线已追迹
        # =====================================================================
        
        if not self._chief_ray_traced:
            self.trace_chief_ray()
        
        # =====================================================================
        # 验证输入参数类型
        # =====================================================================
        
        if not isinstance(input_rays, RealRays):
            raise TypeError(
                f"input_rays 参数类型错误：期望 RealRays，"
                f"实际为 {type(input_rays).__name__}"
            )
        
        # =====================================================================
        # 获取光线数量
        # =====================================================================
        
        x_array = np.asarray(input_rays.x)
        n_rays = len(x_array)
        
        # =====================================================================
        # 保存输入光线
        # =====================================================================
        
        self.input_rays = RealRays(
            x=np.asarray(input_rays.x).copy(),
            y=np.asarray(input_rays.y).copy(),
            z=np.asarray(input_rays.z).copy(),
            L=np.asarray(input_rays.L).copy(),
            M=np.asarray(input_rays.M).copy(),
            N=np.asarray(input_rays.N).copy(),
            intensity=np.asarray(input_rays.i).copy(),
            wavelength=np.asarray(input_rays.w).copy(),
        )
        self.input_rays.opd = np.asarray(input_rays.opd).copy()
        
        # =====================================================================
        # 处理空输入情况
        # =====================================================================
        
        if n_rays == 0:
            empty_rays = RealRays(
                x=np.array([]),
                y=np.array([]),
                z=np.array([]),
                L=np.array([]),
                M=np.array([]),
                N=np.array([]),
                intensity=np.array([]),
                wavelength=np.array([]),
            )
            self.output_rays = empty_rays
            return empty_rays
        
        # =====================================================================
        # 验证方向余弦归一化
        # =====================================================================
        
        L_array = np.asarray(input_rays.L)
        M_array = np.asarray(input_rays.M)
        N_array = np.asarray(input_rays.N)
        
        norm_squared = L_array**2 + M_array**2 + N_array**2
        
        if not np.allclose(norm_squared, 1.0, rtol=1e-6):
            bad_indices = np.where(~np.isclose(norm_squared, 1.0, rtol=1e-6))[0]
            first_bad_idx = bad_indices[0]
            bad_value = norm_squared[first_bad_idx]
            raise ValueError(
                f"光线方向余弦未归一化：光线 {first_bad_idx} 的 "
                f"L² + M² + N² = {bad_value:.6f}，期望为 1.0"
            )
        
        # =====================================================================
        # 复制光线对象，避免修改原始数据
        # =====================================================================
        
        traced_rays = RealRays(
            x=np.asarray(input_rays.x).copy(),
            y=np.asarray(input_rays.y).copy(),
            z=np.asarray(input_rays.z).copy(),
            L=np.asarray(input_rays.L).copy(),
            M=np.asarray(input_rays.M).copy(),
            N=np.asarray(input_rays.N).copy(),
            intensity=np.asarray(input_rays.i).copy(),
            wavelength=np.asarray(input_rays.w).copy(),
        )
        traced_rays.opd = np.asarray(input_rays.opd).copy()
        
        # =====================================================================
        # 调用带符号 OPD 的光线追迹
        # =====================================================================
        
        self._trace_with_signed_opd(traced_rays, skip=1)
        
        # =====================================================================
        # 应用抛物面反射修正（如果需要）
        # =====================================================================
        
        first_surface = self.surfaces[0]
        is_parabola = np.isclose(first_surface.conic, -1.0, atol=1e-6)
        is_mirror = first_surface.surface_type == 'mirror'
        
        if is_parabola and is_mirror:
            self._apply_parabola_correction(traced_rays)
        
        # =====================================================================
        # 投影到出射面并计算 OPD 增量
        # =====================================================================
        
        self._project_to_exit_plane(traced_rays)
        
        # =====================================================================
        # 存储输出光线
        # =====================================================================
        
        self.output_rays = traced_rays
        
        return traced_rays

    
    def get_exit_chief_ray_direction(self) -> Tuple[float, float, float]:
        """获取出射主光线方向
        
        如果主光线尚未追迹，会自动调用 trace_chief_ray()。
        
        返回:
            出射主光线方向 (L, M, N)，全局坐标系，归一化
        """
        if not self._chief_ray_traced:
            self.trace_chief_ray()
        
        if self.exit_plane is None:
            raise RuntimeError(
                "出射面尚未定义。请先调用 trace_chief_ray() 方法。"
            )
        
        return self.exit_plane.normal
    
    # =========================================================================
    # 私有方法（方法签名，具体实现待后续任务完成）
    # =========================================================================
    
    def _create_optic_global(self) -> None:
        """使用 optiland 原生的绝对坐标 API 创建光学系统
        
        配置反射镜（material='mirror'）和折射面。
        使用表面的绝对位置 (x, y, z) 和旋转角度 (rx, ry, rz) 定义表面。
        
        创建的光学系统结构：
        - index=0: 物面（无穷远）
        - index=1: 第一个光学表面（设为光阑）
        - index=2, 3, ...: 后续光学表面
        
        说明:
            - 物面设置在无穷远处（z=-np.inf）
            - 第一个光学表面设置为光阑（is_stop=True）
            - 反射镜使用 material='mirror'
            - 折射面使用指定的材料名称
            - 使用 optiland 的绝对坐标定位能力（x=, y=, z=, rx=, ry=, rz=）
            - 出射面将在后续方法中处理
        
        Validates:
            - Requirements 3.2: 使用表面的绝对位置和旋转角度定义表面
            - Requirements 4.1: 接受顶点位置作为绝对坐标
            - Requirements 4.2: 接受旋转角度定义表面朝向
            - Requirements 4.3: 使用 optiland 的全局坐标能力设置表面
            - Requirements 4.4: 正确处理离轴系统的表面定义
        """
        from optiland.optic import Optic
        
        # 创建光学系统
        optic = Optic()
        
        # =====================================================================
        # 设置系统参数
        # =====================================================================
        
        # 设置孔径（使用默认值，因为 GlobalSurfaceDefinition 不包含 semi_aperture）
        aperture_diameter = 10.0
        optic.set_aperture(aperture_type='EPD', value=aperture_diameter)
        
        # 设置视场类型为角度，轴上视场
        optic.set_field_type(field_type='angle')
        optic.add_field(y=0, x=0)
        
        # 设置波长
        optic.add_wavelength(value=self.wavelength, is_primary=True)
        
        # =====================================================================
        # 添加物面（index=0）
        # =====================================================================
        # 物面设置在无穷远处，使用 z=-np.inf
        optic.add_surface(index=0, radius=np.inf, z=-np.inf)
        
        # =====================================================================
        # 添加入射面（index=1）
        # =====================================================================
        # 入射面是一个透明平面，位于入射面定义的位置
        # 这样可以确保光线从正确的位置开始追迹
        entrance_x = float(self.entrance_plane.position[0])
        entrance_y = float(self.entrance_plane.position[1])
        entrance_z = float(self.entrance_plane.position[2])
        
        optic.add_surface(
            index=1,
            radius=np.inf,
            x=entrance_x,
            y=entrance_y,
            z=entrance_z,
        )
        
        # =====================================================================
        # 添加光学表面
        # =====================================================================
        
        for i, surface_def in enumerate(self.surfaces):
            # 表面索引从 2 开始（0=物面，1=入射面）
            surface_index = i + 2
            is_stop = (i == 0)  # 第一个光学表面设为光阑
            
            # 确定材料
            if surface_def.surface_type == 'mirror':
                material = 'mirror'
            else:
                material = surface_def.material
            
            # 获取曲率半径
            radius = surface_def.radius
            
            # 获取顶点位置（绝对坐标）
            vertex_x = float(surface_def.vertex_position[0])
            vertex_y = float(surface_def.vertex_position[1])
            vertex_z = float(surface_def.vertex_position[2])
            
            # 获取旋转角度
            # 注意：对于抛物面，需要特殊处理
            is_parabola = np.isclose(surface_def.conic, -1.0, atol=1e-6)
            
            if is_parabola:
                # 抛物面：离轴效果通过顶点位置实现，不设置额外的倾斜角度
                # 抛物面的反射方向由其几何形状自然决定
                tilt_x = 0.0
                tilt_y = 0.0
                tilt_z = 0.0
                
                # 如果原本有非零倾斜角，发出警告
                if (abs(surface_def.tilt_x) > 1e-6 or 
                    abs(surface_def.tilt_y) > 1e-6 or 
                    abs(surface_def.tilt_z) > 1e-6):
                    print(
                        f"警告: 检测到抛物面 (Surface {surface_index}) 具有非零倾斜角 "
                        f"(tilt_x={np.degrees(surface_def.tilt_x):.2f}°, "
                        f"tilt_y={np.degrees(surface_def.tilt_y):.2f}°, "
                        f"tilt_z={np.degrees(surface_def.tilt_z):.2f}°)。"
                        f"对于 OAP，倾斜通常通过顶点偏移定义。"
                        f"optiland 中的倾斜参数将被强制置为 0。"
                    )
            else:
                # 其他表面：使用 GlobalSurfaceDefinition 中的倾斜角度
                tilt_x = self._avoid_exact_45_degrees(surface_def.tilt_x)
                tilt_y = self._avoid_exact_45_degrees(surface_def.tilt_y)
                tilt_z = surface_def.tilt_z
            
            # 使用 optiland 的绝对坐标 API 添加表面
            # 参考：optiland-master/docs/gallery/reflective/laser_system.ipynb
            optic.add_surface(
                index=surface_index,
                radius=radius,
                conic=surface_def.conic,
                material=material,
                is_stop=is_stop,
                # 绝对坐标定位
                x=vertex_x,
                y=vertex_y,
                z=vertex_z,
                # 旋转角度（弧度）
                rx=tilt_x,
                ry=tilt_y,
                rz=tilt_z,
            )
        
        # =====================================================================
        # 添加像面（index=最后）
        # =====================================================================
        # 像面设置在最后一个光学表面之后一定距离
        # 对于反射镜，像面应该在反射方向上
        # 这里暂时设置一个较大的距离，确保光线能够正确追迹
        # 实际的出射面位置将在 trace_chief_ray() 后根据主光线交点确定
        last_surface = self.surfaces[-1]
        image_index = len(self.surfaces) + 2
        
        # 计算像面位置：沿入射方向延伸一定距离
        # 对于反射镜，这个位置会在追迹后被忽略，因为我们使用解析方法计算交点
        entrance_normal = np.array(self.entrance_plane.normal, dtype=np.float64)
        entrance_pos = np.array(self.entrance_plane.position, dtype=np.float64)
        
        # 像面位置：入射面位置 + 入射方向 × 较大距离
        # 这确保像面在光学表面之后
        image_distance = 1000.0  # mm，足够大的距离
        image_pos = entrance_pos + entrance_normal * image_distance
        
        optic.add_surface(
            index=image_index,
            radius=np.inf,
            x=float(image_pos[0]),
            y=float(image_pos[1]),
            z=float(image_pos[2]),
        )
        
        # 存储创建的光学系统
        self.optic = optic
    
    @staticmethod
    def _avoid_exact_45_degrees(angle: float) -> float:
        """避免精确的 45° 角度
        
        optiland 在精确的 45° (π/4) 角度时存在数值问题，会导致光线追迹返回 NaN。
        此函数通过添加极小的偏移量来避免这个问题。
        
        参数:
            angle: 输入角度（弧度）
        
        返回:
            调整后的角度（弧度），如果接近 45° 的整数倍则添加小偏移量
        
        说明:
            - 检查角度是否接近 ±45°, ±135° 等（即 π/4 的奇数倍）
            - 如果是，添加 1e-10 弧度的偏移量
            - 这个偏移量足够小，不会影响光学计算精度
        """
        # 检查是否接近 45° 的奇数倍
        # 45° = π/4, 135° = 3π/4, 225° = 5π/4, 315° = 7π/4
        pi_over_4 = np.pi / 4
        
        # 计算角度除以 π/4 的商
        quotient = angle / pi_over_4
        
        # 检查是否接近奇数
        nearest_odd = round(quotient)
        if nearest_odd % 2 == 1:  # 奇数
            # 检查是否足够接近
            if abs(quotient - nearest_odd) < 1e-10:
                # 添加小偏移量
                return angle + 1e-10
        
        return angle
    
    def _trace_with_signed_opd(self, rays: RealRays, skip: int = 0) -> None:
        """使用带符号 OPD 进行光线追迹
        
        复用 ElementRaytracer 的带符号 OPD 计算逻辑。
        
        与 optiland 标准追迹的区别：
        - optiland 使用 abs(t) 计算 OPD
        - 本方法使用带符号的 t 计算 OPD
        - 对于抛物面，修正反射方向
        
        为什么需要带符号的 OPD：
        入射面和出射面与镜面是交错的：
        - 入射面上的某些光线位于镜面的"后方"，找交点时 t < 0，
          然后反射后经过 t > 0 到达出射面
        - 另一部分光线则相反：先 t > 0 到达镜面，反射后 t < 0 到达出射面
        
        正负 t 值会在整个光程中相互抵消，只有使用带符号的 OPD 才能
        使残差 OPD 正确趋近于零。
        
        符号计算公式：
        - t 的符号 = sign(dz) * sign(N_before)
        - 其中 N_before 是追迹前光线的 Z 方向分量
        
        参数:
            rays: 光线对象（将被原地修改）
            skip: 跳过的表面数量
        
        Validates:
            - Requirements 3.6: 正确计算每条光线的累积 OPD
            - Requirements 7.1: 使用主光线作为参考（主光线 OPD = 0）
        """
        # 获取表面列表
        surfaces = self.optic.surface_group.surfaces
        
        # 初始化 OPD（如果尚未初始化）
        if rays.opd is None:
            rays.opd = np.zeros(len(rays.x))
        
        # 逐个表面追迹
        for i, surface in enumerate(surfaces):
            if i < skip:
                continue
            
            # 保存追迹前的 OPD、z 坐标、方向
            opd_before = np.asarray(rays.opd).copy()
            z_before = np.asarray(rays.z).copy()
            N_before = np.asarray(rays.N).copy()
            L_before = np.asarray(rays.L).copy()
            M_before = np.asarray(rays.M).copy()
            
            # 使用 optiland 的表面追迹（会使用 abs(t)）
            surface.trace(rays)
            
            # 获取追迹后的 z 坐标和 OPD
            z_after = np.asarray(rays.z)
            opd_after = np.asarray(rays.opd)
            
            # 计算 OPD 增量（optiland 使用 abs(t)）
            opd_increment_abs = opd_after - opd_before
            
            # 计算 dz 来确定 t 的符号
            dz = z_after - z_before
            
            # 正确的符号计算：sign(t) = sign(dz) * sign(N_before)
            sign_dz = np.sign(dz)
            sign_N = np.sign(N_before)
            
            # 处理零值情况
            sign_dz[sign_dz == 0] = 1
            sign_N[sign_N == 0] = 1
            
            # t 的符号
            sign_t = sign_dz * sign_N
            
            # 计算带符号的 OPD 增量
            opd_increment_signed = sign_t * opd_increment_abs
            
            # 更新 OPD
            rays.opd = opd_before + opd_increment_signed

    
    def _apply_parabola_correction(self, rays: RealRays) -> None:
        """应用抛物面反射修正
        
        复用 ElementRaytracer 的抛物面反射修正逻辑。
        对于抛物面反射镜，反射方向直接指向焦点。
        
        optiland 使用几何法向量计算反射，但对于离轴抛物面这是不正确的。
        正确的反射方向应该使用抛物面的光学性质：
        - 对于平行于光轴的入射光线，反射后通过焦点
        - 反射方向 r = (F - P) / |F - P|
        
        参数:
            rays: 光线对象（将被原地修改）
        
        Validates:
            - Requirements 4.5: 通过顶点位置自然实现离轴效果
        """
        # 获取光线位置（在全局坐标系中，已经在表面上）
        x = np.asarray(rays.x)
        y = np.asarray(rays.y)
        z = np.asarray(rays.z)
        
        # 获取抛物面参数
        first_surface = self.surfaces[0]
        radius = first_surface.radius
        focal_length = radius / 2.0
        
        # 抛物面顶点位置（全局坐标系）
        vertex = np.array(first_surface.vertex_position, dtype=np.float64)
        
        # 焦点位置（全局坐标系）：顶点 + (0, 0, f)
        focus = vertex + np.array([0.0, 0.0, focal_length])
        
        # 计算每条光线的正确反射方向
        n_rays = len(x)
        L_new = np.zeros(n_rays)
        M_new = np.zeros(n_rays)
        N_new = np.zeros(n_rays)
        
        for j in range(n_rays):
            # 光线位置（全局坐标系）
            P = np.array([x[j], y[j], z[j]])
            
            # 从交点到焦点的向量
            P_to_F = focus - P
            dist_to_focus = np.linalg.norm(P_to_F)
            
            if dist_to_focus < 1e-10:
                # 光线恰好在焦点，保持沿 +Z 方向
                L_new[j] = 0.0
                M_new[j] = 0.0
                N_new[j] = 1.0
                continue
            
            # 反射方向直接指向焦点
            r = P_to_F / dist_to_focus
            
            L_new[j] = r[0]
            M_new[j] = r[1]
            N_new[j] = r[2]
        
        # 更新光线方向
        rays.L = L_new
        rays.M = M_new
        rays.N = N_new
    
    def _project_to_exit_plane(self, rays: RealRays) -> None:
        """将光线投影到出射面并计算 OPD 增量
        
        计算每条光线与出射面的交点位置，并更新 OPD。
        
        出射面方程（在全局坐标系中）：
        (P - P0) · n = 0
        其中 P0 = 出射面原点，n = 出射面法向量
        
        光线方程：P = P_ray + t * D_ray
        代入得：t = (P0 - P_ray) · n / (D_ray · n)
        
        参数:
            rays: 光线对象（将被原地修改）
        
        Validates:
            - Requirements 2.3: 计算每条光线与出射面的交点位置
            - Requirements 2.4: 计算光线从表面到出射面的 OPD 增量
            - Requirements 2.5: 正确处理光线方向与出射面法向量平行的边界情况
        """
        if self.exit_plane is None:
            raise RuntimeError(
                "出射面尚未定义。请先调用 trace_chief_ray() 方法。"
            )
        
        # 获取光线数据
        x_ray = np.asarray(rays.x)
        y_ray = np.asarray(rays.y)
        z_ray = np.asarray(rays.z)
        L_ray = np.asarray(rays.L)
        M_ray = np.asarray(rays.M)
        N_ray = np.asarray(rays.N)
        opd_ray = np.asarray(rays.opd)
        
        # 出射面原点（全局坐标系）
        P0 = np.array(self.exit_plane.position, dtype=np.float64)
        
        # 出射面法向量（全局坐标系）
        n = np.array(self.exit_plane.normal, dtype=np.float64)
        
        # 计算每条光线到出射面的距离 t
        # t = (P0 - P_ray) · n / (D_ray · n)
        P_ray = np.stack([x_ray, y_ray, z_ray], axis=0)  # (3, n_rays)
        D_ray = np.stack([L_ray, M_ray, N_ray], axis=0)  # (3, n_rays)
        
        # (P0 - P_ray) · n
        diff = P0.reshape(3, 1) - P_ray  # (3, n_rays)
        numerator = np.sum(diff * n.reshape(3, 1), axis=0)  # (n_rays,)
        
        # D_ray · n
        denominator = np.sum(D_ray * n.reshape(3, 1), axis=0)  # (n_rays,)
        
        # 处理光线与出射面平行的情况
        parallel_mask = np.abs(denominator) < 1e-10
        denominator = np.where(parallel_mask, 1e-10, denominator)
        
        # 计算 t
        t = numerator / denominator
        
        # 对于平行光线，设置 t = 0（保持原位置）
        t = np.where(parallel_mask, 0.0, t)
        
        # 更新光线位置到出射面
        x_exit = x_ray + t * L_ray
        y_exit = y_ray + t * M_ray
        z_exit = z_ray + t * N_ray
        
        # 更新 OPD（带符号）
        # OPD 增量 = t（因为在空气中 n=1）
        opd_exit = opd_ray + t
        
        # 更新 rays
        rays.x = x_exit
        rays.y = y_exit
        rays.z = z_exit
        rays.opd = opd_exit
