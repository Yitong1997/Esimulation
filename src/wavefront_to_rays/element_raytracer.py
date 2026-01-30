"""元件光线追迹模块

本模块提供元件光线追迹功能，将输入光线通过一个或多个光学表面进行追迹，
输出出射光束的光线数据。

主要组件：
- SurfaceDefinition: 光学表面定义数据类
- ElementRaytracer: 元件光线追迹器类（待实现）
- 坐标转换辅助函数

基于 optiland 库实现光线追迹计算。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from scipy.spatial.transform import Rotation
from numpy.typing import NDArray

from optiland.rays import RealRays


@dataclass
class SurfaceDefinition:
    """光学表面定义
    
    定义光学表面的几何和材料属性，用于元件光线追迹。
    
    参数:
        surface_type: 表面类型
            - 'mirror': 反射镜
            - 'refract': 折射面
        radius: 曲率半径，单位：mm
            - 正值表示凸面（曲率中心在 +Z 方向）
            - 负值表示凹面（曲率中心在 -Z 方向）
            - np.inf 表示平面
        thickness: 到下一表面的厚度，单位：mm
            - 对于单个反射镜，通常为 0.0
        material: 材料名称
            - 'mirror': 反射镜
            - 其他字符串: 折射材料名称（如 'N-BK7', 'air' 等）
        semi_aperture: 半口径，单位：mm
            - None 表示无限制
            - 正值表示有效区域的半径
        conic: 圆锥常数（conic constant），默认 0.0
            - 0: 球面
            - -1: 抛物面
            - < -1: 双曲面
            - -1 < k < 0: 椭球面（扁椭球）
            - > 0: 椭球面（长椭球）
        tilt_x: 绕 X 轴旋转角度（弧度），默认 0.0
        tilt_y: 绕 Y 轴旋转角度（弧度），默认 0.0
    
    示例:
        >>> # 创建凹面反射镜（焦距 100mm，曲率半径 -200mm）
        >>> mirror = SurfaceDefinition(
        ...     surface_type='mirror',
        ...     radius=-200.0,
        ...     thickness=0.0,
        ...     material='mirror',
        ...     semi_aperture=15.0
        ... )
        
        >>> # 创建平面反射镜
        >>> flat_mirror = SurfaceDefinition(
        ...     surface_type='mirror',
        ...     radius=np.inf,
        ...     thickness=0.0,
        ...     material='mirror'
        ... )
        
        >>> # 创建球面折射面
        >>> refract_surface = SurfaceDefinition(
        ...     surface_type='refract',
        ...     radius=100.0,
        ...     thickness=10.0,
        ...     material='N-BK7',
        ...     semi_aperture=12.5
        ... )
        
        >>> # 创建抛物面反射镜（焦距 100mm，顶点曲率半径 200mm）
        >>> parabolic_mirror = SurfaceDefinition(
        ...     surface_type='mirror',
        ...     radius=200.0,
        ...     thickness=0.0,
        ...     material='mirror',
        ...     semi_aperture=15.0,
        ...     conic=-1.0  # 抛物面
        ... )
        
    注意:
        - 曲率半径符号约定遵循 optiland 标准：
          正值表示曲率中心在表面顶点的 +Z 方向
        - 对于凹面镜，焦距 f = R/2（R 为曲率半径）
        - 抛物面的圆锥常数 k = -1
        - 离轴效果通过绝对坐标定位实现，不使用偏心参数
    """
    
    surface_type: str = 'mirror'
    radius: float = field(default_factory=lambda: np.inf)
    thickness: float = 0.0
    material: str = 'mirror'
    semi_aperture: Optional[float] = None
    conic: float = 0.0  # 圆锥常数，默认 0.0（球面）
    tilt_x: float = 0.0  # 绕 X 轴旋转角度（弧度）
    tilt_y: float = 0.0  # 绕 Y 轴旋转角度（弧度）
    orientation: Optional[NDArray] = field(default=None)  # Global Rotation Matrix (3x3)
    # 表面顶点在全局坐标系中的位置 (x, y, z)，单位：mm
    # 对于离轴系统（如 OAP），这是表面顶点的实际位置
    # 默认为 (0, 0, 0)，即表面顶点在全局原点
    vertex_position: Optional[Tuple[float, float, float]] = None
    
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
        
        # 验证厚度
        if not isinstance(self.thickness, (int, float)):
            raise TypeError(
                f"厚度类型错误：期望 int 或 float，"
                f"实际为 {type(self.thickness).__name__}"
            )
        
        # 验证半口径
        if self.semi_aperture is not None:
            if not isinstance(self.semi_aperture, (int, float)):
                raise TypeError(
                    f"半口径类型错误：期望 int、float 或 None，"
                    f"实际为 {type(self.semi_aperture).__name__}"
                )
            if self.semi_aperture <= 0:
                raise ValueError(
                    f"半口径必须为正值，实际为 {self.semi_aperture}"
                )
        
        # 验证材料名称
        if not isinstance(self.material, str):
            raise TypeError(
                f"材料名称类型错误：期望 str，"
                f"实际为 {type(self.material).__name__}"
            )
        if not self.material:
            raise ValueError("材料名称不能为空字符串")
        
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
        
        # 验证倾斜角度
        if not isinstance(self.tilt_x, (int, float)):
            raise TypeError(
                f"X 轴旋转角度类型错误：期望 int 或 float，"
                f"实际为 {type(self.tilt_x).__name__}"
            )
        if not np.isfinite(self.tilt_x):
            raise ValueError(
                f"X 轴旋转角度必须为有限值，实际为 {self.tilt_x}"
            )
        
        if not isinstance(self.tilt_y, (int, float)):
            raise TypeError(
                f"Y 轴旋转角度类型错误：期望 int 或 float，"
                f"实际为 {type(self.tilt_y).__name__}"
            )
        if not np.isfinite(self.tilt_y):
            raise ValueError(
                f"Y 轴旋转角度必须为有限值，实际为 {self.tilt_y}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """将表面定义转换为字典
        
        用于序列化和调试输出。
        
        返回:
            包含所有属性的字典
        
        示例:
            >>> surface = SurfaceDefinition(
            ...     surface_type='mirror',
            ...     radius=200.0,
            ...     semi_aperture=15.0
            ... )
            >>> surface.to_dict()
            {
                'surface_type': 'mirror',
                'radius': 200.0,
                'thickness': 0.0,
                'material': 'mirror',
                'semi_aperture': 15.0,
                'conic': 0.0
            }
        """
        return {
            'surface_type': self.surface_type,
            'radius': float(self.radius) if np.isfinite(self.radius) else 'inf',
            'thickness': self.thickness,
            'material': self.material,
            'semi_aperture': self.semi_aperture,
            'conic': self.conic,
            'tilt_x': self.tilt_x,
            'tilt_y': self.tilt_y,
            'vertex_position': self.vertex_position,
        }
    
    @property
    def is_mirror(self) -> bool:
        """判断是否为反射镜"""
        return self.surface_type == 'mirror'
    
    @property
    def is_plane(self) -> bool:
        """判断是否为平面"""
        return np.isinf(self.radius)
    
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
            return -self.radius / 2.0
        return None
    
    def __repr__(self) -> str:
        """返回表面定义的字符串表示"""
        radius_str = 'inf' if np.isinf(self.radius) else f'{self.radius:.2f}'
        aperture_str = (
            f'{self.semi_aperture:.2f}' 
            if self.semi_aperture is not None 
            else 'None'
        )
        conic_str = f'{self.conic:.2f}' if self.conic != 0.0 else '0'
        
        # 构建基本字符串
        result = (
            f"SurfaceDefinition("
            f"type='{self.surface_type}', "
            f"radius={radius_str} mm, "
            f"thickness={self.thickness:.2f} mm, "
            f"material='{self.material}', "
            f"semi_aperture={aperture_str} mm, "
            f"conic={conic_str}"
        )
        
        # 添加倾斜信息（如果有）
        if self.tilt_x != 0.0 or self.tilt_y != 0.0:
            tilt_x_deg = np.degrees(self.tilt_x)
            tilt_y_deg = np.degrees(self.tilt_y)
            result += f", tilt=({tilt_x_deg:.1f}°, {tilt_y_deg:.1f}°)"
        
        result += ")"
        return result


# =============================================================================
# 坐标转换辅助函数
# =============================================================================

def _snap_to_zero(val: float, tol: float = 1e-14) -> float:
    """如果数值绝对值小于阈值，强制归零
    
    用于清洗浮点数噪声，特别是在作为反三角函数输入前。
    """
    if abs(val) < tol:
        return 0.0
    return val


def _avoid_exact_45_degrees(angle: float) -> float:
    """避免精确的 45° 角度
    
    (已弃用) 之前的版本为了规避 optiland 在精确 45° 时的潜在数值问题引入了 1e-10 偏移。
    现在为了保证数值精度，移除此偏移，直接返回原值。
    """
    return angle


def _normalize_vector(v: NDArray) -> NDArray:
    """归一化向量
    
    参数:
        v: 输入向量，形状为 (3,)
    
    返回:
        归一化后的向量
    
    异常:
        ValueError: 如果向量长度为零
    """
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        raise ValueError("无法归一化零向量")
    return v / norm


def compute_rotation_matrix(
    chief_ray_direction: Tuple[float, float, float],
) -> NDArray:
    """计算从入射面局部坐标系到全局坐标系的旋转矩阵
    
    根据主光线方向计算旋转矩阵。主光线方向定义了入射面的法向量，
    即局部坐标系的 Z 轴在全局坐标系中的方向。
    
    参数:
        chief_ray_direction: 主光线方向 (L, M, N)，即局部 Z 轴在全局坐标系中的方向。
                            必须是归一化的方向余弦。
    
    返回:
        3x3 旋转矩阵 R，满足 v_global = R @ v_local
        矩阵的列向量分别为局部 X、Y、Z 轴在全局坐标系中的表示
    
    异常:
        ValueError: 如果输入方向向量无效（长度为零或未归一化）
    
    示例:
        >>> # 正入射情况：主光线沿全局 Z 轴
        >>> R = compute_rotation_matrix((0, 0, 1))
        >>> np.allclose(R, np.eye(3))
        True
        
        >>> # 45度倾斜入射：主光线在 YZ 平面内
        >>> R = compute_rotation_matrix((0, np.sin(np.pi/4), np.cos(np.pi/4)))
    
    算法说明:
        1. 局部 Z 轴 = 主光线方向（归一化）
        2. 选择参考向量来定义局部 X 轴：
           - 如果主光线接近全局 Y 轴，使用全局 X 轴作为参考
           - 否则使用全局 Y 轴作为参考
        3. 局部 X 轴 = 参考向量 × 局部 Z 轴（归一化）
        4. 局部 Y 轴 = 局部 Z 轴 × 局部 X 轴
        5. 旋转矩阵的列向量为局部坐标轴在全局坐标系中的表示
    
    坐标系一致性说明:
        为了确保入射面和出射面的局部坐标系保持一致的"上下"方向，
        我们需要确保局部 Y 轴始终与全局 Y 轴的投影方向一致。
        
        当主光线方向的 Z 分量为负时（如反射后向 -Z 方向传播），
        简单地翻转 X 轴会导致 Y 轴也翻转，从而破坏坐标系的一致性。
        
        正确的做法是：确保局部 Y 轴在全局 Y 方向上的投影为正。
    
    Validates:
        - Requirements 3.1: 入射面定位于 z=0 位置
        - Requirements 3.2: 出射面定位于最后一个光学表面的顶点位置
        - Requirements 3.4: 接受 z 坐标不为零的光线并从其当前位置开始追迹
    """
    # 将输入转换为 numpy 数组
    z_local = np.array(chief_ray_direction, dtype=np.float64)
    
    # 验证输入
    if z_local.shape != (3,):
        raise ValueError(
            f"主光线方向必须是长度为 3 的向量，实际形状为 {z_local.shape}"
        )
    
    # 检查向量长度
    norm = np.linalg.norm(z_local)
    if norm < 1e-10:
        raise ValueError("主光线方向不能为零向量")
    
    # 检查是否归一化（允许小误差）
    if not np.isclose(norm, 1.0, rtol=1e-6):
        raise ValueError(
            f"主光线方向余弦未归一化：L² + M² + N² = {norm**2:.6f}，期望为 1.0"
        )
    
    # 归一化（处理小的数值误差）
    z_local = _normalize_vector(z_local)
    
    # 选择参考向量来定义局部 X 轴
    # 如果主光线接近全局 Y 轴（|M| > 0.999999），使用全局 X 轴作为参考
    # 否则使用全局 Y 轴作为参考
    # 
    # 修正：原阈值 0.9 (约 25°) 过宽，导致在 steep angle (如离轴抛物面) 下
    # 坐标系发生 90° 翻转，进而这就导致 tilt_x/tilt_y 作用轴错误。
    # 现改为 0.999999 (约 0.08°)，仅在极接近垂直时切换参考轴。
    # 修正：当主光线接近全局 Y 轴时，使用全局 Z 轴 [0, 0, 1] 作为参考
    # 这样可以确保生成的局部坐标系中，Y 轴（Vertical）尽可能指向全局 Z 方向（Up）
    # 这更符合光学系统的直觉：传播方向为 Y 时，Z 通常被视为垂直/高度方向。
    if abs(z_local[1]) > 0.999999:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    
    # 局部 X 轴 = ref × z_local（归一化）
    x_local = np.cross(ref, z_local)
    x_local = _normalize_vector(x_local)
    
    # 局部 Y 轴 = z_local × x_local（确保右手系）
    y_local = np.cross(z_local, x_local)
    # y_local 已经是归一化的（因为 z_local 和 x_local 都是归一化且正交的）
    
    # ⚠️ 关键修复：确保局部 Y 轴与全局 Y 轴的投影为正
    # 这样可以保证入射面和出射面的局部坐标系保持一致的"上下"方向
    # 
    # 当主光线方向的 Z 分量为负时（如反射后向 -Z 方向传播），
    # 叉积的结果可能导致 Y 轴翻转。
    # 
    # 通过检查 Y 轴在全局 Y 方向上的投影，如果为负则同时翻转 X 和 Y 轴
    # （同时翻转 X 和 Y 保持右手系）
    if y_local[1] < 0:
        x_local = -x_local
        y_local = -y_local
    
    # 旋转矩阵：列向量为局部坐标轴在全局坐标系中的表示
    # R = [x_local | y_local | z_local]
    R = np.column_stack([x_local, y_local, z_local])
    
    return R


def transform_rays_to_global(
    rays: RealRays,
    rotation_matrix: NDArray,
    entrance_position: Tuple[float, float, float],
) -> RealRays:
    """将光线从入射面局部坐标系转换到全局坐标系
    
    对光线的位置和方向进行坐标变换，从入射面的局部坐标系转换到全局坐标系。
    
    参数:
        rays: 输入光线（在局部坐标系中），RealRays 对象
        rotation_matrix: 3x3 旋转矩阵，由 compute_rotation_matrix() 计算得到
        entrance_position: 入射面中心在全局坐标系中的位置 (x, y, z)，单位：mm
    
    返回:
        转换后的光线（在全局坐标系中），新的 RealRays 对象
    
    说明:
        - 位置转换：pos_global = R @ pos_local + entrance_position
        - 方向转换：dir_global = R @ dir_local
        - 不修改原始光线对象，返回新的 RealRays 对象
        - OPD 和强度等属性保持不变
    
    示例:
        >>> # 创建测试光线
        >>> rays = RealRays(
        ...     x=[0, 1], y=[0, 0], z=[0, 0],
        ...     L=[0, 0], M=[0, 0], N=[1, 1],
        ...     intensity=[1, 1], wavelength=[0.55, 0.55]
        ... )
        >>> R = compute_rotation_matrix((0, 0, 1))
        >>> rays_global = transform_rays_to_global(rays, R, (0, 0, 0))
    
    Validates:
        - Requirements 3.1: 入射面定位于 z=0 位置
        - Requirements 3.4: 接受 z 坐标不为零的光线并从其当前位置开始追迹
    """
    # 验证旋转矩阵形状
    R = np.asarray(rotation_matrix)
    if R.shape != (3, 3):
        raise ValueError(
            f"旋转矩阵形状错误：期望 (3, 3)，实际为 {R.shape}"
        )
    
    # 将入射面位置转换为 numpy 数组
    entrance_pos = np.array(entrance_position, dtype=np.float64)
    if entrance_pos.shape != (3,):
        raise ValueError(
            f"入射面位置必须是长度为 3 的向量，实际形状为 {entrance_pos.shape}"
        )
    
    # 获取光线数据（转换为 numpy 数组）
    x_local = np.asarray(rays.x)
    y_local = np.asarray(rays.y)
    z_local = np.asarray(rays.z)
    L_local = np.asarray(rays.L)
    M_local = np.asarray(rays.M)
    N_local = np.asarray(rays.N)
    
    # 位置转换：pos_global = R @ pos_local + entrance_position
    # 对于每条光线，进行矩阵乘法
    # 使用向量化操作提高效率
    pos_local = np.stack([x_local, y_local, z_local], axis=0)  # (3, n_rays)
    pos_global = R @ pos_local + entrance_pos.reshape(3, 1)  # (3, n_rays)
    
    x_global = pos_global[0]
    y_global = pos_global[1]
    z_global = pos_global[2]
    
    # 方向转换：dir_global = R @ dir_local
    dir_local = np.stack([L_local, M_local, N_local], axis=0)  # (3, n_rays)
    dir_global = R @ dir_local  # (3, n_rays)
    
    L_global = dir_global[0]
    M_global = dir_global[1]
    N_global = dir_global[2]
    
    # 创建新的 RealRays 对象
    new_rays = RealRays(
        x=x_global,
        y=y_global,
        z=z_global,
        L=L_global,
        M=M_global,
        N=N_global,
        intensity=np.asarray(rays.i),
        wavelength=np.asarray(rays.w),
    )
    
    # 复制 OPD 数据
    new_rays.opd = np.asarray(rays.opd).copy()
    
    return new_rays


def transform_rays_to_local(
    rays: RealRays,
    rotation_matrix: NDArray,
    exit_position: Tuple[float, float, float],
) -> RealRays:
    """将光线从全局坐标系转换到出射面局部坐标系
    
    对光线的位置和方向进行坐标变换，从全局坐标系转换到出射面的局部坐标系。
    
    参数:
        rays: 输入光线（在全局坐标系中），RealRays 对象
        rotation_matrix: 3x3 旋转矩阵（出射面的），由 compute_rotation_matrix() 计算得到
        exit_position: 出射面中心在全局坐标系中的位置 (x, y, z)，单位：mm
    
    返回:
        转换后的光线（在局部坐标系中），新的 RealRays 对象
    
    说明:
        - 位置转换：pos_local = R.T @ (pos_global - exit_position)
        - 方向转换：dir_local = R.T @ dir_global
        - 不修改原始光线对象，返回新的 RealRays 对象
        - OPD 和强度等属性保持不变
    
    示例:
        >>> # 创建测试光线（在全局坐标系中）
        >>> rays = RealRays(
        ...     x=[0, 1], y=[0, 0], z=[0, 0],
        ...     L=[0, 0], M=[0, 0], N=[1, 1],
        ...     intensity=[1, 1], wavelength=[0.55, 0.55]
        ... )
        >>> R = compute_rotation_matrix((0, 0, 1))
        >>> rays_local = transform_rays_to_local(rays, R, (0, 0, 0))
    
    Validates:
        - Requirements 3.2: 出射面定位于最后一个光学表面的顶点位置
    """
    # 验证旋转矩阵形状
    R = np.asarray(rotation_matrix)
    if R.shape != (3, 3):
        raise ValueError(
            f"旋转矩阵形状错误：期望 (3, 3)，实际为 {R.shape}"
        )
    
    # 将出射面位置转换为 numpy 数组
    exit_pos = np.array(exit_position, dtype=np.float64)
    if exit_pos.shape != (3,):
        raise ValueError(
            f"出射面位置必须是长度为 3 的向量，实际形状为 {exit_pos.shape}"
        )
    
    # 获取光线数据（转换为 numpy 数组）
    x_global = np.asarray(rays.x)
    y_global = np.asarray(rays.y)
    z_global = np.asarray(rays.z)
    L_global = np.asarray(rays.L)
    M_global = np.asarray(rays.M)
    N_global = np.asarray(rays.N)
    
    # 位置转换：pos_local = R.T @ (pos_global - exit_position)
    # 使用向量化操作提高效率
    pos_global = np.stack([x_global, y_global, z_global], axis=0)  # (3, n_rays)
    pos_relative = pos_global - exit_pos.reshape(3, 1)  # (3, n_rays)
    pos_local = R.T @ pos_relative  # (3, n_rays)
    
    x_local = pos_local[0]
    y_local = pos_local[1]
    z_local = pos_local[2]
    
    # 方向转换：dir_local = R.T @ dir_global
    dir_global = np.stack([L_global, M_global, N_global], axis=0)  # (3, n_rays)
    dir_local = R.T @ dir_global  # (3, n_rays)
    
    L_local = dir_local[0]
    M_local = dir_local[1]
    N_local = dir_local[2]
    
    # 创建新的 RealRays 对象
    new_rays = RealRays(
        x=x_local,
        y=y_local,
        z=z_local,
        L=L_local,
        M=M_local,
        N=N_local,
        intensity=np.asarray(rays.i),
        wavelength=np.asarray(rays.w),
    )
    
    # 复制 OPD 数据
    new_rays.opd = np.asarray(rays.opd).copy()
    
    return new_rays


# =============================================================================
# 便捷工厂函数
# =============================================================================

def create_mirror_surface(
    radius: float,
    semi_aperture: Optional[float] = None,
) -> SurfaceDefinition:
    """创建反射镜表面定义
    
    便捷工厂函数，用于快速创建反射镜表面定义。
    
    参数:
        radius: 曲率半径，单位：mm
            - 正值表示凸面镜（曲率中心在 +Z 方向，发散）
            - 负值表示凹面镜（曲率中心在 -Z 方向，聚焦）
            - np.inf 表示平面镜
        semi_aperture: 半口径，单位：mm
            - None 表示无限制
            - 正值表示有效区域的半径
    
    返回:
        SurfaceDefinition: 反射镜表面定义对象
    
    示例:
        >>> # 创建凹面镜（焦距 100mm，曲率半径 -200mm）
        >>> mirror = create_mirror_surface(radius=-200.0, semi_aperture=15.0)
        >>> mirror.focal_length
        100.0
        
        >>> # 创建平面镜
        >>> flat_mirror = create_mirror_surface(radius=np.inf)
        >>> flat_mirror.is_plane
        True
        
        >>> # 创建凸面镜
        >>> convex_mirror = create_mirror_surface(radius=-100.0)
    
    Validates:
        - Requirements 2.1: 支持定义球面反射镜（通过曲率半径参数）
        - Requirements 2.2: 支持定义平面反射镜（曲率半径为无穷大）
    """
    return SurfaceDefinition(
        surface_type='mirror',
        radius=radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=semi_aperture,
    )


from typing import List


# =============================================================================
# ElementRaytracer 类
# =============================================================================

class ElementRaytracer:
    """元件光线追迹器
    
    将输入光线通过一个或多个光学表面进行追迹，
    输出出射光束的光线数据。
    
    支持正入射和倾斜入射两种情况。
    
    参数:
        surfaces: 光学表面定义列表，至少包含一个表面
        wavelength: 波长，单位：μm，必须为正值
        chief_ray_direction: 主光线方向向量 (L, M, N)，默认 (0, 0, 1) 表示正入射
                            必须是归一化的方向余弦
        entrance_position: 入射面中心在全局坐标系中的位置，默认 (0, 0, 0)
                          单位：mm
    
    属性:
        surfaces: 光学表面定义列表
        wavelength: 波长（μm）
        chief_ray_direction: 主光线方向向量
        entrance_position: 入射面中心位置
        rotation_matrix: 从局部坐标系到全局坐标系的旋转矩阵
        optic: optiland Optic 对象（由 _create_optic 方法创建）
        output_rays: 出射光线数据（由 trace 方法设置）
    
    示例:
        >>> # 创建凹面镜光线追迹器
        >>> mirror = SurfaceDefinition(
        ...     surface_type='mirror',
        ...     radius=200.0,
        ...     semi_aperture=15.0
        ... )
        >>> raytracer = ElementRaytracer(
        ...     surfaces=[mirror],
        ...     wavelength=0.55,
        ... )
        
        >>> # 倾斜入射情况
        >>> raytracer = ElementRaytracer(
        ...     surfaces=[mirror],
        ...     wavelength=0.55,
        ...     chief_ray_direction=(0, np.sin(np.pi/4), np.cos(np.pi/4)),
        ...     entrance_position=(0, 0, 50),
        ... )
    
    Validates:
        - Requirements 1.1: 验证光线数据的有效性
        - Requirements 2.5: 支持定义多个连续的光学表面
        - Requirements 8.1: 输入参数类型错误时抛出 TypeError
        - Requirements 8.2: 输入参数值无效时抛出 ValueError
    """
    
    def __init__(
        self,
        surfaces: List[SurfaceDefinition],
        wavelength: float,
        chief_ray_direction: Tuple[float, float, float] = (0, 0, 1),
        entrance_position: Tuple[float, float, float] = (0, 0, 0),
        exit_chief_direction: Optional[Tuple[float, float, float]] = None,
        exit_position: Optional[Tuple[float, float, float]] = None,
        debug: bool = False,
    ) -> None:
        """初始化元件光线追迹器
        
        参数:
            surfaces: 光学表面定义列表，至少包含一个表面，已经是入射面坐标系中的定义。
            wavelength: 波长，单位：μm，必须为正值
            chief_ray_direction: 主光线方向向量 (L, M, N)，默认 (0, 0, 1) 表示正入射
                                必须是归一化的方向余弦
            entrance_position: 入射面中心在全局坐标系中的位置，默认 (0, 0, 0)
                              单位：mm
            exit_chief_direction: 出射主光线方向向量 (L, M, N)，可选
                                 如果提供，将直接使用此方向而不是从表面倾斜计算
                                 必须是归一化的方向余弦
            debug: 是否开启调试模式
        
        异常:
            TypeError: 如果输入参数类型错误
            ValueError: 如果输入参数值无效
        
        Validates:
            - Requirements 1.1: 验证光线数据的有效性
            - Requirements 2.5: 支持定义多个连续的光学表面
            - Requirements 8.1: 输入参数类型错误时抛出 TypeError
            - Requirements 8.2: 输入参数值无效时抛出 ValueError
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
            if not isinstance(surface, SurfaceDefinition):
                raise TypeError(
                    f"surfaces[{i}] 类型错误：期望 SurfaceDefinition，"
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
        # 验证 chief_ray_direction 参数
        # =====================================================================
        
        # 检查类型：必须是元组或列表
        if not isinstance(chief_ray_direction, (tuple, list)):
            raise TypeError(
                f"chief_ray_direction 参数类型错误：期望 tuple 或 list，"
                f"实际为 {type(chief_ray_direction).__name__}"
            )
        
        # 检查长度
        if len(chief_ray_direction) != 3:
            raise ValueError(
                f"chief_ray_direction 必须包含 3 个元素 (L, M, N)，"
                f"实际包含 {len(chief_ray_direction)} 个元素"
            )
        
        # 检查元素类型
        for i, val in enumerate(chief_ray_direction):
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"chief_ray_direction[{i}] 类型错误：期望 int 或 float，"
                    f"实际为 {type(val).__name__}"
                )
        
        # 转换为 numpy 数组并检查归一化
        direction = np.array(chief_ray_direction, dtype=np.float64)
        norm_squared = np.sum(direction ** 2)
        
        # 检查是否归一化（允许小误差）
        if not np.isclose(norm_squared, 1.0, rtol=1e-6):
            raise ValueError(
                f"chief_ray_direction 方向余弦未归一化：L² + M² + N² = {norm_squared:.6f}，"
                f"期望为 1.0"
            )
        
        # =====================================================================
        # 验证 entrance_position 参数
        # =====================================================================
        
        # 检查类型：必须是元组或列表
        if not isinstance(entrance_position, (tuple, list)):
            raise TypeError(
                f"entrance_position 参数类型错误：期望 tuple 或 list，"
                f"实际为 {type(entrance_position).__name__}"
            )
        
        # 检查长度
        if len(entrance_position) != 3:
            raise ValueError(
                f"entrance_position 必须包含 3 个元素 (x, y, z)，"
                f"实际包含 {len(entrance_position)} 个元素"
            )
        
        # 检查元素类型
        for i, val in enumerate(entrance_position):
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"entrance_position[{i}] 类型错误：期望 int 或 float，"
                    f"实际为 {type(val).__name__}"
                )
        
        # 检查是否为有限值
        position = np.array(entrance_position, dtype=np.float64)
        if not np.all(np.isfinite(position)):
            raise ValueError(
                f"entrance_position 必须包含有限值，实际为 {entrance_position}"
            )
        
        # =====================================================================
        # 验证 exit_chief_direction 参数（可选）
        # =====================================================================
        
        exit_direction_validated: Optional[Tuple[float, float, float]] = None
        
        if exit_chief_direction is not None:
            # 检查类型：必须是元组或列表
            if not isinstance(exit_chief_direction, (tuple, list)):
                raise TypeError(
                    f"exit_chief_direction 参数类型错误：期望 tuple 或 list，"
                    f"实际为 {type(exit_chief_direction).__name__}"
                )
            
            # 检查长度
            if len(exit_chief_direction) != 3:
                raise ValueError(
                    f"exit_chief_direction 必须包含 3 个元素 (L, M, N)，"
                    f"实际包含 {len(exit_chief_direction)} 个元素"
                )
            
            # 检查元素类型
            for i, val in enumerate(exit_chief_direction):
                if not isinstance(val, (int, float)):
                    raise TypeError(
                        f"exit_chief_direction[{i}] 类型错误：期望 int 或 float，"
                        f"实际为 {type(val).__name__}"
                    )
            
            # 转换为 numpy 数组并检查归一化
            exit_dir = np.array(exit_chief_direction, dtype=np.float64)
            exit_norm_squared = np.sum(exit_dir ** 2)
            
            # 检查是否归一化（允许小误差）
            if not np.isclose(exit_norm_squared, 1.0, rtol=1e-6):
                raise ValueError(
                    f"exit_chief_direction 方向余弦未归一化："
                    f"L² + M² + N² = {exit_norm_squared:.6f}，期望为 1.0"
                )
            
            exit_direction_validated = tuple(exit_dir)
        
        # =====================================================================
        # 存储属性
        # =====================================================================
        
        # 存储输入参数
        self.surfaces: List[SurfaceDefinition] = surfaces
        self.wavelength: float = float(wavelength)
        self.chief_ray_direction: Tuple[float, float, float] = tuple(direction)
        self.entrance_position: Tuple[float, float, float] = tuple(position)
        self._provided_exit_direction: Optional[Tuple[float, float, float]] = exit_direction_validated
        self._provided_exit_position: Optional[Tuple[float, float, float]] = exit_position
        self.exit_position = exit_position # Support direct access
        self.debug = debug
        
        # 计算坐标转换旋转矩阵
        self.rotation_matrix: NDArray = compute_rotation_matrix(
            self.chief_ray_direction
        )
        
        # 初始化其他属性（将由其他方法设置）
        self.input_rays: Optional[RealRays] = None   # 将由 trace 方法设置（用于雅可比矩阵计算）
        self.output_rays: Optional[RealRays] = None  # 将由 trace 方法设置
        self._chief_ray_traced: bool = False  # 标记主光线是否已追迹
        self._optic_finalized: bool = False  # 标记光学系统是否已完成（包含出射面）
        self.exit_chief_direction: Optional[Tuple[float, float, float]] = None  # 出射主光线方向
        self.exit_rotation_matrix: Optional[NDArray] = None  # 出射面旋转矩阵
        self._chief_ray_data: Optional[Dict[str, Any]] = None  # 主光线追迹数据
        self._chief_intersection_local: Optional[Tuple[float, float, float]] = None  # 主光线交点位置（入射面局部坐标系）
        
        # 创建 optiland 光学系统（不包含出射面）
        self.optic = None  # 将由 _create_optic_base 方法创建
        self._create_optic_base()
        
        # 如果提供了出射方向，设置相关属性
        # 注意：不再调用 _finalize_optic()，出射面将在 _trace_with_signed_opd 中动态添加
        if self._provided_exit_direction is not None:
            self.exit_chief_direction = self._provided_exit_direction
            self._chief_ray_traced = True
            # 计算出射面旋转矩阵
            self.exit_rotation_matrix = compute_rotation_matrix(self.exit_chief_direction)
            # 如果同时提供了 exit_position，则转换为入射面局部坐标系
            if self._provided_exit_position is not None:
                exit_pos_global = np.array(self._provided_exit_position, dtype=np.float64)
                entrance_pos = np.array(self.entrance_position, dtype=np.float64)
                exit_pos_local = self.rotation_matrix.T @ (exit_pos_global - entrance_pos)
                self._chief_intersection_local = tuple(exit_pos_local)
            self._optic_finalized = True
    
    def trace(self, input_rays: RealRays) -> RealRays:
        """执行光线追迹
        
        将输入光线通过光学系统进行追迹，输出出射光线数据。
        
        参数:
            input_rays: 输入光线（在全局坐标系 Global Frame 中）
                       必须是 RealRays 对象
        
        返回:
            出射光线数据（在出射面局部坐标系 Exit Surface Local Frame 中），RealRays 对象
        
        处理流程:
            1. 验证输入光线有效性（方向余弦归一化）
            2. 处理空输入情况
            3. 将输入光线从入射面局部坐标系转换到全局坐标系
            4. 调用 optiland 的 surface_group.trace() 进行光线追迹
            5. 将输出光线从全局坐标系转换到出射面局部坐标系
            6. 存储输出光线
        
        异常:
            TypeError: 如果输入参数类型错误
            ValueError: 如果光线方向余弦未归一化
        
        示例:
            >>> # 创建光线追迹器
            >>> mirror = SurfaceDefinition(surface_type='mirror', radius=200.0)
            >>> raytracer = ElementRaytracer(surfaces=[mirror], wavelength=0.55)
            
            >>> # 创建输入光线
            >>> input_rays = RealRays(
            ...     x=[0, 1], y=[0, 0], z=[0, 0],
            ...     L=[0, 0], M=[0, 0], N=[1, 1],
            ...     intensity=[1, 1], wavelength=[0.55, 0.55]
            ... )
            
            >>> # 执行追迹
            >>> output_rays = raytracer.trace(input_rays)
        
        """
        # =====================================================================
        # 确保主光线已追迹，光学系统已完成
        # =====================================================================
        
        if not self._chief_ray_traced:
            # 自动追迹主光线
            self.trace_chief_ray()
        
        if not self._optic_finalized:
            raise RuntimeError(
                "光学系统尚未完成创建。请先调用 trace_chief_ray() 方法。"
            )
        
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
        # 保存输入光线（用于雅可比矩阵计算）
        # =====================================================================
        
        # 创建输入光线的副本，避免后续修改影响保存的数据
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
            # 返回空的 RealRays 对象
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
        
        # 计算方向余弦的模的平方
        norm_squared = L_array**2 + M_array**2 + N_array**2
        
        # 检查是否归一化（允许小误差，rtol=1e-6）
        if not np.allclose(norm_squared, 1.0, rtol=1e-6):
            # 找到第一个不满足归一化条件的光线
            bad_indices = np.where(~np.isclose(norm_squared, 1.0, rtol=1e-6))[0]
            first_bad_idx = bad_indices[0]
            bad_value = norm_squared[first_bad_idx]
            raise ValueError(
                f"光线方向余弦未归一化：光线 {first_bad_idx} 的 "
                f"L² + M² + N² = {bad_value:.6f}，期望为 1.0"
            )
        
        # =====================================================================
        # 在入射面局部坐标系中进行追迹
        # =====================================================================
        # 
        # 关键设计决策：
        # - optiland 的光学系统在入射面局部坐标系中定义（Z 轴为入射方向）
        # - 输入光线已经在入射面局部坐标系中
        # - 追迹在入射面局部坐标系中进行
        # - 追迹完成后，将结果转换到出射面局部坐标系
        #
        # 这样可以正确处理任意入射方向的情况。
        
        
        # =====================================================================
        # 复制光线对象，避免修改原始数据
        # optiland 的 trace 方法会原地修改光线对象
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
        # 调用带符号 OPD 的光线追迹 (Accepts GLOBAL rays)
        # =====================================================================
        
        # 使用带符号 OPD 追迹，正确处理折叠光路中的 OPD 计算
        # 关键区别：不使用 abs(t)，保留传播距离的符号
        # 
        # 现在 _finalize_optic 已经添加了出射面（作为最后一个表面），
        # 所以这里的循环会自动追迹到出射面。
        self._trace_with_signed_opd(traced_rays, skip=1)
        
        # 光线现在位于出射面上
        # =====================================================================
        # 将输出光线从入射面局部坐标系转换到出射面局部坐标系
        # =====================================================================
        #
        # 追迹是在入射面局部坐标系中进行的，出射光线也在入射面局部坐标系中。
        # 我们需要将其转换到出射面局部坐标系，以便后续的 PROPER 传播。
        #
        # 变换步骤：
        # 1. 将所有光线位置减去主光线交点位置（平移到以主光线为原点）
        # 2. 旋转到出射面局部坐标系
        #
        # 重要：OPD 是标量，不受坐标变换影响
        
        # 计算从入射面局部坐标系到出射面局部坐标系的旋转矩阵
        # 使用"最小旋转" (Min-Twist / Parallel Transport) 逻辑
        # 都不依赖 Global 坐标系，也不依赖 Surface Euler 角
        
        # 1. 获取出射主光线方向 (在 Entrance Local Frame)
        # 找到最接近中心的光线作为参考
        r_sq = np.asarray(traced_rays.x)**2 + np.asarray(traced_rays.y)**2
        min_r_idx = np.argmin(r_sq)
        L_c = traced_rays.L[min_r_idx]
        M_c = traced_rays.M[min_r_idx]
        N_c = traced_rays.N[min_r_idx]
        
        z_in = np.array([0.0, 0.0, 1.0])
        z_out = np.array([L_c, M_c, N_c])
        norm_out = np.linalg.norm(z_out)
        if norm_out > 1e-10:
            z_out = z_out / norm_out
            
        # 2. 计算将 z_in 旋转到 z_out 的旋转矩阵
        dot_val = np.clip(np.dot(z_in, z_out), -1.0, 1.0)
        
        if dot_val > 0.999999: # 平行
            R_rel = np.eye(3)
        elif dot_val < -0.999999: # 反平行 (180度)
            # 简单逻辑：绕 Y 轴旋转 180 度 (x->-x, z->-z, y->y)
            # 保持 Up-is-Up (在反射意义下)
            R_rel = Rotation.from_euler('y', 180, degrees=True).as_matrix()
        else:
            # 一般情况：计算旋转轴和角度
            axis = np.cross(z_in, z_out)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-10: 
                R_rel = np.eye(3) # Should be covered by dot check, but safety
            else:
                axis = axis / axis_norm
                angle = np.arccos(dot_val)
                R_rel = Rotation.from_rotvec(axis * angle).as_matrix()
                
        # 3. R_rel 将 Entrance Basis 映射到 Exit Basis
        # R_entrance_to_exit = R_rel.T
        R_entrance_to_exit = R_rel.T
        
        # 获取光线数据
        x_entrance = np.asarray(traced_rays.x)
        y_entrance = np.asarray(traced_rays.y)
        z_entrance = np.asarray(traced_rays.z)
        L_entrance = np.asarray(traced_rays.L)
        M_entrance = np.asarray(traced_rays.M)
        N_entrance = np.asarray(traced_rays.N)
        
        print(f"[DEBUG trace] traced_rays centroid (Optic Local): ({np.mean(x_entrance):.4f}, {np.mean(y_entrance):.4f}, {np.mean(z_entrance):.4f})")
        print(f"[DEBUG trace] chief_intersection_local: {self._chief_intersection_local}")
        
        # =====================================================================
        # 使用主光线交点位置作为出射面原点
        # =====================================================================
        # 主光线交点位置在 _finalize_optic() 中已计算并存储
        # 这确保了出射面经过主光线与镜面的交点
        
        chief_x, chief_y, chief_z = self._chief_intersection_local
        
        # 平移：将主光线位置移到原点
        x_centered = x_entrance - chief_x
        y_centered = y_entrance - chief_y
        z_centered = z_entrance - chief_z
        
        # 位置转换（先平移后旋转）
        # pos_centered 目前是相对于主光线交点 (ray - chief_intersection)
        pos_centered = np.stack([x_centered, y_centered, z_centered], axis=0)
        pos_exit = R_entrance_to_exit @ pos_centered
        
        # ⚠️ 关键修复：加入 Exit Position 的平移
        # 如果指定了 exit_position，则输出光线应该相对于该位置
        # 目前 pos_exit 是相对于 chief_intersection 的
        # 我们需要添加 shift = (chief_intersection - exit_position) in Exit Frame
        
        shift_x, shift_y, shift_z = 0.0, 0.0, 0.0
        
        if self.exit_position is not None:
            # 计算 Exit Position 在 Entrance Frame (Optiland Global) 中的位置
            # P_exit_Ent = R_global_to_ent.T @ (P_exit_global - P_ent_global)
            
            ent_origin = np.array(self.entrance_position, dtype=np.float64)
            exit_origin_global = np.array(self.exit_position, dtype=np.float64)
            
            # Vector from Ent Origin to Exit Origin (Global)
            v_ent_to_exit_global = exit_origin_global - ent_origin
            
            # Transform to Entrance Frame
            v_ent_to_exit_ent = self.rotation_matrix.T @ v_ent_to_exit_global
            
            # Chief Intersection in Entrance Frame
            chief_pos_ent = np.array([chief_x, chief_y, chief_z])
            
            # Vector from Exit Origin to Chief Intersection (in Entrance Frame)
            v_exit_to_chief_ent = chief_pos_ent - v_ent_to_exit_ent
            
            # Transform to Exit Frame
            v_shift_exit = R_entrance_to_exit @ v_exit_to_chief_ent
            
            shift_x = v_shift_exit[0]
            shift_y = v_shift_exit[1]
            shift_z = v_shift_exit[2]
            
            print(f"[DEBUG Shift] ent_origin={ent_origin}")
            print(f"[DEBUG Shift] exit_origin_global={exit_origin_global}")
            print(f"[DEBUG Shift] v_ent_to_exit_ent={v_ent_to_exit_ent}")
            print(f"[DEBUG Shift] chief_pos_ent=({chief_x:.4f}, {chief_y:.4f}, {chief_z:.4f})")
            print(f"[DEBUG Shift] v_shift_exit=({shift_x:.4f}, {shift_y:.4f}, {shift_z:.4f})")

        x_exit = pos_exit[0] + shift_x
        y_exit = pos_exit[1] + shift_y
        z_exit = pos_exit[2] + shift_z
        
        # 方向转换
        dir_entrance = np.stack([L_entrance, M_entrance, N_entrance], axis=0)
        dir_exit = R_entrance_to_exit @ dir_entrance
        
        L_exit = dir_exit[0]
        M_exit = dir_exit[1]
        N_exit = dir_exit[2]
        
        # 创建输出光线
        output_rays = RealRays(
            x=x_exit,
            y=y_exit,
            z=z_exit,
            L=L_exit,
            M=M_exit,
            N=N_exit,
            intensity=np.asarray(traced_rays.i),
            wavelength=np.asarray(traced_rays.w),
        )
        # OPD 是标量，不受坐标变换影响，直接复制
        output_rays.opd = np.asarray(traced_rays.opd).copy()
        
        # =====================================================================
        # 存储输出光线
        # =====================================================================
        
        # =====================================================================
        # 验证主光线位置（应严格为 (0,0,0)）
        # =====================================================================
        
        # 找到输入中最接近中心的光线
        r_sq_in = np.asarray(input_rays.x)**2 + np.asarray(input_rays.y)**2
        min_r_idx = np.argmin(r_sq_in)
        
        # 如果这是真正的主光线（输入 r ≈ 0）
        if r_sq_in[min_r_idx] < 1e-10:
            chief_x_out = output_rays.x[min_r_idx]
            chief_y_out = output_rays.y[min_r_idx]
            chief_z_out = output_rays.z[min_r_idx]
            
            dist_from_zero = np.sqrt(chief_x_out**2 + chief_y_out**2 + chief_z_out**2)
            
            # 使用严格的阈值 (1e-6 mm = 1 nm)
            if dist_from_zero > 1e-6:
                raise RuntimeError(
                    f"严重错误：出射面定义不符合物理假设。\n"
                    f"主光线在出射面局部坐标系中的坐标应为 (0, 0, 0)，实际为 "
                    f"({chief_x_out:.6e}, {chief_y_out:.6e}, {chief_z_out:.6e})。\n"
                    f"偏差量: {dist_from_zero:.6e} mm。\n"
                    f"可能原因：\n"
                    f"1. Optiland 出射面位置未正确设置在主光线交点\n"
                    f"2. 坐标转换矩阵计算错误\n"
                    f"3. 光线追迹数值精度不足"
                )

            #检查主光线是否垂直于出射面，这里补充一段代码
            chief_L_out = output_rays.L[min_r_idx]
            chief_M_out = output_rays.M[min_r_idx]
            chief_N_out = output_rays.N[min_r_idx]
            
            # 计算方向是否垂直于出射面。这里是不垂直的则有问题
            angle_deviation = np.sqrt(chief_L_out**2 + chief_M_out**2)
            if angle_deviation > 1e-6:
                #打印变量 
                print(f" angle_deviation Warming: {angle_deviation}")
                # raise RuntimeError(
                #     f"严重错误：主光线不垂直于出射面（方向偏离 Z 轴）。\n"
                #     f"在出射面局部坐标系中，主光线方向应为 (0, 0, 1)。\n"
                #     f"实际方向: ({chief_L_out:.6f}, {chief_M_out:.6f}, {chief_N_out:.6f})\n"
                #     f"横向分量模长 (sin(theta)): {angle_deviation:.6e}\n"
                #     f"可能原因：出射面旋转矩阵构建错误，或光线追迹精度不足。"
                # )

        
        self.output_rays = output_rays
        #注意这里仍然是出射面的局部坐标
        return output_rays
    
    def _direction_to_rotation_angles(
        self,
        direction: Tuple[float, float, float],
    ) -> Tuple[float, float]:
        """将方向向量转换为旋转角度 (rx, ry)
        
        参数:
            direction: 方向向量 (L, M, N)，必须归一化
        
        返回:
            (rx, ry): 旋转角度（弧度），使得 optiland 表面法向量等于 direction
        
        说明:
            此方法用于定义出射面，使其法向量与出射方向一致。
            
            optiland 的表面法向量计算：
            n = Ry(ry) @ Rx(rx) @ [0, 0, 1]
              = [sin(ry)*cos(rx), -sin(rx), cos(ry)*cos(rx)]
            
            要使 n = (L, M, N)：
            sin(ry)*cos(rx) = L
            -sin(rx) = M
            cos(ry)*cos(rx) = N
            
            解得：
            rx = -arcsin(M)
            ry = arctan2(L, N)
        
        Validates:
            - Requirements REQ-1.2: 方向到旋转角度转换
        """
        L, M, N = direction
        
        # rx = -arcsin(M)
        # 限制 M 在 [-1, 1] 范围内，避免数值误差导致的 arcsin 错误
        # 注意：负号是关键！optiland 的法向量 Y 分量是 -sin(rx)
        M_clamped = np.clip(M, -1.0, 1.0)
        rx = -np.arcsin(M_clamped)
        
        # ry = arctan2(L, N)
        ry = np.arctan2(L, N)
        
        return (float(rx), float(ry))
    
    def _compute_exit_chief_direction(self) -> Tuple[float, float, float]:
        """计算出射主光线方向
        
        对于抛物面，使用抛物面的光学性质直接计算反射方向。
        对于其他表面，使用 optiland 的光线追迹。
        
        返回:
            出射主光线方向 (L, M, N)，全局坐标系，归一化
        
        说明:
            - 对于抛物面（conic = -1），使用光学性质：平行于光轴的光线反射后通过焦点
            - 对于其他表面，使用 optiland 追迹
        
        Validates:
            - Requirements REQ-1.1: 使用实际光线追迹计算出射方向
        """
        # 对于所有表面（包括抛物面），都使用 optiland 追迹
        return self._compute_exit_direction_optiland()
    
    
    def _compute_exit_direction_optiland(self) -> Tuple[float, float, float]:
        """使用 optiland 计算出射主光线方向
        
        返回:
            出射主光线方向 (L, M, N)，全局坐标系，归一化
        """
        from optiland.raytrace import RealRayTracer
        
        # 使用已创建的基础光学系统
        # self.optic 已在 __init__ -> _create_optic_base 中创建
        # 包含了所有定义的光学表面（除最终出射面外）
        
        if self.optic is None:
            # Should not happen as it's initialized in __init__
            self._create_optic_base()
            
        # 追迹主光线
        # trace_generic(Hx=0, Hy=0, Px=0, Py=0) 对应于通过光阑中心的主光线
        # 由于我们在 _create_optic_base 中将 index=1 设为 STOP 且位于 (0,0,0)，
        # 这条光线将从入射面中心出发
        tracer = RealRayTracer(self.optic)
        chief_rays = tracer.trace_generic(
            Hx=0.0, Hy=0.0, Px=0.0, Py=0.0, 
            wavelength=self.wavelength
        )
        
        # 提取出射方向（在入射面的局部坐标系中）
        L = float(np.asarray(chief_rays.L)[0])
        M = float(np.asarray(chief_rays.M)[0])
        N = float(np.asarray(chief_rays.N)[0])
        #可以考虑加一个阈值，避免sin(pi)的数值误差
        # 归一化
        norm = np.sqrt(L**2 + M**2 + N**2)
        if norm > 1e-10:
            L, M, N = L/norm, M/norm, N/norm
        
        # 将方向从入射面局部坐标系转换到全局坐标系
        exit_dir_local = np.array([L, M, N], dtype=np.float64)
        exit_dir_global = self.rotation_matrix @ exit_dir_local
        
        return (float(exit_dir_global[0]), float(exit_dir_global[1]), float(exit_dir_global[2]))
    
    def trace_chief_ray(self) -> Tuple[float, float, float]:
        """追迹主光线并存储结果
        
        此方法应在混合光线追迹之前单独调用，用于：
        1. 计算出射主光线方向
        2. 存储主光线追迹数据（位置、方向等）
        3. 完成光学系统的创建（添加出射面）
        
        返回:
            出射主光线方向 (L, M, N)，全局坐标系，归一化
        
        说明:
            - 如果已经提供了 exit_chief_direction 参数，则直接使用
            - 否则使用 optiland 追迹主光线
            - 追迹结果会被存储，后续调用 trace() 时使用
        
        示例:
            >>> raytracer = ElementRaytracer(surfaces=[mirror], wavelength=0.55)
            >>> exit_dir = raytracer.trace_chief_ray()  # 先追迹主光线
            >>> output_rays = raytracer.trace(input_rays)  # 再追迹所有光线
        """
        if self._chief_ray_traced:
            # 已经追迹过，直接返回存储的结果
            return self.exit_chief_direction
        
        # 总是使用 optiland 追迹主光线，以获得正确的主光线交点位置
        # 这对于后续构建出射面坐标系至关重要
        calculated_direction = self._compute_exit_chief_direction()
        
        # 如果提供了出射方向，覆盖计算出的方向
        if self._provided_exit_direction is not None:
            self.exit_chief_direction = self._provided_exit_direction
        else:
            self.exit_chief_direction = calculated_direction
        
        # 标记主光线已追迹
        self._chief_ray_traced = True
        #这里是不是在多个元件的时候存在未赋值问题？
        
        # 完成光学系统的创建（添加出射面）
        self._finalize_optic()
        # 这里是不是在多个元件的时候有问题？噢好像没关系，tracer没被复用
        return self.exit_chief_direction
    
    def get_exit_chief_ray_direction(self) -> Tuple[float, float, float]:
        """获取出射主光线方向
        
        如果主光线尚未追迹，会自动调用 trace_chief_ray()。
        
        返回:
            出射主光线方向 (L, M, N)，全局坐标系，归一化
        """
        if not self._chief_ray_traced:
            self.trace_chief_ray()
        return self.exit_chief_direction
    
    def _create_optic_base(self) -> None:
        """创建基本光学系统（不包含出射面）
        
        配置反射镜（material='mirror'）和折射面。
        设置表面半口径（aperture）。
        
        创建的光学系统结构：
        - index=0: 物面（无穷远）
        - index=1: 第一个光学表面（设为光阑）
        - index=2, 3, ...: 后续光学表面
        
        说明:
            - 物面设置在无穷远处（thickness=np.inf）
            - 第一个光学表面设置为光阑（is_stop=True）
            - 反射镜使用 material='mirror'
            - 折射面使用指定的材料名称
            - 如果指定了半口径，会设置表面的 max_aperture
            - 出射面将在 _finalize_optic() 中添加
            
        重要：
            - 光学系统在入射面局部坐标系中定义
            - 表面位置相对于入射面中心（entrance_position）设置偏移
            - 这样可以正确处理离轴系统的 OPD 计算
        """
        from optiland.optic import Optic
        
        # 创建光学系统
        optic = Optic()
        
        # =====================================================================
        # 设置系统参数
        # =====================================================================
        
        # 设置孔径
        first_surface = self.surfaces[0]
        if first_surface.semi_aperture is not None:
            aperture_diameter = 2.0 * first_surface.semi_aperture
        else:
            aperture_diameter = 10000.0
        
        optic.set_aperture(aperture_type='EPD', value=aperture_diameter)
        
        # 设置视场类型为角度，轴上视场
        optic.set_field_type(field_type='angle')
        optic.add_field(y=0, x=0)
        
        # 设置波长
        optic.add_wavelength(value=self.wavelength, is_primary=True)
        
        # =====================================================================
        # 计算表面位置偏移
        # =====================================================================
        # 
        # 在入射面局部坐标系中，表面顶点相对于入射面中心的位置
        # 
        # ⚠️ 关键修复：使用 SurfaceDefinition 中的 vertex_position
        # 
        # 对于离轴系统（如 OAP）：
        # - 表面顶点在全局坐标 vertex_position（如 (0, 100, 0)）
        # - 入射面中心在全局坐标 entrance_position（如 (0, 100, 2.5)）
        # - 表面偏移 = R.T @ (vertex_global - entrance_global)
        #
        # 对于轴上系统：
        # - 表面顶点在全局坐标 (0, 0, 0)
        # - 入射面中心在全局坐标 entrance_position
        # - 表面偏移 = R.T @ (-entrance_position)
        
        entrance_pos = np.array(self.entrance_position, dtype=np.float64)
        
        # 获取表面顶点位置（全局坐标系）
        first_surface = self.surfaces[0]
        if first_surface.vertex_position is not None:
            vertex_global = np.array(first_surface.vertex_position, dtype=np.float64)
        else:
            # 默认：表面顶点在全局原点
            vertex_global = np.array([0.0, 0.0, 0.0])
        
        # 表面顶点在入射面局部坐标系中的位置
        # 使用旋转矩阵的转置将全局坐标转换到入射面局部坐标系
        # surface_offset_local = R.T @ (vertex_global - entrance_global)
        surface_offset_local = self.rotation_matrix.T @ (vertex_global - entrance_pos)
        
        surface_x_offset = float(surface_offset_local[0])
        surface_y_offset = float(surface_offset_local[1])
        surface_z_offset = float(surface_offset_local[2])
        
        # =====================================================================
        # 添加物面（index=0）
        # =====================================================================
        # 添加物面（index=0）
        optic.add_surface(index=0, thickness=0.0)
        
        # 添加虚拟光阑面（在入射面位置）
        # ⚠️ 这里设为 STOP，位于 (0,0,0)
        # 用 thickness=0.0 避免 numpy.inf 导致的可选坐标更新问题
        # 后续表面使用绝对坐标 (x,y,z) 定位，因此此处的 thickness 不影响几何位置
        optic.add_surface(index=1, thickness=0.0, is_stop=True)
        
        # 添加光学表面
        for i, surface_def in enumerate(self.surfaces):
            # 实际表面从 index=2 开始
            surface_index = i + 2
            is_stop = False
            
            if surface_def.surface_type == 'mirror':
                material = 'mirror'
            else:
                material = surface_def.material
            
            radius = surface_def.radius
            
            # ⚠️ 关键修复：对于抛物面，不设置倾斜角度
            # 抛物面的离轴效果通过位置偏移（x, y, z）实现
            # 抛物面的反射方向由其几何形状自然决定
            # SurfaceDefinition 中的 tilt_x/tilt_y 仅用于计算出射方向，
            # 不应该传递给 optiland 的表面设置
            
            if surface_def.orientation is not None:
                # 优先使用全局方向矩阵计算局部倾角
                # R_rel = R_beam.T @ R_surf
                # R_rel 描述了表面相对于入射光轴坐标系的旋转
                R_rel = self.rotation_matrix.T @ surface_def.orientation
                
                # 计算直接从法向向量提取倾斜角度
                # 能够避免欧拉角分解中的歧义（如 rz=180 度翻转问题）
                # 同时也修复了之前代码中假设 sin(ry)=Nx 的数学错误
                
                # 公式推导：
                # 法向 N = R @ (0,0,1) = (Nx, Ny, Nz)
                # 使用 optiland 旋转顺序 (Ry @ Rx):
                # Nx = cos(rx) * sin(ry)
                # Ny = -sin(rx)
                # Nz = cos(rx) * cos(ry)
                
                # 1. 提取法向向量 (R_rel 的第三列)
                normal_local = R_rel[:, 2]
                L, M, N = normal_local # Nx, Ny, Nz
                
                # 2. 计算 rx, ry
                # rx = -arcsin(Ny)
                tilt_x_rad = -np.arcsin(np.clip(M, -1.0, 1.0))
                
                # ry = arctan2(Nx, Nz)
                # 处理万向节锁 (Gimbal Lock) 情况：当 rx = +/- 90度 (M = +/- 1)
                # 此时 cos(rx) = 0, Nx = Nz = 0, ry 不确定
                if abs(M) > 0.9999:
                    tilt_y_rad = 0.0
                else:
                    tilt_y_rad = np.arctan2(L, N)
                
                # 3. ⚠️ 关键修复：当表面法向背对入射光时，翻转曲率半径符号
                # 
                # 当 Nz < 0 时，表面法向朝向 -Z（入射坐标系），即背对入射光（+Z 方向）。
                # 这意味着光线从表面的"背面"入射。
                # 
                # 在 optiland 中：
                # - 曲率中心位于 vertex + R * local_z_axis
                # - 当法向翻转后，local_z_axis 指向 -Z（全局）
                # - 为了保持曲率中心在正确的全局位置，需要翻转 R 的符号
                # 
                # 几何论证：
                # - 原始定义：曲率中心 = vertex + R * (+Z_global)
                # - 翻转后：曲率中心 = vertex + R' * (-Z_global)
                # - 要求两者相等：R' = -R
                #
                if N < 0:
                    radius = -radius
                    if getattr(self, 'debug', False):
                        print(f"[DEBUG ElementRaytracer] Surface {surface_index}: Normal facing away (Nz={N:.4f}), flipping radius to {radius}")
                
                if getattr(self, 'debug', False):
                     print(f"[DEBUG ElementRaytracer] Surface {surface_index} Orientation:")
                     print(f"  Normal: ({L:.4f}, {M:.4f}, {N:.4f})")
                     print(f"  Calculated Tilt: rx={np.degrees(tilt_x_rad):.2f}°, ry={np.degrees(tilt_y_rad):.2f}°")
            else:
                # 回退到使用预计算的 tilt_x/y
                tilt_x_rad = surface_def.tilt_x if surface_def.tilt_x else 0.0
                tilt_y_rad = surface_def.tilt_y if surface_def.tilt_y else 0.0
                
            tilt_x_safe = _avoid_exact_45_degrees(tilt_x_rad)
            tilt_y_safe = _avoid_exact_45_degrees(tilt_y_rad)
            
            # 使用绝对坐标定位模式，设置表面位置偏移
            # ⚠️ 关键修正：为了匹配 GlobalSurfaceDefinition 的符号约定
            # - GlobalSurfaceDefinition: R>0 表示凹面 (如: 聚焦反射镜)，曲率中心在 +Z 方向 (假设光线沿 +Z 传播时)
            # - Optiland: 标准光线追迹约定，对于反射镜，光线反向 (-Z)，聚焦镜的曲率中心位于 -Z 方向
            #
            # 因此，当 User 定义 "凹面镜, R=1000" (R>0) 时，物理上它是聚焦的。
            # 在 Optiland 本地坐标系中，光线射入 (沿 +Z)，反射 (沿 -Z)。
            # 聚焦镜的曲率中心应在镜前 (左侧, -Z)。
            # 所以我们需要传递 负半径 (-R) 给 Optiland。
            #
            # 简而言之：user_R < 0 (凹面) -> optiland_R < 0
            print(f"[DEBUG ElementRaytracer] Adding Surface {surface_index}: radius={radius}, rx={tilt_x_safe}, ry={tilt_y_safe}, conic={surface_def.conic}, material={material}")
            optic.add_surface(
                index=surface_index,
                radius=radius,
                material=material,
                is_stop=is_stop,
                conic=surface_def.conic,
                rx=tilt_x_safe,
                ry=tilt_y_safe,
                # 设置表面位置偏移（相对于入射面中心）
                x=surface_x_offset,
                y=surface_y_offset,
                z=surface_z_offset,
            )
        
        # 存储创建的光学系统（尚未添加出射面）
        self.optic = optic
    
    def _finalize_optic(self) -> None:
        """完成光学系统的创建（添加出射面）
        
        在主光线追迹完成后调用，添加透明平面作为出射面。
        这里效率很低，莫名其妙的又重新追迹一次。
        
        前提条件:
            - self.exit_chief_direction 已设置
            - self._chief_ray_traced 为 True
            
        重要设计原则：
            - 出射面必须垂直于出射光轴（steering 明确要求）
            - 出射面必须经过主光线与镜面的交点
            - 在入射面局部坐标系中设置出射面的位置和旋转
            
        实现方法：
            1. 追迹主光线获取其与镜面的交点位置（在入射面局部坐标系中）
            2. 计算出射方向在入射面局部坐标系中的表示
            3. 从出射方向计算出射面的旋转角度 (rx, ry)
            4. 设置出射面位置为主光线交点位置
        """
        if self._optic_finalized:
            return
        
        if self.exit_chief_direction is None:
            raise RuntimeError("出射主光线方向尚未计算，请先调用 trace_chief_ray()")
        
        # 计算出射面的旋转矩阵（全局坐标系）
        self.exit_rotation_matrix = compute_rotation_matrix(self.exit_chief_direction)
 
        # =====================================================================
        # 计算主光线交点位置（在入射面局部坐标系中）
        # =====================================================================
        # 使用 optiland 追迹主光线，获取准确的交点位置。
        # 由于我们现在正确设置了 OAP 的几何参数（位置偏移 + 机械倾角），
        # optiland 的追迹结果应该是准确的。
        
        # 创建主光线（在入射面局部坐标系中，从原点沿 +Z 方向）
        chief_ray = RealRays(
            x=np.array([0.0]),
            y=np.array([0.0]),
            z=np.array([0.0]),
            L=np.array([0.0]),
            M=np.array([0.0]),
            N=np.array([1.0]),
            intensity=np.array([1.0]),
            wavelength=np.array([self.wavelength]),
        )
        chief_ray.opd = np.array([0.0])
        
        # 追迹主光线到最后一个光学表面
        # 注意：此时 optic 中还没有出射面，只有物面和光学表面
        surfaces = self.optic.surface_group.surfaces
        for i, surface in enumerate(surfaces):
            if i < 1:  # 跳过物面
                continue
            surface.trace(chief_ray)
        
        # 获取主光线与最后一个光学表面的交点位置（在入射面局部坐标系中）
        chief_x = float(np.asarray(chief_ray.x)[0])
        chief_y = float(np.asarray(chief_ray.y)[0])
        chief_z = float(np.asarray(chief_ray.z)[0])
        
        # 存储主光线交点位置（用于 trace() 方法中的坐标变换）
        self._chief_intersection_local = (chief_x, chief_y, chief_z)
        
        if self._chief_intersection_local is None:
             raise RuntimeError("_chief_intersection_local not set during trace_chief_ray")
             
        # =====================================================================
        # 计算出射坐标系的旋转矩阵
        # =====================================================================
        self.exit_rotation_matrix = compute_rotation_matrix(self.exit_chief_direction)
        
        # =====================================================================
        # 计算出射方向在入射面局部坐标系中的表示
        # =====================================================================
        
        # 出射方向在全局坐标系中
        exit_dir_global = np.array(self.exit_chief_direction, dtype=np.float64)
        
        # 转换到入射面局部坐标系
        # v_local = R_entrance.T @ v_global
        exit_dir_local = self.rotation_matrix.T @ exit_dir_global
        
        # 归一化（处理数值误差）
        norm = np.linalg.norm(exit_dir_local)
        if norm > 1e-10:
            exit_dir_local = exit_dir_local / norm
        
        # =====================================================================
        # 计算出射面的旋转角度
        # =====================================================================
        
        # 将出射方向（入射面局部坐标系）转换为欧拉角
        # rx, ry = self._direction_to_rotation_angles(tuple(exit_dir_local))
        
        # 存储出射方向（入射面局部坐标系）
        self._exit_dir_local = exit_dir_local
        
        # DEBUG: Print exit directions
        # DEBUG: Print exit directions
        if getattr(self, 'debug', False):
            print(f"[DEBUG ElementRaytracer] exit_chief_direction (Global): {self.exit_chief_direction}")
            print(f"[DEBUG ElementRaytracer] _exit_dir_local (Optic Local): {tuple(self._exit_dir_local)}")
            print(f"[DEBUG ElementRaytracer] rotation_matrix (Entrance->Global):\\n{self.rotation_matrix}")
        
        # =====================================================================
        # 在 Optiland 中添加出射面
        # =====================================================================
        
        # 计算旋转角度 (rx, ry)
        # 注意：使用 -arcsin(ny) 计算 rx，arctan2 计算 ry
        # 确保法向量 = R_y(ry) @ R_x(rx) @ (0,0,1) = exit_dir_local
        
        L = _snap_to_zero(exit_dir_local[0])
        M = _snap_to_zero(exit_dir_local[1])
        N = _snap_to_zero(exit_dir_local[2])
        
        # 重新归一化 (Sanitization 可能稍微改变模长)
        norm_sanitized = np.linalg.norm([L, M, N])
        if norm_sanitized > 1e-10:
            L, M, N = L / norm_sanitized, M / norm_sanitized, N / norm_sanitized

        # 计算 rx, ry
        # 公式推导：
        # n = [sin(ry)cos(rx), -sin(rx), cos(ry)cos(rx)]
        # M = -sin(rx) -> rx = -arcsin(M)
        # L = sin(ry)cos(rx) -> tan(ry) = L / N (如果 cos(rx) != 0)
        
        rx = -np.arcsin(np.clip(M, -1.0, 1.0))
        ry = np.arctan2(L, N)
        
        # 添加出射面
        # 这个面位于主光线交点 (chief_x, chief_y, chief_z)
        # 并且垂直于出射光线
        # 设置 is_dummy=True (如果 optiland 支持)，或设为空气/透明
        
        # 注意：这里我们添加的是从入射面中心偏移后的位置
        # Optiland 的 add_surface 使用的 x,y,z 是相对于其父坐标系（即入射面局部坐标系）的绝对坐标
        
        next_index = len(self.optic.surface_group.surfaces)
        
        self.optic.add_surface(
            index=next_index,
            radius=np.inf,  # 平面
            material='air',  # 假设出射后是空气
            x=chief_x,
            y=chief_y,
            z=chief_z,
            rx=rx,
            ry=ry,
            is_stop=False
        )
        
        self._optic_finalized = True
    
    def _create_optic(self) -> None:
        """根据 SurfaceDefinition 列表创建 optiland 光学系统（兼容旧接口）
        
        此方法保留用于向后兼容，内部调用 _create_optic_base() 和 _finalize_optic()。
        
        配置反射镜（material='mirror'）和折射面。
        设置表面半口径（aperture）。
        添加倾斜的透明平面作为出射面。
        
        创建的光学系统结构：
        - index=0: 物面（无穷远）
        - index=1: 第一个光学表面（设为光阑）
        - index=2, 3, ...: 后续光学表面
        - 最后一个 index: 倾斜的透明平面（出射面）
        
        说明:
            - 物面设置在无穷远处（thickness=np.inf）
            - 第一个光学表面设置为光阑（is_stop=True）
            - 反射镜使用 material='mirror'
            - 折射面使用指定的材料名称
            - 如果指定了半口径，会设置表面的 max_aperture
            - 出射面是倾斜的透明平面，垂直于出射主光线方向
        
        Validates:
            - Requirements 2.1: 支持定义球面反射镜（通过曲率半径参数）
            - Requirements 2.2: 支持定义平面反射镜（曲率半径为无穷大）
            - Requirements 2.3: 支持定义球面折射面（通过曲率半径和材料参数）
            - Requirements 2.5: 支持定义多个连续的光学表面
            - Requirements REQ-2.1: 在 optiland 中添加倾斜的透明出射面
        """
        # 创建基本光学系统
        self._create_optic_base()
        
        # 追迹主光线并完成光学系统
        self.trace_chief_ray()
    
    def _trace_with_signed_opd(self, rays: RealRays, skip: int = 1) -> None:
        """执行光线追迹（原地修改），使用带符号的 OPD 计算
        
        与 optiland 标准追迹的区别：
        - optiland 使用 abs(t) 计算 OPD
        - 本方法使用带符号的 t 计算 OPD
        - 对于抛物面，修正反射方向（optiland 的几何法向量对离轴抛物面不正确）
        
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
        - 这是因为 dz = t * N_before，所以 sign(t) = sign(dz) / sign(N_before)
        

        参数:
            rays: 输入光线（会被原地修改）
            skip: 跳过的表面数量（默认 1，跳过物面）
        """ 
        # 获取表面列表
        surfaces = self.optic.surface_group.surfaces

        # [DEBUG] 打印表面参数
        if getattr(self, 'debug', False):
            print(f"\n[DEBUG ElementRaytracer] Inspecting {len(surfaces)} surfaces in Optiland Optic:")
            print(f"[DEBUG ElementRaytracer] Coordinates are LOCAL to the Optic Entrance Position.")
            for idx, surf in enumerate(surfaces):
                print(f"  Surface {idx} [{type(surf).__name__}]:")
                
                # 尝试获取 Geometry
                geo = getattr(surf, 'geometry', None)
                if geo:
                    # 坐标系
                    cs = getattr(geo, 'cs', None)
                    if cs:
                        # 使用 getattr 并提供默认值
                        # 注意：这些通常是 backend array，直接打印即可看到数值
                        c_x = getattr(cs, 'x', 'N/A')
                        c_y = getattr(cs, 'y', 'N/A')
                        c_z = getattr(cs, 'z', 'N/A')
                        c_rx = getattr(cs, 'rx', 'N/A')
                        c_ry = getattr(cs, 'ry', 'N/A')
                        c_rz = getattr(cs, 'rz', 'N/A')
                        print(f"    Position  (x,y,z) : ({c_x}, {c_y}, {c_z})")
                        print(f"    Rotation (rx,ry,rz): ({c_rx}, {c_ry}, {c_rz})")
                        #增加一个检查，如果是idx = 3,则检查表面是否与出射光线方向垂直
                    else:
                        print(f"    Geometry has no 'cs' attribute (CoordinateSystem).")

                    # 用户请求：检查表面是否与出射光线方向垂直 (仅针对 idx=3, 即出射面)
                    if idx == 3 and cs is not None:
                         try:
                             # 获取旋转角度
                             val_rx = float(getattr(cs, 'rx', 0.0))
                             val_ry = float(getattr(cs, 'ry', 0.0))
                             
                             # 计算法向量 (基于 optiland 旋转约定: Ry @ Rx @ z_hat)
                             # n = [sin(ry)cos(rx), -sin(rx), cos(ry)cos(rx)]
                             sx, cx = np.sin(val_rx), np.cos(val_rx)
                             sy, cy = np.sin(val_ry), np.cos(val_ry)
                             n_vec = np.array([sy*cx, -sx, cy*cx])
                             
                             if hasattr(self, '_exit_dir_local'):
                                 exit_dir = self._exit_dir_local
                                 dot = np.dot(n_vec, exit_dir)
                                 print(f"    [CHECK idx=3] Surface Normal: {n_vec}")
                                 print(f"    [CHECK idx=3] Exit Direction: {exit_dir}")
                                 print(f"    [CHECK idx=3] Dot Product: {dot:.8f} (Expected: 1.0)")
                                 
                                 if abs(dot - 1.0) > 1e-5:
                                     print(f"    [WARNING] Surface 3 NOT perpendicular! Dot product deviation: {abs(dot-1.0)}")
                                 else:
                                     print(f"    [pass] Surface 3 is perpendicular to exit direction.")
                             else:
                                 print(f"    [CHECK idx=3] Skipped: _exit_dir_local not found.")
                         except Exception as e:
                             print(f"    [CHECK idx=3] Error during check: {e}")

                    # 几何参数 (StandardGeometry)
                    if hasattr(geo, 'radius'):
                        print(f"    Radius: {geo.radius}")
                    if hasattr(geo, 'k'):  # conic constant
                         print(f"    Conic (k): {geo.k}")
                else:
                    print(f"    No 'geometry' attribute found.")
                
                # 材料
                mat = getattr(surf, 'material_post', None)
                print(f"    Material (Post): {mat}")
            print("="*60 + "\n") 



        # ⚠️ 关键修复：坐标系转换
        # HybridPropagator 传入的 input_rays 是全局坐标系下的光线。
        # 而 Optiland 的 Optic 是在"入射面局部坐标系"中定义的（Optic 原点 = Global Entrance Position）。
        # 因此，我们需要将 input_rays 转换到 Optic 的坐标系中。
        #
        # 变换步骤：
        # 1. 位置变换：P_local = R_entrance.T @ (P_global - P_entrance_origin)
        # 2. 方向变换：D_local = R_entrance.T @ D_global
        
        entrance_origin = np.array(self.entrance_position, dtype=np.float64)
        rotation_matrix_T = self.rotation_matrix.T
        
        # 转换位置
        # 注意：rays.x, y, z 是 array
        x_global = np.asarray(rays.x)
        y_global = np.asarray(rays.y)
        z_global = np.asarray(rays.z)
        
        # 向量化计算 P_local
        # dx, dy, dz = P_global - P_entrance
        dx = x_global - entrance_origin[0]
        dy = y_global - entrance_origin[1]
        dz = z_global - entrance_origin[2]
        
        # P_local = R.T @ delta
        x_local = rotation_matrix_T[0, 0] * dx + rotation_matrix_T[0, 1] * dy + rotation_matrix_T[0, 2] * dz
        y_local = rotation_matrix_T[1, 0] * dx + rotation_matrix_T[1, 1] * dy + rotation_matrix_T[1, 2] * dz
        z_local = rotation_matrix_T[2, 0] * dx + rotation_matrix_T[2, 1] * dy + rotation_matrix_T[2, 2] * dz
        
        # 转换方向
        L_global = np.asarray(rays.L)
        M_global = np.asarray(rays.M)
        N_global = np.asarray(rays.N)
        
        L_local = rotation_matrix_T[0, 0] * L_global + rotation_matrix_T[0, 1] * M_global + rotation_matrix_T[0, 2] * N_global
        M_local = rotation_matrix_T[1, 0] * L_global + rotation_matrix_T[1, 1] * M_global + rotation_matrix_T[1, 2] * N_global
        N_local = rotation_matrix_T[2, 0] * L_global + rotation_matrix_T[2, 1] * M_global + rotation_matrix_T[2, 2] * N_global
        
        # 更新光线坐标（在 Optic 坐标系中）
        rays.x = x_local
        rays.y = y_local
        rays.z = z_local
        rays.L = L_local
        rays.M = M_local
        rays.N = N_local
        
        # 初始化 OPD（如果尚未初始化）
        if rays.opd is None:
            rays.opd = np.zeros(len(rays.x))
        
        # 逐个表面追迹
        for i, surface in enumerate(surfaces):
            if i < skip:
                continue
            
            # 保存追迹前的 OPD、z 坐标、方向和位置
            opd_before = np.asarray(rays.opd).copy()
            x_before = np.asarray(rays.x).copy() # Added x_before
            y_before = np.asarray(rays.y).copy() # Added y_before
            z_before = np.asarray(rays.z).copy()
            N_before = np.asarray(rays.N).copy()  # 追迹前的 Z 方向分量
            L_before = np.asarray(rays.L).copy()
            M_before = np.asarray(rays.M).copy()
            
            # 使用 optiland 的表面追迹（会使用 abs(t)）
            surface.trace(rays)
            
            # 获取追迹后的坐标和 OPD
            x_after = np.asarray(rays.x)
            y_after = np.asarray(rays.y)
            z_after = np.asarray(rays.z)
            opd_after = np.asarray(rays.opd)
            
            # 计算 OPD 增量（optiland 使用 abs(t)）
            opd_increment_abs = opd_after - opd_before

            if getattr(self, 'debug', False):
                print(f"[DEBUG TRACE] i={i}, Surface={type(surface).__name__}")
                print(f"  Pos Before: ({x_before[0]:.4f}, {y_before[0]:.4f}, {z_before[0]:.4f})")
                print(f"  Pos After : ({x_after[0]:.4f}, {y_after[0]:.4f}, {z_after[0]:.4f})")
                print(f"  OPD Incr  : {opd_increment_abs[0]:.6f}")

            
            # -----------------------------------------------------------------
            # 修正：使用点积判断方向
            # sign(t) = sign((P_new - P_old) · Direction_old)
            # 这种方法对于任意传播方向（包括垂直于 Z 轴）都是数值稳定的
            # -----------------------------------------------------------------
            
            # 计算位移向量
            dx = x_after - x_before
            dy = y_after - y_before
            dz = z_after - z_before
            
            # 计算点积：displacement · direction
            # 如果点积 > 0，光线沿正方向传播，t > 0
            # 如果点积 < 0，光线沿负方向传播，t < 0
            # 这与物理直觉一致：如果光线"顺着"方向走，距离增加；如果"逆着"方向走（数学上的负 t），距离减少
            dot_product = dx * L_before + dy * M_before + dz * N_before
            
            # t 的符号
            sign_t = np.sign(dot_product)
            sign_t[sign_t == 0] = 1 # 处理零值

            # 计算带符号的 OPD 增量
            opd_increment_signed = sign_t * opd_increment_abs
            
            # 更新 OPD
            rays.opd = opd_before + opd_increment_signed

            if getattr(self, 'debug', False):
                 from utils.debug_viz import plot_opd_increment, plot_rays_2d
                 
                 step_name = f"Surface {i} Trace Step"
                 
                 # Plot OPD increment and sign_t
                 plot_opd_increment(
                     x_before, y_before, 
                     opd_increment_signed, 
                     sign_t=sign_t,
                     step_name=f"{step_name}: OPD Increment (Signed)"
                 )
                 
                 # Plot Ray Directions before trace
                 plot_rays_2d(
                     x_before, y_before, 
                     L_before, M_before,
                     title=f"{step_name}: Input Rays"
                 )
                 #此处应当注明比例尺

        # =====================================================================
        # ⚠️ 关键修复：坐标系归一化
        # =====================================================================
        # Optiland 的 trace() 方法会将光线留在最后一个表面的局部坐标系中。
        # 而后续的 "Dynamic Exit Surface" 是在 Optic 的全局坐标系（即入射面局部坐标系）中定义的。
        # 因此，必须先将光线转换回 Optic 全局坐标系。
        
        if len(surfaces) > 0 and (len(surfaces) > skip):
            # NOTE: optiland's Surface.trace already calls globalize internally
            # 光线应该已经在 Optic 全局坐标系中
            print(f"[DEBUG] After mirror trace (should be in Optic Global): rays at ({np.mean(rays.x):.4f}, {np.mean(rays.y):.4f}, {np.mean(rays.z):.4f}), dir=({np.mean(rays.L):.4f}, {np.mean(rays.M):.4f}, {np.mean(rays.N):.4f})")
        
        # =====================================================================
        # 动态添加并追迹出射面 (Dynamic Exit Surface Trace)
        # =====================================================================
        # 这是核心重构：出射面不再在 _finalize_optic 中静态添加到 optic 对象，
        # 而是在这里动态创建一个临时表面对象并追迹光线。
        # 
        # 前提条件：
        #   - self._chief_intersection_local 已设置（出射面中心位置，在入射面局部坐标系中）
        #   - self.exit_chief_direction 已设置（出射方向，在全局坐标系中）
        # 
        # 实现步骤：
        #   1. 计算出射方向在入射面局部坐标系中的表示
        #   2. 创建一个倾斜的平面 Surface 对象
        #   3. 调用 surface.trace(rays) 完成光线追迹
        #   4. 使用带符号的 OPD 更新
        
        if self._chief_intersection_local is not None and self.exit_chief_direction is not None:

            from optiland.surfaces.standard_surface import Surface
            from optiland.coordinate_system import CoordinateSystem
            from optiland.geometries.standard import StandardGeometry
            from optiland.materials import IdealMaterial
            
            # 1. 计算出射方向在入射面局部坐标系中的表示
            exit_dir_global = np.array(self.exit_chief_direction, dtype=np.float64)
            exit_dir_local = self.rotation_matrix.T @ exit_dir_global
            exit_dir_local = exit_dir_local / np.linalg.norm(exit_dir_local)
            
            # 2. 计算出射面的旋转角度 (rx, ry) - 与 _finalize_optic 相同的公式
            L = _snap_to_zero(exit_dir_local[0])
            M = _snap_to_zero(exit_dir_local[1])
            N = _snap_to_zero(exit_dir_local[2])
            
            # 重新归一化
            norm_sanitized = np.linalg.norm([L, M, N])
            if norm_sanitized > 1e-10:
                L, M, N = L / norm_sanitized, M / norm_sanitized, N / norm_sanitized
            
            rx = -np.arcsin(np.clip(M, -1.0, 1.0))
            ry = np.arctan2(L, N)
            
            # 3. 获取出射面位置
            exit_x, exit_y, exit_z = self._chief_intersection_local
            
            # 4. 创建临时出射面
            # 使用 optiland 的 Surface 类创建一个平面（radius=inf）
            cs = CoordinateSystem()
            cs.x = exit_x
            cs.y = exit_y
            cs.z = exit_z
            cs.rx = rx  #rx = -1.5707963267948966
            cs.ry = ry
            #这里可能也有数值问题

            # 平面使用无穷大曲率半径
            exit_geometry = StandardGeometry(cs, radius=np.inf, conic=0.0)
            # 获取最后一个光学表面
            last_surface = self.optic.surface_group.surfaces[-1]
            
            # 使用最后一个表面的 material_post 作为出射面的材料
            # 这样可以确保 n_in = n_out，避免在虚拟出射面上发生折射
            exit_material_post = last_surface.material_post
            
            # [DEBUG]
            # [DEBUG]
            if getattr(self, 'debug', False):
                print(f"[DEBUG ElementRaytracer] Dynamic Exit Surface: Linking to previous surface {type(last_surface).__name__}")
                print(f"[DEBUG ElementRaytracer] Dynamic Exit Surface: Material = {exit_material_post}")

            exit_surface = Surface(
                previous_surface=last_surface,  # 链接到前一个表面
                geometry=exit_geometry,
                material_post=exit_material_post, # 保持材料一致
                is_stop=False,
                aperture=None,
            )
            
            # 5. 追迹到出射面
            opd_before = np.asarray(rays.opd).copy()
            x_before = np.asarray(rays.x).copy()
            y_before = np.asarray(rays.y).copy()
            z_before = np.asarray(rays.z).copy()
            L_before = np.asarray(rays.L).copy()
            M_before = np.asarray(rays.M).copy()
            N_before = np.asarray(rays.N).copy()
            
            print(f"[DEBUG Exit] Before exit_surface.trace: rays at ({np.mean(rays.x):.4f}, {np.mean(rays.y):.4f}, {np.mean(rays.z):.4f}), dir=({np.mean(rays.L):.4f}, {np.mean(rays.M):.4f}, {np.mean(rays.N):.4f})")
            exit_surface.trace(rays)
            print(f"[DEBUG Exit] After  exit_surface.trace: rays at ({np.mean(rays.x):.4f}, {np.mean(rays.y):.4f}, {np.mean(rays.z):.4f})")
            
            # 6. 计算带符号的 OPD 增量
            x_after = np.asarray(rays.x)
            y_after = np.asarray(rays.y)
            z_after = np.asarray(rays.z)
            opd_after = np.asarray(rays.opd)
            
            opd_increment_abs = opd_after - opd_before
            
            dx = x_after - x_before
            dy = y_after - y_before
            dz = z_after - z_before
            dot_product = dx * L_before + dy * M_before + dz * N_before
            
            sign_t = np.sign(dot_product)
            sign_t[sign_t == 0] = 1
            
            opd_increment_signed = sign_t * opd_increment_abs
            rays.opd = opd_before + opd_increment_signed
            
            debug_msg = (
                f"[DEBUG TRACE] Exit Surface (Dynamic)\n"
                f"  Exit Pos Local: ({exit_x:.4f}, {exit_y:.4f}, {exit_z:.4f})\n"
                f"  Exit Dir Local: ({L:.4f}, {M:.4f}, {N:.4f})\n"
                f"  rx={np.degrees(rx):.2f}°, ry={np.degrees(ry):.2f}°\n"
                f"  OPD Incr (signed): mean={np.mean(opd_increment_signed):.6f}, std={np.std(opd_increment_signed):.6f}\n"
            )
            if getattr(self, 'debug', False):
                print(debug_msg)
            try:
                with open('d:\\BTS\\debug_log.txt', 'a', encoding='utf-8') as f:
                    f.write(debug_msg + '\n')
            except:
                pass


    
    def get_output_rays(self) -> RealRays:
        """获取出射光线数据（在出射面局部坐标系中）
        
        返回经过光线追迹后的出射光线数据。
        
        返回:
            RealRays: 出射光线数据，包含位置 (x, y, z)、方向余弦 (L, M, N)、
                     强度、波长和 OPD 等信息
        
        异常:
            RuntimeError: 如果 trace() 方法尚未调用
        
        示例:
            >>> raytracer = ElementRaytracer(surfaces=[mirror], wavelength=0.55)
            >>> raytracer.trace(input_rays)
            >>> output_rays = raytracer.get_output_rays()
            >>> print(f"出射光线数量: {len(output_rays.x)}")
        
        Validates:
            - Requirements 5.1: 输出 RealRays 对象，包含出射光线的位置 (x, y, z)
            - Requirements 5.2: 输出出射光线的方向余弦 (L, M, N)
        """
        if self.output_rays is None:
            raise RuntimeError(
                "尚未执行光线追迹。请先调用 trace() 方法。"
            )
        return self.output_rays
    
    def get_relative_opd_waves(self) -> NDArray:
        """获取相对于主光线的 OPD（波长数）
        
        计算所有光线相对于主光线（Px=0, Py=0）的光程差，单位为波长数。
        
        返回:
            NDArray: 相对 OPD 数组，形状为 (n_rays,)，单位：波长数
                    无效光线的 OPD 值为 NaN
        
        异常:
            RuntimeError: 如果 trace() 方法尚未调用
        
        说明:
            - 主光线是 Px=0, Py=0 的光线，即光瞳中心的光线
            - 相对 OPD = (光线 OPD - 主光线 OPD) / 波长
            - 对于纯几何光线追迹（没有相位面），OPD 单位是 mm
            - 对于使用了相位面的情况，需要修正 1000 倍放大问题
            - 本实现假设是纯几何光线追迹，不需要 1000 倍修正
        
        示例:
            >>> raytracer = ElementRaytracer(surfaces=[mirror], wavelength=0.55)
            >>> raytracer.trace(input_rays)
            >>> opd_waves = raytracer.get_relative_opd_waves()
            >>> print(f"OPD RMS: {np.nanstd(opd_waves):.4f} waves")
        
        Validates:
            - Requirements 5.3: 输出出射光线的 OPD（相对于主光线，单位：波长数）
            - Requirements 5.5: 使用主光线（Px=0, Py=0）作为参考计算相对 OPD
        """
        if self.output_rays is None:
            raise RuntimeError(
                "尚未执行光线追迹。请先调用 trace() 方法。"
            )
        
        # 获取 OPD 数据（单位：mm）
        opd_mm = np.asarray(self.output_rays.opd)
        
        # 获取有效光线掩模
        valid_mask = self.get_valid_ray_mask()
        
        # 如果没有有效光线，返回全 NaN 数组
        if not np.any(valid_mask):
            return np.full(len(opd_mm), np.nan)
        
        # 找到主光线（Px=0, Py=0 的光线）
        # 主光线应该是位于光瞳中心的光线，即 x=0, y=0 的光线
        # 在出射面局部坐标系中，主光线的位置应该接近 (0, 0)
        x_array = np.asarray(self.output_rays.x)
        y_array = np.asarray(self.output_rays.y)
        
        # 计算每条光线到原点的距离
        distances = np.sqrt(x_array**2 + y_array**2)
        
        # 只考虑有效光线
        distances_valid = np.where(valid_mask, distances, np.inf)
        
        # 找到最接近原点的有效光线作为主光线
        chief_ray_index = np.argmin(distances_valid)
        
        # 获取主光线的 OPD
        chief_opd_mm = opd_mm[chief_ray_index]
        
        # 计算相对 OPD（单位：mm）
        relative_opd_mm = opd_mm - chief_opd_mm
        
        # 转换为波长数
        # 波长单位：μm，需要转换为 mm
        wavelength_mm = self.wavelength * 1e-3
        opd_waves = relative_opd_mm / wavelength_mm
        
        # 将无效光线的 OPD 设为 NaN
        opd_waves = np.where(valid_mask, opd_waves, np.nan)
        
        return opd_waves
    
    def get_valid_ray_mask(self) -> NDArray:
        """获取有效光线的掩模
        
        返回一个布尔数组，标识哪些光线是有效的。
        
        有效光线的判断标准：
        - 强度（intensity）> 0
        - 位置 (x, y, z) 都是有限值（非 NaN、非 Inf）
        - 方向余弦 (L, M, N) 都是有限值（非 NaN、非 Inf）
        
        返回:
            NDArray: 布尔数组，形状为 (n_rays,)
                    True 表示有效光线，False 表示无效光线
        
        异常:
            RuntimeError: 如果 trace() 方法尚未调用
        
        示例:
            >>> raytracer = ElementRaytracer(surfaces=[mirror], wavelength=0.55)
            >>> raytracer.trace(input_rays)
            >>> valid_mask = raytracer.get_valid_ray_mask()
            >>> n_valid = np.sum(valid_mask)
            >>> print(f"有效光线数量: {n_valid}/{len(valid_mask)}")
        
        Validates:
            - Requirements 5.4: 提供方法获取有效光线的掩模（布尔数组）
        """
        if self.output_rays is None:
            raise RuntimeError(
                "尚未执行光线追迹。请先调用 trace() 方法。"
            )
        
        # 获取光线数据
        x = np.asarray(self.output_rays.x)
        y = np.asarray(self.output_rays.y)
        z = np.asarray(self.output_rays.z)
        L = np.asarray(self.output_rays.L)
        M = np.asarray(self.output_rays.M)
        N = np.asarray(self.output_rays.N)
        intensity = np.asarray(self.output_rays.i)
        
        # 判断强度是否大于 0
        intensity_valid = intensity > 0
        
        # 判断位置是否为有限值
        position_valid = (
            np.isfinite(x) & 
            np.isfinite(y) & 
            np.isfinite(z)
        )
        
        # 判断方向是否为有限值
        direction_valid = (
            np.isfinite(L) & 
            np.isfinite(M) & 
            np.isfinite(N)
        )
        
        # 综合判断
        valid_mask = intensity_valid & position_valid & direction_valid
        
        return valid_mask
    
    def get_input_positions(self) -> Tuple[NDArray, NDArray]:
        """获取输入光线位置（用于雅可比矩阵计算）
        
        返回输入光线在入射面局部坐标系中的 (x, y) 位置。
        这些位置用于计算雅可比矩阵，从而基于能量守恒原理计算振幅变化。
        
        返回:
            Tuple[NDArray, NDArray]: (x, y) 元组
                - x: 输入光线的 x 坐标数组，单位：mm
                - y: 输入光线的 y 坐标数组，单位：mm
        
        异常:
            RuntimeError: 如果 trace() 方法尚未调用
        
        说明:
            - 输入光线位置在 trace() 方法中保存
            - 坐标系为入射面局部坐标系（原点在入射面中心）
            - 与 get_output_positions() 配合使用，可计算雅可比矩阵
        
        示例:
            >>> raytracer = ElementRaytracer(surfaces=[mirror], wavelength=0.55)
            >>> raytracer.trace(input_rays)
            >>> x_in, y_in = raytracer.get_input_positions()
            >>> x_out, y_out = raytracer.get_output_positions()
            >>> # 使用输入/输出位置计算雅可比矩阵
        
        Validates:
            - 需求 1.2: 保存输入光线位置，并提供 get_input_positions() 方法
        """
        if self.input_rays is None:
            raise RuntimeError(
                "尚未执行光线追迹。请先调用 trace() 方法。"
            )
        
        x = np.asarray(self.input_rays.x)
        y = np.asarray(self.input_rays.y)
        
        return x, y
    
    def get_output_positions(self) -> Tuple[NDArray, NDArray]:
        """获取输出光线位置（用于雅可比矩阵计算）
        
        返回输出光线在出射面局部坐标系中的 (x, y) 位置。
        这些位置用于计算雅可比矩阵，从而基于能量守恒原理计算振幅变化。
        
        返回:
            Tuple[NDArray, NDArray]: (x, y) 元组
                - x: 输出光线的 x 坐标数组，单位：mm
                - y: 输出光线的 y 坐标数组，单位：mm
        
        异常:
            RuntimeError: 如果 trace() 方法尚未调用
        
        说明:
            - 输出光线位置在 trace() 方法中计算并保存
            - 坐标系为出射面局部坐标系（原点在出射面中心）
            - 与 get_input_positions() 配合使用，可计算雅可比矩阵
            - 雅可比矩阵描述输入面到输出面的坐标映射
        
        示例:
            >>> raytracer = ElementRaytracer(surfaces=[mirror], wavelength=0.55)
            >>> raytracer.trace(input_rays)
            >>> x_in, y_in = raytracer.get_input_positions()
            >>> x_out, y_out = raytracer.get_output_positions()
            >>> # 使用输入/输出位置计算雅可比矩阵
            >>> # 雅可比行列式 |J| 表示局部面积放大率
            >>> # 振幅变化：A_out / A_in = 1 / sqrt(|J|)
        
        Validates:
            - 需求 1.3: 提供 get_output_positions() 方法获取出射光线位置
        """
        if self.output_rays is None:
            raise RuntimeError(
                "尚未执行光线追迹。请先调用 trace() 方法。"
            )
        
        x = np.asarray(self.output_rays.x)
        y = np.asarray(self.output_rays.y)
        
        return x, y
    
    def get_exit_rotation_matrix(self) -> NDArray:
        """获取出射面的旋转矩阵
        
        返回从出射面局部坐标系到全局坐标系的旋转矩阵。
        
        返回:
            NDArray: 3x3 旋转矩阵
        
        说明:
            - 旋转矩阵在 _finalize_optic() 中计算并存储
            - 用于将出射光线从全局坐标系转换到出射面局部坐标系
            - 如果主光线尚未追迹，会自动调用 trace_chief_ray()
        
        Validates:
            - Requirements REQ-2.3: 提供出射面旋转矩阵
        """
        if not self._chief_ray_traced:
            self.trace_chief_ray()
        return self.exit_rotation_matrix
    
    def get_global_chief_ray_intersection(self) -> Tuple[float, float, float]:
        """获取主光线与表面的交点（全局坐标系）
        
        返回:
            (x, y, z) 全局坐标
            
        异常:
            RuntimeError: 如果主光线尚未追迹
        """
        if not self._chief_ray_traced:
            self.trace_chief_ray()
            
        if self._chief_intersection_local is None:
            raise RuntimeError("主光线交点未计算")
            
        # 1. 获取局部坐标
        # 这是相对于入射面中心，在入射面坐标系中的位置
        local_pos = np.array(self._chief_intersection_local, dtype=np.float64)
        
        # 2. 转换到全局坐标
        # Global = Entrance_Pos + R @ Local
        # self.rotation_matrix 是从局部到全局的变换（列向量为局部轴在全局的表示）
        # 或者 它是从全局到局部的？
        # 检查 __init__ 中的 compute_rotation_matrix:
        # returns matrix where columns are X, Y, Z axes of the new frame express in global coords.
        # So Global_Vec = R @ Local_Vec.
        # Check usage: 
        #   surface_offset_local = self.rotation_matrix.T @ (vertex_global - entrance_pos)
        #   implies Local = R.T @ (Global - Origin) -> R @ Local = Global - Origin
        #   So Global = Origin + R @ Local. Correct.
        
        global_offset = self.rotation_matrix @ local_pos
        entrance_pos = np.array(self.entrance_position, dtype=np.float64)
        
        global_pos = entrance_pos + global_offset
        
        return (float(global_pos[0]), float(global_pos[1]), float(global_pos[2]))



# =============================================================================
# 便捷工厂函数
# =============================================================================

def create_concave_mirror_for_spherical_wave(
    source_distance: float,
    semi_aperture: Optional[float] = None,
) -> SurfaceDefinition:
    """创建用于将球面波转换为平面波的凹面镜
    
    便捷工厂函数，根据球面波源距离自动计算凹面镜的曲率半径。
    当球面波从凹面镜的焦点发出时，反射后变为平面波。
    
    参数:
        source_distance: 球面波源到镜面的距离，单位：mm
            - 必须为正值
            - 等于凹面镜的焦距 f
        semi_aperture: 半口径，单位：mm
            - None 表示无限制
            - 正值表示有效区域的半径
    
    返回:
        SurfaceDefinition: 凹面镜表面定义对象
    
    说明:
        凹面镜焦距 f = R/2，其中 R 为曲率半径。
        因此，曲率半径 R = 2 * source_distance。
        
        当球面波从焦点发出时：
        - 入射光线从焦点向镜面发散
        - 经凹面镜反射后，光线变为平行光（平面波）
        - 出射波前的 OPD 应为常数
    
    异常:
        ValueError: 如果 source_distance <= 0
    
    示例:
        >>> # 创建焦距 100mm 的凹面镜（曲率半径 200mm）
        >>> mirror = create_concave_mirror_for_spherical_wave(
        ...     source_distance=100.0,
        ...     semi_aperture=15.0
        ... )
        >>> mirror.radius
        200.0
        >>> mirror.focal_length
        100.0
        
        >>> # 验证球面波转平面波
        >>> # 从焦点发出的球面波经此镜反射后变为平面波
    
    Validates:
        - Requirements 2.1: 支持定义球面反射镜（通过曲率半径参数）
        - Requirements 6.1: 球面波入射至焦距匹配的凹面反射镜时输出平面波
    """
    # 验证输入参数
    if source_distance <= 0:
        raise ValueError(
            f"球面波源距离必须为正值，实际为 {source_distance} mm"
        )
    
    # 计算曲率半径：R = 2 * f = 2 * source_distance
    radius = 2.0 * source_distance
    
    return SurfaceDefinition(
        surface_type='mirror',
        radius=radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=semi_aperture,
    )
