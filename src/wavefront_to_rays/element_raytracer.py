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
            - 正值表示凹面（曲率中心在 +Z 方向）
            - 负值表示凸面（曲率中心在 -Z 方向）
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
        >>> # 创建凹面反射镜（焦距 100mm，曲率半径 200mm）
        >>> mirror = SurfaceDefinition(
        ...     surface_type='mirror',
        ...     radius=200.0,
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
    
    Validates:
        - Requirements 2.1: 支持定义球面反射镜（通过曲率半径参数）
        - Requirements 2.2: 支持定义平面反射镜（曲率半径为无穷大）
        - Requirements 2.3: 支持定义球面折射面（通过曲率半径和材料参数）
        - Requirements 2.4: 正值曲率半径表示凹面镜（曲率中心在 +Z 方向）
        - Requirements 2.6: 接受表面半口径参数以限制有效区域
        - Requirements 2.1 (gaussian-beam): 支持定义抛物面反射镜（通过 conic=-1）
    """
    
    surface_type: str = 'mirror'
    radius: float = field(default_factory=lambda: np.inf)
    thickness: float = 0.0
    material: str = 'mirror'
    semi_aperture: Optional[float] = None
    conic: float = 0.0  # 圆锥常数，默认 0.0（球面）
    tilt_x: float = 0.0  # 绕 X 轴旋转角度（弧度）
    tilt_y: float = 0.0  # 绕 Y 轴旋转角度（弧度）
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
            return self.radius / 2.0
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
    # 如果主光线接近全局 Y 轴（|M| > 0.9），使用全局 X 轴作为参考
    # 否则使用全局 Y 轴作为参考
    if abs(z_local[1]) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
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
            - 正值表示凹面镜（曲率中心在 +Z 方向）
            - 负值表示凸面镜（曲率中心在 -Z 方向）
            - np.inf 表示平面镜
        semi_aperture: 半口径，单位：mm
            - None 表示无限制
            - 正值表示有效区域的半径
    
    返回:
        SurfaceDefinition: 反射镜表面定义对象
    
    示例:
        >>> # 创建凹面镜（焦距 100mm，曲率半径 200mm）
        >>> mirror = create_mirror_surface(radius=200.0, semi_aperture=15.0)
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
    ) -> None:
        """初始化元件光线追迹器
        
        参数:
            surfaces: 光学表面定义列表，至少包含一个表面
            wavelength: 波长，单位：μm，必须为正值
            chief_ray_direction: 主光线方向向量 (L, M, N)，默认 (0, 0, 1) 表示正入射
                                必须是归一化的方向余弦
            entrance_position: 入射面中心在全局坐标系中的位置，默认 (0, 0, 0)
                              单位：mm
            exit_chief_direction: 出射主光线方向向量 (L, M, N)，可选
                                 如果提供，将直接使用此方向而不是从表面倾斜计算
                                 必须是归一化的方向余弦
        
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
        
        # 如果提供了出射方向，直接完成光学系统
        if self._provided_exit_direction is not None:
            self.exit_chief_direction = self._provided_exit_direction
            self._chief_ray_traced = True
            self._finalize_optic()
    
    def trace(self, input_rays: RealRays) -> RealRays:
        """执行光线追迹
        
        将输入光线通过光学系统进行追迹，输出出射光线数据。
        
        参数:
            input_rays: 输入光线（来自 wavefront_sampler，在入射面局部坐标系中）
                       必须是 RealRays 对象
        
        返回:
            出射光线数据（在出射面局部坐标系中），RealRays 对象
        
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
        
        Validates:
            - Requirements 1.1: 验证光线数据的有效性
            - Requirements 1.2: 方向余弦不满足归一化条件时抛出 ValueError
            - Requirements 1.3: 支持任意数量的输入光线
            - Requirements 1.4: 输入光线数量为零时返回空的输出光线集合
            - Requirements 4.1: 计算每条光线与光学表面的交点
            - Requirements 4.4: 累计计算光线的 OPD
            - Requirements 4.5: 光线未能到达光学表面时标记为无效
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
        # 调用带符号 OPD 的光线追迹
        # =====================================================================
        
        # 使用带符号 OPD 追迹，正确处理折叠光路中的 OPD 计算
        # 关键区别：不使用 abs(t)，保留传播距离的符号
        # 注意：现在不追迹到出射面，只追迹到光学表面
        self._trace_with_signed_opd(traced_rays, skip=1)
        
        # =====================================================================
        # 将光线投影到出射面
        # =====================================================================
        #
        # 出射面是垂直于出射方向的平面，经过主光线交点。
        # 我们需要计算每条光线与出射面的交点，并更新 OPD。
        #
        # 出射面方程（在入射面局部坐标系中）：
        # (P - P0) · n = 0
        # 其中 P0 = 主光线交点，n = 出射方向
        #
        # 光线方程：P = P_ray + t * D_ray
        # 代入得：t = (P0 - P_ray) · n / (D_ray · n)
        
        # 获取光线数据
        x_ray = np.asarray(traced_rays.x)
        y_ray = np.asarray(traced_rays.y)
        z_ray = np.asarray(traced_rays.z)
        L_ray = np.asarray(traced_rays.L)
        M_ray = np.asarray(traced_rays.M)
        N_ray = np.asarray(traced_rays.N)
        opd_ray = np.asarray(traced_rays.opd)
        
        # 主光线交点（出射面原点）
        chief_x, chief_y, chief_z = self._chief_intersection_local
        P0 = np.array([chief_x, chief_y, chief_z])
        
        # 出射方向（出射面法向量）
        n = np.array(self._exit_dir_local)
        
        # 计算每条光线到出射面的距离 t
        # t = (P0 - P_ray) · n / (D_ray · n)
        P_ray = np.stack([x_ray, y_ray, z_ray], axis=0)  # (3, n_rays)
        D_ray = np.stack([L_ray, M_ray, N_ray], axis=0)  # (3, n_rays)
        
        # (P0 - P_ray) · n
        diff = P0.reshape(3, 1) - P_ray  # (3, n_rays)
        numerator = np.sum(diff * n.reshape(3, 1), axis=0)  # (n_rays,)
        
        # D_ray · n
        denominator = np.sum(D_ray * n.reshape(3, 1), axis=0)  # (n_rays,)
        
        # 避免除以零
        denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        
        # 计算 t
        t = numerator / denominator
        
        # 更新光线位置到出射面
        x_exit_local = x_ray + t * L_ray
        y_exit_local = y_ray + t * M_ray
        z_exit_local = z_ray + t * N_ray
        
        # 更新 OPD（带符号）
        # OPD 增量 = t（因为在空气中 n=1）
        # 符号由 t 自然决定
        opd_exit = opd_ray + t
        
        # 更新 traced_rays
        traced_rays.x = x_exit_local
        traced_rays.y = y_exit_local
        traced_rays.z = z_exit_local
        traced_rays.opd = opd_exit
        
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
        
<<<<<<< Updated upstream
        # 计算从入射面局部坐标系到出射面局部坐标系的旋转矩阵
        # R_entrance_to_exit = R_exit.T @ R_entrance
        R_entrance_to_exit = self.exit_rotation_matrix.T @ self.rotation_matrix
=======
        # =====================================================================
        # 将输出光线从入射面局部坐标系转换到出射面局部坐标系
        # =====================================================================
        # 使用确定的几何关系进行坐标转换，而不是依赖于光线分布推断。
        # 
        # 变换公式：
        # P_exit = R_exit^T @ (P_global - O_exit)
        # P_global = R_ent @ P_ent + O_ent
        # 
        # 代入得：
        # P_exit = R_exit^T @ (R_ent @ P_ent + O_ent - O_exit)
        #        = (R_exit^T @ R_ent) @ P_ent + R_exit^T @ (O_ent - O_exit)
        # 
        # 定义：
        # R_ent_to_exit = R_exit^T @ R_ent
        # shift_vec = R_exit^T @ (O_ent - O_exit)
        
        if self.exit_rotation_matrix is None:
             raise RuntimeError("exit_rotation_matrix not set via trace_chief_ray")

        # 1. 计算旋转矩阵 R_ent_to_exit
        R_ent_to_global = self.rotation_matrix
        R_global_to_exit = self.exit_rotation_matrix.T
        R_ent_to_exit = R_global_to_exit @ R_ent_to_global
        
        # 2. 准备原点坐标 (Global)
        ent_origin_global = np.array(self.entrance_position, dtype=np.float64)
        
        if self.exit_position is not None:
            exit_origin_global = np.array(self.exit_position, dtype=np.float64)
        else:
            # 如果未指定出射位置，使用主光线与表面的交点作为出射原点
            if self._chief_intersection_local is None:
                 raise RuntimeError("_chief_intersection_local not set")
            
            p_chief_local = np.array(self._chief_intersection_local)
            # Convert Local -> Global to get Exit Origin
            p_chief_global = R_ent_to_global @ p_chief_local + ent_origin_global.reshape(3)
            exit_origin_global = p_chief_global
            
        # 3. 计算位移向量 (在 Exit Frame 下)
        delta_origin_global = ent_origin_global.reshape(3) - exit_origin_global.reshape(3)
        shift_vec_exit = R_global_to_exit @ delta_origin_global
>>>>>>> Stashed changes
        
        if getattr(self, 'debug', False):
            print(f"[DEBUG trace] Coordinate Transform:")
            print(f"  Ent Origin (Global): {ent_origin_global}")
            print(f"  Exit Origin (Global): {exit_origin_global}")
            print(f"  Shift Vector (Exit Frame): {shift_vec_exit}")

        # 4. 应用变换
        
        # 获取光线数据 (Entrance Local)
        x_entrance = np.asarray(traced_rays.x)
        y_entrance = np.asarray(traced_rays.y)
        z_entrance = np.asarray(traced_rays.z)
        L_entrance = np.asarray(traced_rays.L)
        M_entrance = np.asarray(traced_rays.M)
        N_entrance = np.asarray(traced_rays.N)
        
<<<<<<< Updated upstream
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
        pos_centered = np.stack([x_centered, y_centered, z_centered], axis=0)
        pos_exit = R_entrance_to_exit @ pos_centered
        
        x_exit = pos_exit[0]
        y_exit = pos_exit[1]
        z_exit = pos_exit[2]
        
        # 方向转换
=======
        # 位置变换
        # P_exit = R_ent_to_exit @ P_ent + shift_vec
        pos_entrance = np.stack([x_entrance, y_entrance, z_entrance], axis=0) # (3, N)
        pos_exit = R_ent_to_exit @ pos_entrance # (3, N)
        
        x_exit = pos_exit[0] + shift_vec_exit[0]
        y_exit = pos_exit[1] + shift_vec_exit[1]
        z_exit = pos_exit[2] + shift_vec_exit[2]
        
        # 方向变换
        # D_exit = R_ent_to_exit @ D_ent
>>>>>>> Stashed changes
        dir_entrance = np.stack([L_entrance, M_entrance, N_entrance], axis=0)
        dir_exit = R_ent_to_exit @ dir_entrance
        
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
        
        self.output_rays = output_rays
        
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
        # 检查是否是抛物面反射镜
        first_surface = self.surfaces[0]
        is_parabola = np.isclose(first_surface.conic, -1.0, atol=1e-6)
        is_mirror = first_surface.surface_type == 'mirror'
        
        if is_parabola and is_mirror:
            # 对于抛物面反射镜，使用光学性质计算反射方向
            return self._compute_parabola_exit_direction()
        else:
            # 对于其他表面，使用 optiland 追迹
            return self._compute_exit_direction_optiland()
    
    def _compute_parabola_exit_direction(self) -> Tuple[float, float, float]:
        """计算抛物面反射镜的出射主光线方向
        
        使用抛物面的光学性质：平行于光轴的光线反射后通过焦点。
        
        返回:
            出射主光线方向 (L, M, N)，全局坐标系，归一化
            
        坐标系说明:
            - 入射面原点在全局 entrance_position
            - 抛物面顶点在全局坐标系中是 vertex_position
            - 焦点在抛物面顶点沿 +Z 方向 focal_length 处
        """
        first_surface = self.surfaces[0]
        radius = first_surface.radius
        focal_length = radius / 2.0
        
        # 入射面中心位置（全局坐标系）
        entrance_pos = np.array(self.entrance_position, dtype=np.float64)
        
        # 抛物面顶点位置（全局坐标系）
        # ⚠️ 关键修复：使用 vertex_position 而不是假设在原点
        if first_surface.vertex_position is not None:
            vertex = np.array(first_surface.vertex_position, dtype=np.float64)
        else:
            vertex = np.array([0.0, 0.0, 0.0])
        
        # 焦点位置（全局坐标系）
        # 焦点在顶点沿光轴方向 f 处
        focus = vertex + np.array([0.0, 0.0, focal_length])
        
        # 主光线从全局原点 (0, 0, 0) 沿 +Z 方向出发
        ray_origin = np.array([0.0, 0.0, 0.0])
        ray_dir = np.array([0.0, 0.0, 1.0])
        
        # 计算主光线与抛物面的交点
        # 抛物面方程（以顶点为原点）：z - vertex[2] = ((x - vertex[0])² + (y - vertex[1])²) / (2R)
        # 主光线参数方程：(0, 0, t)
        # 代入抛物面方程：
        #   t - vertex[2] = (vertex[0]² + vertex[1]²) / (2R)
        #   t = vertex[2] + (vertex[0]² + vertex[1]²) / (2R)
        
        t_intersection = vertex[2] + (vertex[0]**2 + vertex[1]**2) / (2 * radius)
        
        # 交点位置（全局坐标系）
        intersection = ray_origin + t_intersection * ray_dir
        
        # 反射方向：指向焦点
        to_focus = focus - intersection
        exit_dir = to_focus / np.linalg.norm(to_focus)
        
        return (float(exit_dir[0]), float(exit_dir[1]), float(exit_dir[2]))
    
    def _compute_parabola_chief_intersection(self) -> Tuple[float, float, float]:
        """计算抛物面反射镜的主光线交点位置（在入射面局部坐标系中）
        
        使用解析计算而不是 optiland 追迹，因为 optiland 的表面设置
        对于离轴抛物面会返回错误的交点位置。
        
        返回:
            主光线交点位置 (x, y, z)，入射面局部坐标系
            
        坐标系说明:
            - 入射面局部坐标系原点在全局 entrance_position
            - 主光线从全局原点 (0, 0, 0) 沿 +Z 方向出发
            - 抛物面顶点在全局坐标系中是 vertex_position
        """
        first_surface = self.surfaces[0]
        radius = first_surface.radius
        
        # 入射面中心位置（全局坐标系）
        entrance_pos = np.array(self.entrance_position, dtype=np.float64)
        
        # 抛物面顶点位置（全局坐标系）
        # ⚠️ 关键修复：使用 vertex_position 而不是假设在原点
        if first_surface.vertex_position is not None:
            vertex = np.array(first_surface.vertex_position, dtype=np.float64)
        else:
            vertex = np.array([0.0, 0.0, 0.0])
        
        # 主光线从全局原点 (0, 0, 0) 沿 +Z 方向出发
        ray_origin = np.array([0.0, 0.0, 0.0])
        ray_dir = np.array([0.0, 0.0, 1.0])
        
        # 计算主光线与抛物面的交点
        # 抛物面方程（以顶点为原点）：z - vertex[2] = ((x - vertex[0])² + (y - vertex[1])²) / (2R)
        # 主光线参数方程：(0, 0, t)
        # 代入抛物面方程：
        #   t - vertex[2] = (vertex[0]² + vertex[1]²) / (2R)
        #   t = vertex[2] + (vertex[0]² + vertex[1]²) / (2R)
        
        t_intersection = vertex[2] + (vertex[0]**2 + vertex[1]**2) / (2 * radius)
        
        # 交点位置（全局坐标系）
        intersection_global = ray_origin + t_intersection * ray_dir
        
        # 将交点从全局坐标系转换到入射面局部坐标系
        # pos_local = R.T @ (pos_global - entrance_pos)
        intersection_local = self.rotation_matrix.T @ (intersection_global - entrance_pos)
        
        return (float(intersection_local[0]), 
                float(intersection_local[1]), 
                float(intersection_local[2]))
        
        # 抛物面顶点在全局原点 (0, 0, 0)
    
    def _compute_exit_direction_optiland(self) -> Tuple[float, float, float]:
        """使用 optiland 计算出射主光线方向
        
        返回:
            出射主光线方向 (L, M, N)，全局坐标系，归一化
        """
        from optiland.optic import Optic
        from optiland.raytrace import RealRayTracer
        
        # 创建临时光学系统
        temp_optic = Optic()
        
        # 设置系统参数
        first_surface = self.surfaces[0]
        if first_surface.semi_aperture is not None:
            aperture_diameter = 2.0 * first_surface.semi_aperture
        else:
            aperture_diameter = 10.0
        
        temp_optic.set_aperture(aperture_type='EPD', value=aperture_diameter)
        temp_optic.set_field_type(field_type='angle')
        temp_optic.add_field(y=0, x=0)
        temp_optic.add_wavelength(value=self.wavelength, is_primary=True)
        
        # 计算表面位置偏移
        entrance_pos = np.array(self.entrance_position, dtype=np.float64)
        surface_offset_local = self.rotation_matrix.T @ (-entrance_pos)
        
        surface_x_offset = float(surface_offset_local[0])
        surface_y_offset = float(surface_offset_local[1])
        surface_z_offset = float(surface_offset_local[2])
        
        # 添加物面
        temp_optic.add_surface(index=0, thickness=np.inf)
        
        # 添加光学表面
        for i, surface_def in enumerate(self.surfaces):
            surface_index = i + 1
            is_stop = (i == 0)
            
            if surface_def.surface_type == 'mirror':
                material = 'mirror'
            else:
                material = surface_def.material
            
            tilt_x_rad = surface_def.tilt_x if surface_def.tilt_x else 0.0
            tilt_y_rad = surface_def.tilt_y if surface_def.tilt_y else 0.0
            tilt_x_safe = _avoid_exact_45_degrees(tilt_x_rad)
            tilt_y_safe = _avoid_exact_45_degrees(tilt_y_rad)
            
            temp_optic.add_surface(
                index=surface_index,
                radius=surface_def.radius,
                material=material,
                is_stop=is_stop,
                conic=surface_def.conic,
                rx=tilt_x_safe,
                ry=tilt_y_safe,
                x=surface_x_offset,
                y=surface_y_offset,
                z=surface_z_offset,
            )
        
        # 添加像面
        image_surface_index = len(self.surfaces) + 1
        temp_optic.add_surface(
            index=image_surface_index, 
            radius=np.inf,
            x=surface_x_offset,
            y=surface_y_offset,
            z=surface_z_offset + 100.0,
        )
        
        # 追迹主光线
        tracer = RealRayTracer(temp_optic)
        chief_rays = tracer.trace_generic(
            Hx=0.0, Hy=0.0, Px=0.0, Py=0.0, 
            wavelength=self.wavelength
        )
        
        # 提取出射方向
        L = float(np.asarray(chief_rays.L)[0])
        M = float(np.asarray(chief_rays.M)[0])
        N = float(np.asarray(chief_rays.N)[0])
        
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
        
        # 如果提供了出射方向，直接使用
        if self._provided_exit_direction is not None:
            self.exit_chief_direction = self._provided_exit_direction
        else:
            # 使用 optiland 追迹主光线
            self.exit_chief_direction = self._compute_exit_chief_direction()
        
        # 标记主光线已追迹
        self._chief_ray_traced = True
        
        # 完成光学系统的创建（添加出射面）
        self._finalize_optic()
        
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
            aperture_diameter = 10.0
        
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
        
        optic.add_surface(index=0, thickness=np.inf)
        
        # =====================================================================
        # 添加光学表面
        # =====================================================================
        
        for i, surface_def in enumerate(self.surfaces):
            surface_index = i + 1
            is_stop = (i == 0)
            
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
            is_parabola = np.isclose(surface_def.conic, -1.0, atol=1e-6)
            
            if is_parabola:
                # 抛物面：不设置倾斜角度
                # 检查原本是否有倾斜
                orig_tilt_x = surface_def.tilt_x if surface_def.tilt_x else 0.0
                orig_tilt_y = surface_def.tilt_y if surface_def.tilt_y else 0.0
                if abs(orig_tilt_x) > 1e-6 or abs(orig_tilt_y) > 1e-6:
                    print(f"Warning: 检测到抛物面 (Surface {surface_index}) 具有非零倾斜角 "
                          f"(tilt_x={np.degrees(orig_tilt_x):.2f}°, tilt_y={np.degrees(orig_tilt_y):.2f}°)。"
                          f"对于 OAP，倾斜通常通过顶点偏移定义。为了确保计算正确性，"
                          f"optiland 中的倾斜参数将被强制置为 0。")
                
<<<<<<< Updated upstream
                tilt_x_safe = 0.0
                tilt_y_safe = 0.0
=======
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
                
                if getattr(self, 'debug', False):
                     print(f"[DEBUG ElementRaytracer] Surface {surface_index} Orientation:")
                     print(f"  Normal: ({L:.4f}, {M:.4f}, {N:.4f})")
                     print(f"  Calculated Tilt: rx={np.degrees(tilt_x_rad):.2f}°, ry={np.degrees(tilt_y_rad):.2f}°")
>>>>>>> Stashed changes
            else:
                # 其他表面：使用 SurfaceDefinition 中的倾斜角度
                tilt_x_rad = surface_def.tilt_x if surface_def.tilt_x else 0.0
                tilt_y_rad = surface_def.tilt_y if surface_def.tilt_y else 0.0
                tilt_x_safe = _avoid_exact_45_degrees(tilt_x_rad)
                tilt_y_safe = _avoid_exact_45_degrees(tilt_y_rad)
            
            # 使用绝对坐标定位模式，设置表面位置偏移
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
            
            if surface_def.semi_aperture is not None:
                surface = optic.surface_group.surfaces[surface_index]
                surface.max_aperture = surface_def.semi_aperture
        
        # 存储创建的光学系统（尚未添加出射面）
        self.optic = optic
    
    def _finalize_optic(self) -> None:
        """完成光学系统的创建（添加出射面）
        
        在主光线追迹完成后调用，添加透明平面作为出射面。
        
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
        #
        # 对于抛物面，使用解析计算而不是 optiland 追迹，因为 optiland 的
        # 表面设置（带有偏移和倾斜）对于离轴抛物面会返回错误的交点位置。
        #
        # 对于其他表面类型，仍然使用 optiland 追迹。
        
        first_surface = self.surfaces[0]
        is_parabola = np.isclose(first_surface.conic, -1.0, atol=1e-6)
        is_mirror = first_surface.surface_type == 'mirror'
        
        if is_parabola and is_mirror:
            # 对于抛物面反射镜，使用解析计算主光线交点
            chief_x, chief_y, chief_z = self._compute_parabola_chief_intersection()
        else:
            # 对于其他表面，使用 optiland 追迹
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
        # 
        # 注意：当出射方向 Z 分量为负时（如 OAP 反射），ry 会接近 180°，
        # 这会导致 optiland 的表面追迹产生数值问题。
        # 
        # 解决方案：不在 optiland 中添加出射面，而是在 trace() 方法中
        # 手动计算光线到出射面的投影。
        # 
        # 这里只计算并存储出射面的参数，不添加到 optiland。
        
        # 存储出射方向（入射面局部坐标系）
        self._exit_dir_local = exit_dir_local
        
        # 不再添加出射面到 optiland
        # 出射面的处理将在 trace() 方法中通过数学变换完成
        
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
        
        抛物面反射方向修正：
        optiland 使用几何法向量计算反射，但对于离轴抛物面这是不正确的。
        正确的反射方向应该使用抛物面的光学性质：
        - 对于平行于光轴的入射光线，反射后通过焦点
        - 正确的法向量公式：n = (u + d) / |u + d|
          其中 u = (P - F) / |P - F| 是从焦点到交点的单位向量，d 是入射方向
        
        参数:
            rays: 输入光线（会被原地修改）
            skip: 跳过的表面数量（默认 1，跳过物面）
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
            
            # 保存追迹前的 OPD、z 坐标、方向和位置
            opd_before = np.asarray(rays.opd).copy()
            z_before = np.asarray(rays.z).copy()
            N_before = np.asarray(rays.N).copy()  # 追迹前的 Z 方向分量
            L_before = np.asarray(rays.L).copy()
            M_before = np.asarray(rays.M).copy()
            
            # 使用 optiland 的表面追迹（会使用 abs(t)）
            surface.trace(rays)
            
            # 获取追迹后的 z 坐标和 OPD
            z_after = np.asarray(rays.z)
            opd_after = np.asarray(rays.opd)
            
            # =====================================================================
            # 抛物面反射方向修正
            # =====================================================================
            # 检查是否是抛物面反射镜
            is_parabola = False
            is_reflective = False
            
            # 检查表面类型
            if hasattr(surface, 'geometry') and hasattr(surface.geometry, 'k'):
                conic = surface.geometry.k
                is_parabola = np.isclose(conic, -1.0, atol=1e-6)
            
            if hasattr(surface, 'interaction_model') and hasattr(surface.interaction_model, 'is_reflective'):
                is_reflective = surface.interaction_model.is_reflective
            
            if is_parabola and is_reflective:
                # 对于抛物面反射镜，使用正确的反射方向公式
                self._correct_parabola_reflection(
                    rays, surface, L_before, M_before, N_before
                )
            
            # 计算 OPD 增量（optiland 使用 abs(t)）
            opd_increment_abs = opd_after - opd_before
            
            # 计算 dz 来确定 t 的符号
            dz = z_after - z_before
            
            # 正确的符号计算：sign(t) = sign(dz) * sign(N_before)
            # 因为 dz = t * N_before，所以 t = dz / N_before
            # sign(t) = sign(dz / N_before) = sign(dz) * sign(N_before)
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
    
    def _correct_parabola_reflection(
        self,
        rays: RealRays,
        surface,
        L_before: NDArray,
        M_before: NDArray,
        N_before: NDArray,
    ) -> None:
        """修正抛物面的反射方向
        
        optiland 使用几何法向量计算反射，但对于离轴抛物面这是不正确的。
        正确的反射方向应该使用抛物面的光学性质：平行于光轴的光线反射后通过焦点。
        
        对于抛物面，反射方向直接指向焦点：
        r = (F - P) / |F - P|
        
        参数:
            rays: 光线对象（会被原地修改方向）
            surface: optiland 表面对象
            L_before, M_before, N_before: 入射方向（未使用，保留接口兼容性）
        """
        # 获取光线位置（在入射面局部坐标系中，已经在表面上）
        x = np.asarray(rays.x)
        y = np.asarray(rays.y)
        z = np.asarray(rays.z)
        
        # 获取抛物面参数
        radius = surface.geometry.radius  # 曲率半径 R = 2f
        focal_length = radius / 2.0  # 焦距 f = R/2
        
        # =====================================================================
        # 焦点位置计算（在入射面局部坐标系中）
        # =====================================================================
        # 
        # ⚠️ 关键修复：焦点位置在全局坐标系中是 vertex + (0, 0, f)
        # 
        # 对于离轴抛物面：
        # - 抛物面顶点在全局 vertex_position（如 (0, 100, 0)）
        # - 焦点在全局 vertex_position + (0, 0, f)（如 (0, 100, 1000)）
        # - 需要将全局焦点位置转换到入射面局部坐标系
        # =====================================================================
        
        # 获取抛物面顶点位置（全局坐标系）
        first_surface = self.surfaces[0]
        if first_surface.vertex_position is not None:
            vertex_global = np.array(first_surface.vertex_position, dtype=np.float64)
        else:
            vertex_global = np.array([0.0, 0.0, 0.0])
        
        # 焦点在全局坐标系中的位置：顶点 + (0, 0, f)
        focus_global = vertex_global + np.array([0.0, 0.0, focal_length])
        
        # 入射面位置（全局坐标系）
        entrance_pos = np.array(self.entrance_position, dtype=np.float64)
        
        # 将焦点从全局坐标系转换到入射面局部坐标系
        # pos_local = R.T @ (pos_global - entrance_pos)
        focus_entrance = self.rotation_matrix.T @ (focus_global - entrance_pos)
        
        # 计算每条光线的正确反射方向
        # 对于抛物面，平行于光轴的光线反射后直接指向焦点
        n_rays = len(x)
        L_new = np.zeros(n_rays)
        M_new = np.zeros(n_rays)
        N_new = np.zeros(n_rays)
        
        for j in range(n_rays):
            # 光线位置（入射面局部坐标系）
            P = np.array([x[j], y[j], z[j]])
            
            # 从交点到焦点的向量
            P_to_F = focus_entrance - P
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
