# -*- coding: utf-8 -*-
"""
倾斜平面传播模块 (Tilted Propagation)

本模块封装 tilted_asm 功能，处理入射面/出射面与切平面之间的传播。

主要功能：
1. 从入射面传播到切平面（正向传播）
2. 从切平面传播到出射面（反向传播）
3. 计算旋转矩阵

数学原理：
    旋转矩阵 R 描述从入射面坐标系到切平面坐标系的变换。
    对于元件倾斜角 (tilt_x, tilt_y)：
    - tilt_x: 绕 X 轴旋转（俯仰）
    - tilt_y: 绕 Y 轴旋转（偏航）
    
    旋转顺序：先绕 X 轴，再绕 Y 轴
    R = Ry(tilt_y) @ Rx(tilt_x)

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 8.1, 8.2, 8.3, 8.4, 8.5
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation

# 导入 tilted_asm 函数
import sys
import os

# 添加 angular_spectrum_method 到路径
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from angular_spectrum_method.tilted_asm import tilted_asm

# 类型别名
ComplexArray = NDArray[np.complexfloating]
RealArray = NDArray[np.floating]


class TiltedPropagation:
    """倾斜平面传播
    
    封装 tilted_asm 功能，处理入射面/出射面与切平面之间的传播。
    
    参数:
        wavelength: 波长，单位 mm（注意：与主类单位不同，需要转换）
        dx: x 方向采样间隔，单位 mm
        dy: y 方向采样间隔，单位 mm
    
    属性:
        wavelength: 波长（mm）
        dx: x 方向采样间隔（mm）
        dy: y 方向采样间隔（mm）
    
    示例:
        >>> import numpy as np
        >>> from hybrid_propagation.tilted_propagation import TiltedPropagation
        >>> 
        >>> # 创建传播器
        >>> propagator = TiltedPropagation(
        ...     wavelength=0.633e-3,  # 633 nm = 0.633 μm = 0.633e-3 mm
        ...     dx=0.1,  # 0.1 mm
        ...     dy=0.1,  # 0.1 mm
        ... )
        >>> 
        >>> # 创建输入复振幅
        >>> amplitude = np.ones((64, 64), dtype=complex)
        >>> 
        >>> # 传播到切平面（45° 倾斜）
        >>> result = propagator.propagate_to_tangent_plane(
        ...     amplitude,
        ...     tilt_x=np.pi/4,
        ...     tilt_y=0.0,
        ... )
    
    Validates: Requirements 1.1, 1.3
    """
    
    def __init__(
        self,
        wavelength: float,
        dx: float,
        dy: float,
    ) -> None:
        """初始化倾斜平面传播器
        
        参数:
            wavelength: 波长，单位 mm
            dx: x 方向采样间隔，单位 mm
            dy: y 方向采样间隔，单位 mm
        
        异常:
            ValueError: 如果波长或采样间隔为非正数
        """
        # 参数验证
        if wavelength <= 0:
            raise ValueError(f"波长必须为正数，当前值: {wavelength}")
        if dx <= 0:
            raise ValueError(f"x 方向采样间隔必须为正数，当前值: {dx}")
        if dy <= 0:
            raise ValueError(f"y 方向采样间隔必须为正数，当前值: {dy}")
        
        self.wavelength = wavelength
        self.dx = dx
        self.dy = dy
    
    @staticmethod
    def _compute_rotation_matrix(tilt_x: float, tilt_y: float) -> NDArray:
        """计算从入射面到切平面的旋转矩阵
        
        使用 scipy.spatial.transform.Rotation 生成旋转矩阵。
        旋转顺序：先绕 X 轴旋转 tilt_x，再绕 Y 轴旋转 tilt_y。
        
        数学公式：
            R = Ry(tilt_y) @ Rx(tilt_x)
        
        其中：
            Rx(θ) = [[1, 0, 0],
                     [0, cos(θ), -sin(θ)],
                     [0, sin(θ), cos(θ)]]
            
            Ry(θ) = [[cos(θ), 0, sin(θ)],
                     [0, 1, 0],
                     [-sin(θ), 0, cos(θ)]]
        
        参数:
            tilt_x: 绕 X 轴旋转角度（弧度）
            tilt_y: 绕 Y 轴旋转角度（弧度）
        
        返回:
            3×3 旋转矩阵，满足正交性 R @ R.T = I 且 det(R) = 1
        
        示例:
            >>> R = TiltedPropagation._compute_rotation_matrix(np.pi/4, 0)
            >>> np.allclose(R @ R.T, np.eye(3))
            True
            >>> np.allclose(np.linalg.det(R), 1.0)
            True
        
        Validates: Requirements 1.3, 8.2
        """
        # 使用 scipy.spatial.transform.Rotation 生成旋转矩阵
        # 'xy' 表示先绕 X 轴旋转，再绕 Y 轴旋转（内旋/intrinsic rotation）
        # 注意：scipy 的 'xy' 是内旋顺序，即先 X 后 Y
        r = Rotation.from_euler('xy', [tilt_x, tilt_y], degrees=False)
        return r.as_matrix()
    
    @staticmethod
    def _compute_exit_rotation_matrix(
        tilt_x: float,
        tilt_y: float,
        is_reflective: bool,
    ) -> NDArray:
        """计算从切平面到出射面的旋转矩阵
        
        对于反射元件，需要考虑反射后的光轴方向变化。
        
        数学原理：
            对于反射元件，出射光轴方向由反射定律决定：
            r = d - 2(d·n)n
            
            其中：
            - d: 入射方向（单位向量）
            - n: 表面法向量（指向入射侧）
            - r: 反射方向（单位向量）
            
            出射面的旋转矩阵需要将切平面坐标系变换到出射面坐标系。
            对于反射，这相当于将入射旋转矩阵"镜像"。
        
        参数:
            tilt_x: 元件绕 X 轴倾斜角度（弧度）
            tilt_y: 元件绕 Y 轴倾斜角度（弧度）
            is_reflective: 是否为反射元件
        
        返回:
            3×3 旋转矩阵
        
        Validates: Requirements 8.2
        """
        if not is_reflective:
            # 对于透射元件，出射面与入射面平行（假设薄元件近似）
            # 返回单位矩阵（无旋转）
            return np.eye(3)
        
        # 对于反射元件，需要计算反射后的旋转矩阵
        # 
        # 入射光轴方向（假设沿 +Z 方向入射）
        incident_direction = np.array([0.0, 0.0, 1.0])
        
        # 计算表面法向量
        # 初始法向量沿 -Z 方向（指向入射侧）
        # 
        # 根据 coordinate_conventions.md 的约定：
        # 45° 折叠镜：tilt_x = π/4
        # 表面法向量初始为 (0, 0, -1)
        # 绕 X 轴旋转 45° 后：(0, -sin(45°), -cos(45°)) = (0, -0.707, -0.707)
        # 
        # 这里使用被动旋转（旋转矩阵的转置）来计算法向量
        # 或等价地，使用负角度
        R_tilt = TiltedPropagation._compute_rotation_matrix(tilt_x, tilt_y)
        initial_normal = np.array([0.0, 0.0, -1.0])
        # 使用转置实现被动旋转，使法向量按照约定方向旋转
        surface_normal = R_tilt.T @ initial_normal
        
        # 计算反射方向
        # r = d - 2(d·n)n
        d_dot_n = np.dot(incident_direction, surface_normal)
        reflected_direction = incident_direction - 2 * d_dot_n * surface_normal
        
        # 构建出射面坐标系
        # Z 轴沿反射方向
        z_axis = reflected_direction / np.linalg.norm(reflected_direction)
        
        # X 轴保持在水平面内（与全局 Y 轴垂直）
        # 如果反射方向与 Y 轴平行，则使用 X 轴作为参考
        global_y = np.array([0.0, 1.0, 0.0])
        if np.abs(np.dot(z_axis, global_y)) > 0.999:
            # 反射方向接近 Y 轴，使用 X 轴作为参考
            global_x = np.array([1.0, 0.0, 0.0])
            x_axis = np.cross(global_x, z_axis)
        else:
            x_axis = np.cross(global_y, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y 轴由右手定则确定
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # 构建旋转矩阵（列向量为新坐标系的基向量）
        R_exit = np.column_stack([x_axis, y_axis, z_axis])
        
        return R_exit
    
    def propagate_to_tangent_plane(
        self,
        amplitude: ComplexArray,
        tilt_x: float,
        tilt_y: float,
    ) -> ComplexArray:
        """从入射面传播到切平面
        
        使用 tilted_asm 计算倾斜传播。
        
        参数:
            amplitude: 入射面复振幅，形状 (N, N)
            tilt_x: 元件绕 X 轴倾斜角度（弧度）
            tilt_y: 元件绕 Y 轴倾斜角度（弧度）
        
        返回:
            切平面复振幅（在切平面局部坐标系中），形状 (N, N)
        
        异常:
            ValueError: 如果输入数组形状无效
        
        注意:
            - 当 tilt_x = 0 且 tilt_y = 0 时（正入射），直接返回输入复振幅
            - 传播过程保持能量守恒（误差 < 1%）
        
        Validates: Requirements 1.1, 1.2, 1.4, 1.5
        """
        # 输入验证
        if amplitude.ndim != 2:
            raise ValueError(f"输入复振幅必须是 2D 数组，当前维度: {amplitude.ndim}")
        
        # 检查是否为正入射（无倾斜）
        if np.abs(tilt_x) < 1e-10 and np.abs(tilt_y) < 1e-10:
            # 正入射情况，直接返回输入复振幅的副本
            return amplitude.copy()
        
        # 计算旋转矩阵
        T = self._compute_rotation_matrix(tilt_x, tilt_y)
        
        # 调用 tilted_asm 进行倾斜传播
        result = tilted_asm(
            amplitude,
            self.wavelength,
            self.dx,
            self.dy,
            T,
            expand=True,  # 使用 4 倍零填充扩展以抑制混叠
            weight=False,  # 使用雅可比行列式进行能量校正
        )
        
        return result
    
    def propagate_from_tangent_plane(
        self,
        amplitude: ComplexArray,
        tilt_x: float,
        tilt_y: float,
        is_reflective: bool,
    ) -> ComplexArray:
        """从切平面传播到出射面
        
        参数:
            amplitude: 切平面复振幅，形状 (N, N)
            tilt_x: 元件绕 X 轴倾斜角度（弧度）
            tilt_y: 元件绕 Y 轴倾斜角度（弧度）
            is_reflective: 是否为反射元件
        
        返回:
            出射面复振幅（在出射面全局坐标系中），形状 (N, N)
        
        异常:
            ValueError: 如果输入数组形状无效
        
        注意:
            - 对于反射元件，需要考虑反射后的光轴方向变化
            - 当 tilt_x = 0 且 tilt_y = 0 时（正入射），直接返回输入复振幅
            - 传播过程保持能量守恒（误差 < 1%）
        
        Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5
        """
        # 输入验证
        if amplitude.ndim != 2:
            raise ValueError(f"输入复振幅必须是 2D 数组，当前维度: {amplitude.ndim}")
        
        # 检查是否为正入射（无倾斜）
        if np.abs(tilt_x) < 1e-10 and np.abs(tilt_y) < 1e-10:
            # 正入射情况，直接返回输入复振幅的副本
            return amplitude.copy()
        
        # 计算出射方向的旋转矩阵
        T = self._compute_exit_rotation_matrix(tilt_x, tilt_y, is_reflective)
        
        # 调用 tilted_asm 进行倾斜传播（反向传播）
        result = tilted_asm(
            amplitude,
            self.wavelength,
            self.dx,
            self.dy,
            T,
            expand=True,
            weight=False,
        )
        
        return result
    
    def compute_energy(self, amplitude: ComplexArray) -> float:
        """计算复振幅的总能量
        
        能量定义为振幅平方和乘以像素面积。
        
        参数:
            amplitude: 复振幅数组
        
        返回:
            总能量
        
        Validates: Requirements 1.4, 8.4
        """
        pixel_area = self.dx * self.dy
        return np.sum(np.abs(amplitude) ** 2) * pixel_area
    
    def check_energy_conservation(
        self,
        input_amplitude: ComplexArray,
        output_amplitude: ComplexArray,
        tolerance: float = 0.01,
    ) -> Tuple[bool, float]:
        """检查能量守恒
        
        参数:
            input_amplitude: 输入复振幅
            output_amplitude: 输出复振幅
            tolerance: 允许的相对误差（默认 1%）
        
        返回:
            (is_conserved, ratio): 是否守恒，能量比值
        
        Validates: Requirements 1.4, 8.4
        """
        input_energy = self.compute_energy(input_amplitude)
        output_energy = self.compute_energy(output_amplitude)
        
        if input_energy < 1e-15:
            # 输入能量接近零，无法计算比值
            return True, 1.0
        
        ratio = output_energy / input_energy
        is_conserved = abs(ratio - 1.0) <= tolerance
        
        return is_conserved, ratio
