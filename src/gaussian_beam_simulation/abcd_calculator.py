"""
ABCD 矩阵计算模块

本模块提供高斯光束传输的 ABCD 矩阵法计算功能，
用于验证混合仿真结果的正确性，以及计算 Pilot Beam 参考相位。

设计理念：Zemax 序列模式
- 元件按顺序排列，使用 thickness 定义间距
- 光束沿光路自动传播
- 使用 propagate_distance() 方法沿光路传播
- 使用 get_beam_at_element() 获取每个元件入射面和出射面的光束参数

ABCD 矩阵法：
高斯光束的复参数 q = z - z0 + i*zR
经过 ABCD 矩阵变换后：q' = (A*q + B) / (C*q + D)

常用 ABCD 矩阵：
- 自由空间传播距离 d：[[1, d], [0, 1]]
- 薄透镜焦距 f：[[1, 0], [-1/f, 1]]
- 球面镜焦距 f：[[1, 0], [-1/f, 1]]（反射后需要考虑方向）
- 平面镜：[[1, 0], [0, 1]]

核心功能：
1. 沿光路传播到任意距离
2. 获取每个元件入射面和出射面的光束参数
3. 计算参考相位（用于 Pilot Beam 方法）

作者：混合光学仿真项目
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
from numpy.typing import NDArray

from .gaussian_beam import GaussianBeam
from .optical_elements import (
    OpticalElement, 
    ParabolicMirror, 
    SphericalMirror, 
    ThinLens,
    FlatMirror,
)


@dataclass
class ABCDResult:
    """ABCD 计算结果
    
    属性:
        z: 位置（mm）
        q: 复光束参数
        w: 光束半径（mm）
        R: 波前曲率半径（mm）
        gouy_phase: Gouy 相位（rad）
        waist_position: 束腰位置（mm）
        waist_radius: 束腰半径（mm）
        path_length: 从起点的光程距离（mm）
    """
    z: float
    q: complex
    w: float
    R: float
    gouy_phase: float
    waist_position: float
    waist_radius: float
    path_length: float = 0.0


class ABCDCalculator:
    """ABCD 矩阵计算器（Zemax 序列模式）
    
    使用 ABCD 矩阵法计算高斯光束的传输。
    
    设计理念：
    - 元件按顺序排列，使用 thickness 定义到下一元件的间距
    - 光束沿光路自动传播
    - 使用 propagate_distance() 方法沿光路传播指定距离
    
    参数:
        beam: 高斯光束对象
        elements: 光学元件列表（按顺序排列）
        initial_distance: 从光束初始面到第一个元件的距离（mm）
    
    示例:
        >>> beam = GaussianBeam(wavelength=0.633, w0=1.0, z0=50.0, z_init=0.0)
        >>> # 反射镜在距离初始面 200mm 处，反射后到像面 100mm
        >>> mirror = ParabolicMirror(thickness=100.0, semi_aperture=15.0, 
        ...                          parent_focal_length=100.0)
        >>> calc = ABCDCalculator(beam, [mirror], initial_distance=200.0)
        >>> result = calc.propagate_distance(250.0)  # 传播 250mm
    """
    
    def __init__(
        self,
        beam: GaussianBeam,
        elements: List[OpticalElement],
        initial_distance: float = 0.0,
    ) -> None:
        self.beam = beam
        self.elements = elements  # 保持用户指定的顺序
        self.initial_distance = initial_distance
        
        # 计算每个元件的位置和光程距离
        self._compute_element_positions()
        
        # 计算初始复光束参数
        self.q_init = self._compute_q(beam.z_init)
    
    def _compute_element_positions(self) -> None:
        """计算每个元件的 z 坐标位置和光程距离
        
        根据 Zemax 序列模式：
        - 第一个元件位置 = z_init + initial_distance
        - 后续元件位置根据前一个元件的 thickness 和传播方向计算
        """
        current_z = self.beam.z_init
        current_path = 0.0
        direction = 1  # 1 = +Z, -1 = -Z
        
        # 传播到第一个元件
        current_z += direction * self.initial_distance
        current_path += self.initial_distance
        
        for element in self.elements:
            # 设置元件位置
            element.z_position = current_z
            element.path_length = current_path
            
            # 如果是反射元件，方向反转
            if element.is_reflective:
                direction = -direction
            
            # 传播到下一个元件
            current_z += direction * element.thickness
            current_path += element.thickness
    
    def _compute_q(self, z: float) -> complex:
        """计算位置 z 处的复光束参数
        
        q = (z - z0) + i * zR
        """
        return complex(z - self.beam.z0, self.beam.zR)
    
    def _q_to_beam_params(self, q: complex, z: float, path_length: float, direction: int = 1) -> ABCDResult:
        """从复光束参数计算光束参数"""
        wavelength_mm = self.beam.wavelength * 1e-3
        
        inv_q = 1.0 / q
        real_part = inv_q.real
        imag_part = inv_q.imag
        
        if abs(real_part) < 1e-10:
            R = np.inf
        else:
            R = 1.0 / real_part
        
        w_sq = -wavelength_mm / (np.pi * imag_part)
        w = np.sqrt(w_sq)
        
        gouy = np.arctan(q.real / q.imag)
        
        if direction > 0:
            waist_position = z - q.real
        else:
            waist_position = z + q.real
        
        zR = q.imag
        waist_radius = np.sqrt(zR * self.beam.m2 * wavelength_mm / np.pi)
        
        return ABCDResult(
            z=z,
            q=q,
            w=w,
            R=R,
            gouy_phase=gouy,
            waist_position=waist_position,
            waist_radius=waist_radius,
            path_length=path_length,
        )
    
    @staticmethod
    def free_space_matrix(d: float) -> NDArray:
        """自由空间传播矩阵"""
        return np.array([[1, d], [0, 1]], dtype=np.float64)
    
    @staticmethod
    def thin_lens_matrix(f: float) -> NDArray:
        """薄透镜矩阵"""
        return np.array([[1, 0], [-1/f, 1]], dtype=np.float64)
    
    @staticmethod
    def mirror_matrix(f: float) -> NDArray:
        """反射镜矩阵"""
        return np.array([[1, 0], [-1/f, 1]], dtype=np.float64)
    
    def _apply_abcd(self, q: complex, abcd: NDArray) -> complex:
        """应用 ABCD 矩阵变换"""
        A, B = abcd[0, 0], abcd[0, 1]
        C, D = abcd[1, 0], abcd[1, 1]
        return (A * q + B) / (C * q + D)
    
    def propagate_distance(self, distance: float) -> ABCDResult:
        """沿光路传播指定距离（Zemax 序列模式）
        
        参数:
            distance: 从初始位置沿光路的累积传播距离（mm）
        
        返回:
            ABCDResult 对象，包含光束参数
        
        示例:
            >>> # 传播 200mm 到达反射镜，再传播 100mm 到达像面
            >>> result = calc.propagate_distance(300.0)
        """
        current_z = self.beam.z_init
        current_q = self.q_init
        direction = 1
        remaining_distance = distance
        
        # 按顺序处理元件
        for element in self.elements:
            # 计算到元件的距离
            d_to_element = element.path_length - (distance - remaining_distance)
            
            if d_to_element < 0:
                # 已经过了这个元件
                continue
            
            if remaining_distance < d_to_element:
                # 不需要经过这个元件，在元件之前停止
                break
            
            # 传播到元件位置
            if d_to_element > 0:
                abcd = self.free_space_matrix(d_to_element)
                current_q = self._apply_abcd(current_q, abcd)
                remaining_distance -= d_to_element
                current_z = element.z_position
            
            # 应用元件变换
            if isinstance(element, (ParabolicMirror, SphericalMirror)):
                f = element.focal_length
                if np.isfinite(f):
                    abcd = self.mirror_matrix(f)
                    current_q = self._apply_abcd(current_q, abcd)
                direction = -direction
            elif isinstance(element, FlatMirror):
                # 平面镜不改变光束参数，只改变方向
                direction = -direction
            elif isinstance(element, ThinLens):
                f = element.focal_length
                abcd = self.thin_lens_matrix(f)
                current_q = self._apply_abcd(current_q, abcd)
        
        # 传播剩余距离
        if remaining_distance > 0:
            abcd = self.free_space_matrix(remaining_distance)
            current_q = self._apply_abcd(current_q, abcd)
            current_z += direction * remaining_distance
        
        return self._q_to_beam_params(current_q, current_z, distance, direction)
    
    def get_output_waist(self) -> Tuple[float, float]:
        """获取输出束腰位置和半径
        
        返回:
            (waist_position, waist_radius) 元组
        """
        # 计算总光程距离
        total_path = self.initial_distance
        for element in self.elements:
            total_path += element.thickness
        
        # 传播到足够远的位置
        far_distance = total_path + 10000
        result = self.propagate_distance(far_distance)
        return result.waist_position, result.waist_radius
    
    def trace_beam_profile(
        self,
        max_distance: float,
        num_points: int = 100,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """追迹光束轮廓
        
        参数:
            max_distance: 最大光程距离（mm）
            num_points: 采样点数
        
        返回:
            (distance_array, z_array, w_array, R_array) 元组
        """
        distance_array = np.linspace(0, max_distance, num_points)
        z_array = np.zeros(num_points)
        w_array = np.zeros(num_points)
        R_array = np.zeros(num_points)
        
        for i, d in enumerate(distance_array):
            result = self.propagate_distance(d)
            z_array[i] = result.z
            w_array[i] = result.w
            R_array[i] = result.R
        
        return distance_array, z_array, w_array, R_array
    
    def get_beam_at_element(
        self,
        element_index: int,
        position: str = 'entrance',
    ) -> ABCDResult:
        """获取指定元件入射面或出射面的光束参数
        
        通过 ABCD 矩阵法计算高斯光束在每个元件处的参数，
        无需从波前拟合估计。由于初始光束参数已知，
        可以精确追踪光束在整个系统中的演变。
        
        参数:
            element_index: 元件索引（从 0 开始）
            position: 'entrance'（入射面）或 'exit'（出射面）
        
        返回:
            ABCDResult 对象，包含光束参数
        
        异常:
            IndexError: 元件索引超出范围
            ValueError: position 参数无效
        
        示例:
            >>> calc = ABCDCalculator(beam, [mirror1, mirror2], initial_distance=100.0)
            >>> # 获取第一个镜子入射面的光束参数
            >>> entrance_params = calc.get_beam_at_element(0, 'entrance')
            >>> print(f"入射光束半径: {entrance_params.w:.3f} mm")
            >>> # 获取第一个镜子出射面的光束参数
            >>> exit_params = calc.get_beam_at_element(0, 'exit')
            >>> print(f"出射光束半径: {exit_params.w:.3f} mm")
        """
        if element_index < 0 or element_index >= len(self.elements):
            raise IndexError(
                f"元件索引 {element_index} 超出范围，"
                f"有效范围为 0 到 {len(self.elements) - 1}"
            )
        
        if position not in ('entrance', 'exit'):
            raise ValueError(
                f"position 参数必须为 'entrance' 或 'exit'，"
                f"实际为 '{position}'"
            )
        
        element = self.elements[element_index]
        
        # 计算到元件的光程距离
        path_to_element = element.path_length
        
        if position == 'entrance':
            # 入射面：传播到元件位置，但不应用元件变换
            return self._propagate_to_element_entrance(element_index)
        else:
            # 出射面：传播到元件位置并应用元件变换
            return self._propagate_to_element_exit(element_index)
    
    def _propagate_to_element_entrance(self, element_index: int) -> ABCDResult:
        """传播到元件入射面（不应用元件变换）"""
        current_z = self.beam.z_init
        current_q = self.q_init
        direction = 1
        current_path = 0.0
        
        # 传播到第一个元件之前的距离
        if self.initial_distance > 0:
            abcd = self.free_space_matrix(self.initial_distance)
            current_q = self._apply_abcd(current_q, abcd)
            current_z += direction * self.initial_distance
            current_path += self.initial_distance
        
        # 按顺序处理元件，直到目标元件
        for i, element in enumerate(self.elements):
            if i == element_index:
                # 到达目标元件入射面
                break
            
            # 应用元件变换
            if isinstance(element, (ParabolicMirror, SphericalMirror)):
                f = element.focal_length
                if np.isfinite(f):
                    abcd = self.mirror_matrix(f)
                    current_q = self._apply_abcd(current_q, abcd)
                direction = -direction
            elif isinstance(element, FlatMirror):
                # 平面镜不改变光束参数，只改变方向
                direction = -direction
            elif isinstance(element, ThinLens):
                f = element.focal_length
                abcd = self.thin_lens_matrix(f)
                current_q = self._apply_abcd(current_q, abcd)
            
            # 传播到下一个元件
            if element.thickness > 0:
                abcd = self.free_space_matrix(element.thickness)
                current_q = self._apply_abcd(current_q, abcd)
                current_z += direction * element.thickness
                current_path += element.thickness
        
        target_element = self.elements[element_index]
        return self._q_to_beam_params(current_q, target_element.z_position, 
                                       target_element.path_length, direction)
    
    def _propagate_to_element_exit(self, element_index: int) -> ABCDResult:
        """传播到元件出射面（应用元件变换）"""
        # 先获取入射面参数
        entrance_result = self._propagate_to_element_entrance(element_index)
        current_q = entrance_result.q
        current_z = entrance_result.z
        current_path = entrance_result.path_length
        
        # 确定当前传播方向
        direction = 1
        for i in range(element_index):
            if self.elements[i].is_reflective:
                direction = -direction
        
        # 应用元件变换
        element = self.elements[element_index]
        if isinstance(element, (ParabolicMirror, SphericalMirror)):
            f = element.focal_length
            if np.isfinite(f):
                abcd = self.mirror_matrix(f)
                current_q = self._apply_abcd(current_q, abcd)
            direction = -direction
        elif isinstance(element, FlatMirror):
            # 平面镜不改变光束参数，只改变方向
            direction = -direction
        elif isinstance(element, ThinLens):
            f = element.focal_length
            abcd = self.thin_lens_matrix(f)
            current_q = self._apply_abcd(current_q, abcd)
        
        return self._q_to_beam_params(current_q, current_z, current_path, direction)
    
    def get_all_element_beam_params(self) -> List[Dict[str, ABCDResult]]:
        """获取所有元件入射面和出射面的光束参数
        
        返回:
            列表，每个元素是包含 'entrance' 和 'exit' 键的字典
        
        示例:
            >>> calc = ABCDCalculator(beam, [mirror1, mirror2], initial_distance=100.0)
            >>> all_params = calc.get_all_element_beam_params()
            >>> for i, params in enumerate(all_params):
            ...     print(f"元件 {i}:")
            ...     print(f"  入射: w={params['entrance'].w:.3f} mm, R={params['entrance'].R:.3f} mm")
            ...     print(f"  出射: w={params['exit'].w:.3f} mm, R={params['exit'].R:.3f} mm")
        """
        results = []
        for i in range(len(self.elements)):
            results.append({
                'entrance': self.get_beam_at_element(i, 'entrance'),
                'exit': self.get_beam_at_element(i, 'exit'),
            })
        return results
    
    def compute_reference_phase_at_position(
        self,
        x: NDArray,
        y: NDArray,
        element_index: int,
        position: str = 'exit',
    ) -> NDArray:
        """在指定位置计算高斯光束参考相位
        
        使用 ABCD 法则计算的光束参数，在给定的 (x, y) 坐标处
        计算高斯光束的球面波前相位。这是 Pilot Beam 方法的核心，
        用于从光线追迹的 OPD 中提取残差相位。
        
        参数:
            x: x 坐标数组（mm）
            y: y 坐标数组（mm）
            element_index: 元件索引
            position: 'entrance' 或 'exit'
        
        返回:
            参考相位数组（弧度）
        
        高斯光束相位公式：
            φ(r) = k * r² / (2 * R)
        
        其中：
        - k = 2π/λ 是波数
        - r² = x² + y² 是径向距离平方
        - R 是波前曲率半径
        
        示例:
            >>> # 在光线位置计算参考相位
            >>> x_rays = np.array([0.0, 1.0, 2.0])
            >>> y_rays = np.array([0.0, 0.0, 0.0])
            >>> ref_phase = calc.compute_reference_phase_at_position(
            ...     x_rays, y_rays, element_index=0, position='exit'
            ... )
        """
        # 获取光束参数
        beam_params = self.get_beam_at_element(element_index, position)
        
        # 计算波数
        wavelength_mm = self.beam.wavelength * 1e-3
        k = 2 * np.pi / wavelength_mm
        
        # 计算径向距离平方
        r_squared = np.asarray(x)**2 + np.asarray(y)**2
        
        # 计算参考相位
        R = beam_params.R
        if np.isinf(R):
            # 平面波前，相位为零
            return np.zeros_like(r_squared)
        
        # 球面波前相位：φ = k * r² / (2 * R)
        return k * r_squared / (2 * R)
    
    def compute_reference_phase_grid(
        self,
        grid_size: int,
        physical_size: float,
        element_index: int,
        position: str = 'exit',
    ) -> NDArray:
        """在网格上计算高斯光束参考相位
        
        参数:
            grid_size: 网格大小（像素）
            physical_size: 物理尺寸（直径，mm）
            element_index: 元件索引
            position: 'entrance' 或 'exit'
        
        返回:
            参考相位网格（弧度），形状 (grid_size, grid_size)
        
        示例:
            >>> ref_phase_grid = calc.compute_reference_phase_grid(
            ...     grid_size=512, physical_size=20.0,
            ...     element_index=0, position='exit'
            ... )
        """
        half_size = physical_size / 2
        coords = np.linspace(-half_size, half_size, grid_size)
        X, Y = np.meshgrid(coords, coords)
        
        return self.compute_reference_phase_at_position(
            X.ravel(), Y.ravel(), element_index, position
        ).reshape(grid_size, grid_size)
