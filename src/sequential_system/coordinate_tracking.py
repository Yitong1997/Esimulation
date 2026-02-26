"""
光轴方向跟踪模块

本模块实现序列光学系统中的光轴方向跟踪，用于：
1. 计算主光线在各元件处的传播方向
2. 计算各元件和采样面在全局坐标系中的位置和方向
3. 支持绘制带有空间坐标的 2D 光路布置图

设计原则：
- 光轴是动态的：每个反射元件都会改变光轴方向
- 所有面的倾斜都相对于当前光轴定义
- 采样面默认垂直于当前光轴（主光线）

坐标系约定：
- 全局坐标系：右手系，Z 轴为初始光轴方向
- 局部坐标系：每个元件有自己的局部坐标系，Z 轴沿当前光轴

作者：混合光学仿真项目
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray


@dataclass
class RayDirection:
    """光线方向（方向余弦）
    
    使用方向余弦 (L, M, N) 表示光线方向：
    - L: x 方向分量
    - M: y 方向分量  
    - N: z 方向分量
    - 满足 L² + M² + N² = 1
    """
    L: float = 0.0
    M: float = 0.0
    N: float = 1.0
    
    def __post_init__(self):
        """归一化方向向量"""
        norm = np.sqrt(self.L**2 + self.M**2 + self.N**2)
        if norm < 1e-15:
            raise ValueError("方向向量不能为零向量")
        self.L /= norm
        self.M /= norm
        self.N /= norm
    
    def to_array(self) -> NDArray:
        """转换为 numpy 数组"""
        return np.array([self.L, self.M, self.N])
    
    @classmethod
    def from_array(cls, arr: NDArray) -> "RayDirection":
        """从 numpy 数组创建"""
        return cls(L=float(arr[0]), M=float(arr[1]), N=float(arr[2]))
    
    def reflect(self, normal: "RayDirection") -> "RayDirection":
        """计算反射后的方向
        
        反射公式：r = d - 2(d·n)n
        其中 d 是入射方向，n 是法向量，r 是反射方向
        
        参数:
            normal: 表面法向量（指向入射侧）
        
        返回:
            反射后的方向
        """
        d = self.to_array()
        n = normal.to_array()
        
        # 反射公式
        r = d - 2 * np.dot(d, n) * n
        
        return RayDirection.from_array(r)
    
    def rotate_x(self, angle: float) -> "RayDirection":
        """绕 X 轴旋转
        
        参数:
            angle: 旋转角度（弧度），正值为右手定则方向
        """
        c, s = np.cos(angle), np.sin(angle)
        new_M = self.M * c - self.N * s
        new_N = self.M * s + self.N * c
        return RayDirection(L=self.L, M=new_M, N=new_N)
    
    def rotate_y(self, angle: float) -> "RayDirection":
        """绕 Y 轴旋转
        
        参数:
            angle: 旋转角度（弧度），正值为右手定则方向
        """
        c, s = np.cos(angle), np.sin(angle)
        new_L = self.L * c + self.N * s
        new_N = -self.L * s + self.N * c
        return RayDirection(L=new_L, M=self.M, N=new_N)
    
    def angle_with(self, other: "RayDirection") -> float:
        """计算与另一方向的夹角（弧度）"""
        dot = self.L * other.L + self.M * other.M + self.N * other.N
        # 限制在 [-1, 1] 范围内，避免数值误差
        dot = np.clip(dot, -1.0, 1.0)
        return np.arccos(dot)


@dataclass
class Position3D:
    """3D 位置"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_array(self) -> NDArray:
        """转换为 numpy 数组"""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: NDArray) -> "Position3D":
        """从 numpy 数组创建"""
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))
    
    def advance(self, direction: RayDirection, distance: float) -> "Position3D":
        """沿指定方向前进指定距离
        
        参数:
            direction: 前进方向
            distance: 前进距离（mm）
        
        返回:
            新位置
        """
        new_pos = self.to_array() + distance * direction.to_array()
        return Position3D.from_array(new_pos)


@dataclass
class LocalCoordinateSystem:
    """局部坐标系
    
    定义元件或采样面的局部坐标系，包括：
    - 原点位置（全局坐标）
    - 光轴方向（局部 Z 轴）
    - 局部 X、Y 轴方向
    """
    origin: Position3D = field(default_factory=Position3D)
    z_axis: RayDirection = field(default_factory=RayDirection)  # 光轴方向
    x_axis: RayDirection = field(default_factory=lambda: RayDirection(L=1.0, M=0.0, N=0.0))
    y_axis: RayDirection = field(default_factory=lambda: RayDirection(L=0.0, M=1.0, N=0.0))
    
    def get_surface_normal(self, tilt_x: float = 0.0, tilt_y: float = 0.0) -> RayDirection:
        """获取表面法向量（考虑倾斜）
        
        倾斜是相对于当前光轴定义的：
        - tilt_x: 绕局部 X 轴旋转
        - tilt_y: 绕局部 Y 轴旋转
        
        参数:
            tilt_x: 绕 X 轴旋转角度（弧度）
            tilt_y: 绕 Y 轴旋转角度（弧度）
        
        返回:
            表面法向量（指向入射侧）
        """
        # 初始法向量沿光轴方向（但指向入射侧，即反方向）
        normal = RayDirection(L=-self.z_axis.L, M=-self.z_axis.M, N=-self.z_axis.N)
        
        # 应用倾斜（先绕 X 轴，再绕 Y 轴）
        if tilt_x != 0:
            normal = normal.rotate_x(tilt_x)
        if tilt_y != 0:
            normal = normal.rotate_y(tilt_y)
        
        return normal


@dataclass
class OpticalAxisState:
    """光轴状态
    
    记录光轴在某一位置的状态，包括：
    - 位置（全局坐标）
    - 方向（方向余弦）
    - 累积光程
    """
    position: Position3D
    direction: RayDirection
    path_length: float  # 累积光程（mm）
    
    def propagate(self, distance: float) -> "OpticalAxisState":
        """沿当前方向传播指定距离
        
        参数:
            distance: 传播距离（mm）
        
        返回:
            新的光轴状态
        """
        new_position = self.position.advance(self.direction, distance)
        return OpticalAxisState(
            position=new_position,
            direction=self.direction,
            path_length=self.path_length + distance,
        )
    
    def reflect(self, surface_normal: RayDirection) -> "OpticalAxisState":
        """在表面反射
        
        参数:
            surface_normal: 表面法向量
        
        返回:
            反射后的光轴状态（位置不变，方向改变）
        """
        new_direction = self.direction.reflect(surface_normal)
        return OpticalAxisState(
            position=self.position,
            direction=new_direction,
            path_length=self.path_length,
        )


class OpticalAxisTracker:
    """光轴跟踪器
    
    跟踪光轴在整个光学系统中的演变，记录：
    - 每个元件处的光轴状态
    - 每个采样面处的光轴状态
    - 全局坐标系中的位置和方向
    
    使用方法:
        tracker = OpticalAxisTracker()
        
        # 添加元件
        for element in elements:
            tracker.add_element(element)
        
        # 获取某位置的光轴状态
        state = tracker.get_state_at_distance(100.0)
        
        # 获取所有元件的全局位置
        positions = tracker.get_element_positions()
    """
    
    def __init__(self):
        """初始化光轴跟踪器"""
        # 初始状态：原点，沿 +Z 方向
        self._initial_state = OpticalAxisState(
            position=Position3D(0.0, 0.0, 0.0),
            direction=RayDirection(0.0, 0.0, 1.0),
            path_length=0.0,
        )
        self._current_state = self._initial_state
        
        # 记录每个元件的状态
        self._element_states: List[Tuple[any, OpticalAxisState, OpticalAxisState]] = []
        # (element, state_before, state_after)
        
        # 记录采样面状态
        self._sampling_states: List[Tuple[float, OpticalAxisState]] = []
        # (distance, state)
    
    def reset(self):
        """重置跟踪器"""
        self._current_state = self._initial_state
        self._element_states.clear()
        self._sampling_states.clear()
    
    def add_element(self, element) -> Tuple[OpticalAxisState, OpticalAxisState]:
        """添加光学元件并更新光轴状态
        
        参数:
            element: 光学元件对象
        
        返回:
            (元件前状态, 元件后状态)
        """
        # 元件前的状态
        state_before = self._current_state
        
        # 如果是反射元件，计算反射后的方向
        if element.is_reflective:
            # 获取表面法向量（考虑倾斜）
            local_cs = LocalCoordinateSystem(
                origin=self._current_state.position,
                z_axis=self._current_state.direction,
            )
            surface_normal = local_cs.get_surface_normal(
                tilt_x=element.tilt_x,
                tilt_y=element.tilt_y,
            )
            
            # 反射
            self._current_state = self._current_state.reflect(surface_normal)
        
        # 传播到下一元件
        state_after_reflection = self._current_state
        self._current_state = self._current_state.propagate(element.thickness)
        
        # 记录
        self._element_states.append((element, state_before, state_after_reflection))
        
        return state_before, state_after_reflection
    
    def get_state_at_distance(self, distance: float) -> OpticalAxisState:
        """获取指定光程距离处的光轴状态
        
        参数:
            distance: 从起点的光程距离（mm）
        
        返回:
            该位置的光轴状态
        """
        # 从初始状态开始
        state = self._initial_state
        
        for element, state_before, state_after in self._element_states:
            elem_distance = state_before.path_length
            
            if distance <= elem_distance:
                # 在此元件之前
                remaining = distance - state.path_length
                return state.propagate(remaining)
            
            # 更新到元件后的状态
            state = state_after.propagate(element.thickness)
        
        # 在所有元件之后
        remaining = distance - state.path_length
        if remaining > 0:
            state = state.propagate(remaining)
        
        return state
    
    def get_element_global_positions(self) -> List[Tuple[any, Position3D, RayDirection, RayDirection]]:
        """获取所有元件的全局位置和方向
        
        返回:
            列表，每项为 (element, position, direction_before, direction_after)
        """
        results = []
        for element, state_before, state_after in self._element_states:
            results.append((
                element,
                state_before.position,
                state_before.direction,
                state_after.direction,
            ))
        return results
    
    def get_sampling_plane_state(self, distance: float) -> OpticalAxisState:
        """获取采样面的光轴状态
        
        参数:
            distance: 采样面的光程距离（mm）
        
        返回:
            采样面处的光轴状态
        """
        return self.get_state_at_distance(distance)
    
    def calculate_beam_path_2d(
        self,
        num_points: int = 100,
        projection: str = "yz",
    ) -> Tuple[NDArray, NDArray]:
        """计算光束路径的 2D 投影
        
        参数:
            num_points: 路径点数
            projection: 投影平面，"xy", "xz", 或 "yz"
        
        返回:
            (x_coords, y_coords) 投影坐标
        """
        if not self._element_states:
            return np.array([0.0]), np.array([0.0])
        
        # 获取总光程
        last_elem, _, state_after = self._element_states[-1]
        total_path = state_after.path_length + last_elem.thickness
        
        # 生成路径点
        distances = np.linspace(0, total_path, num_points)
        positions = []
        
        for d in distances:
            state = self.get_state_at_distance(d)
            positions.append(state.position.to_array())
        
        positions = np.array(positions)
        
        # 选择投影
        if projection == "xy":
            return positions[:, 0], positions[:, 1]
        elif projection == "xz":
            return positions[:, 0], positions[:, 2]
        else:  # "yz"
            return positions[:, 2], positions[:, 1]  # z 作为水平轴


def calculate_reflection_direction(
    incident: RayDirection,
    tilt_x: float,
    tilt_y: float,
) -> RayDirection:
    """计算反射后的光线方向
    
    假设入射光线沿当前光轴方向，表面法向量由倾斜角度决定。
    
    参数:
        incident: 入射方向
        tilt_x: 表面绕 X 轴倾斜角度（弧度）
        tilt_y: 表面绕 Y 轴倾斜角度（弧度）
    
    返回:
        反射后的方向
    
    示例:
        >>> # 45° 折叠镜
        >>> incident = RayDirection(0, 0, 1)  # 沿 +Z 入射
        >>> reflected = calculate_reflection_direction(incident, np.pi/4, 0)
        >>> # 反射后应该沿 -Y 方向
        >>> print(f"L={reflected.L:.3f}, M={reflected.M:.3f}, N={reflected.N:.3f}")
    """
    # 表面法向量初始沿 -Z（指向入射侧）
    normal = RayDirection(0, 0, -1)
    
    # 应用倾斜
    if tilt_x != 0:
        normal = normal.rotate_x(tilt_x)
    if tilt_y != 0:
        normal = normal.rotate_y(tilt_y)
    
    # 计算反射
    return incident.reflect(normal)
