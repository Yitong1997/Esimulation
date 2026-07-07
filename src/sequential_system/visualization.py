"""
光路可视化模块

本模块实现 LayoutVisualizer 类，用于绘制 2D 光路图和采样面结果。

验证需求:
- Requirements 7.1-7.7: 可视化

作者：混合光学仿真项目
"""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING, Optional, List
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from .system import SequentialOpticalSystem
    from .sampling import SamplingResult, SimulationResults


def draw_parabolic_mirror(ax, z_pos: float, aperture: float, focal_length: float,
                          color: str = 'purple', label: str = None, 
                          mirror_thickness: float = 3.0):
    """绘制抛物面镜的2D截面"""
    y = np.linspace(-aperture, aperture, 100)
    
    if abs(focal_length) > 1e-6 and np.isfinite(focal_length):
        sag = y**2 / (4 * focal_length)
    else:
        sag = np.zeros_like(y)
    
    z_front = z_pos + sag
    
    if focal_length > 0:
        z_back = z_front - mirror_thickness
    else:
        z_back = z_front + mirror_thickness
    
    ax.fill_betweenx(y, z_front, z_back, color=color, alpha=0.6, label=label)
    ax.plot(z_front, y, color=color, linewidth=2)
    ax.plot(z_back, y, color=color, linewidth=1, linestyle='--', alpha=0.5)
    ax.plot([z_front[0], z_back[0]], [y[0], y[0]], color=color, linewidth=1.5)
    ax.plot([z_front[-1], z_back[-1]], [y[-1], y[-1]], color=color, linewidth=1.5)


def draw_spherical_mirror(ax, z_pos: float, aperture: float, radius: float,
                          color: str = 'blue', label: str = None,
                          mirror_thickness: float = 3.0):
    """绘制球面镜的2D截面"""
    y = np.linspace(-aperture, aperture, 100)
    
    if abs(radius) > 1e-6 and np.isfinite(radius):
        sag = radius - np.sign(radius) * np.sqrt(radius**2 - y**2)
    else:
        sag = np.zeros_like(y)
    
    z_front = z_pos + sag
    
    if radius > 0:
        z_back = z_front - mirror_thickness
    else:
        z_back = z_front + mirror_thickness
    
    ax.fill_betweenx(y, z_front, z_back, color=color, alpha=0.6, label=label)
    ax.plot(z_front, y, color=color, linewidth=2)
    ax.plot(z_back, y, color=color, linewidth=1, linestyle='--', alpha=0.5)


def draw_thin_lens(ax, z_pos: float, aperture: float, focal_length: float,
                   color: str = 'green', label: str = None):
    """绘制薄透镜的2D截面"""
    y = np.array([-aperture, 0, aperture])
    
    if focal_length > 0:
        # 会聚透镜：中间厚
        x_offset = np.array([2, 4, 2])
    else:
        # 发散透镜：中间薄
        x_offset = np.array([4, 2, 4])
    
    z_left = z_pos - x_offset / 2
    z_right = z_pos + x_offset / 2
    
    ax.fill_betweenx(y, z_left, z_right, color=color, alpha=0.5, label=label)
    ax.plot(z_left, y, color=color, linewidth=2)
    ax.plot(z_right, y, color=color, linewidth=2)


def draw_flat_mirror(ax, z_pos: float, aperture: float, tilt_angle: float = 0,
                     color: str = 'gray', label: str = None,
                     mirror_thickness: float = 3.0):
    """绘制平面镜的2D截面"""
    y = np.array([-aperture, aperture])
    z = np.array([z_pos, z_pos])
    
    ax.fill_betweenx(y, z, z - mirror_thickness, color=color, alpha=0.6, label=label)
    ax.plot(z, y, color=color, linewidth=2)


def draw_gaussian_beam_2d(ax, z_array: np.ndarray, w_array: np.ndarray,
                          color: str = 'red', alpha: float = 0.3,
                          label: str = None, show_rays: bool = True):
    """绘制高斯光束的2D表示"""
    ax.fill_between(z_array, w_array, -w_array, color=color, alpha=alpha, label=label)
    ax.plot(z_array, w_array, color=color, linewidth=1.5)
    ax.plot(z_array, -w_array, color=color, linewidth=1.5)
    
    if show_rays and len(z_array) > 1:
        for frac in [0.3, 0.6]:
            ray_y = w_array * frac
            ax.plot(z_array, ray_y, color=color, linewidth=0.5, alpha=0.5)
            ax.plot(z_array, -ray_y, color=color, linewidth=0.5, alpha=0.5)


class LayoutVisualizer:
    """光路 2D 可视化器
    
    支持两种绘图模式：
    1. 展开模式（默认）：沿光程距离展开，所有元件在一条直线上
    2. 空间模式：显示真实的空间坐标，包括折叠光路
    """
    
    def __init__(self, system: "SequentialOpticalSystem") -> None:
        self._system = system
    
    def draw(
        self,
        figsize: Tuple[float, float] = (14, 6),
        show: bool = True,
        mode: str = "unfolded",
    ) -> Tuple["Figure", "Axes"]:
        """绘制 2D 光路图
        
        参数:
            figsize: 图形大小
            show: 是否显示图形
            mode: 绘图模式
                - "unfolded": 展开模式，沿光程距离展开
                - "spatial": 空间模式，显示真实空间坐标
        
        返回:
            (fig, ax) 元组
        """
        if mode == "spatial":
            return self._draw_spatial(figsize=figsize, show=show)
        else:
            return self._draw_unfolded(figsize=figsize, show=show)
    
    def _draw_unfolded(
        self,
        figsize: Tuple[float, float] = (14, 6),
        show: bool = True,
    ) -> Tuple["Figure", "Axes"]:
        """绘制展开的 2D 光路图（原有实现）"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize)
        
        total_path = self._system.total_path_length
        if total_path == 0:
            total_path = 100.0
        
        # 绘制光束包络
        self._draw_beam_envelope(ax, total_path)
        
        # 绘制光学元件（实际面形）
        self._draw_elements_realistic(ax)
        
        # 绘制采样面
        self._draw_sampling_planes(ax)
        
        ax.set_xlabel("Path Length (mm)")
        ax.set_ylabel("Y Position (mm)")
        ax.set_title("Sequential Optical System Layout")
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal', adjustable='datalim')
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig, ax
    
    def _draw_spatial(
        self,
        figsize: Tuple[float, float] = (14, 8),
        show: bool = True,
    ) -> Tuple["Figure", "Axes"]:
        """绘制空间坐标的 2D 光路图
        
        显示真实的空间坐标，包括折叠光路。
        使用 YZ 平面投影（Z 为水平轴，Y 为垂直轴）。
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize)
        
        tracker = self._system.axis_tracker
        
        # 绘制光束路径
        self._draw_beam_path_spatial(ax, tracker)
        
        # 绘制光学元件
        self._draw_elements_spatial(ax, tracker)
        
        # 绘制采样面
        self._draw_sampling_planes_spatial(ax, tracker)
        
        ax.set_xlabel("Z Position (mm)")
        ax.set_ylabel("Y Position (mm)")
        ax.set_title("Sequential Optical System Layout (Spatial View)")
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal', adjustable='datalim')
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig, ax
    
    def _draw_beam_path_spatial(self, ax, tracker) -> None:
        """绘制空间坐标中的光束路径"""
        # 获取光束路径的 2D 投影
        z_coords, y_coords = tracker.calculate_beam_path_2d(
            num_points=200,
            projection="yz"
        )
        
        # 绘制光束中心线
        ax.plot(z_coords, y_coords, 'b-', linewidth=2, alpha=0.7, label='Beam Path')
        
        # 绘制入射光方向箭头
        self._draw_incident_beam_arrow(ax, tracker)
    
    def _draw_incident_beam_arrow(self, ax, tracker) -> None:
        """绘制入射光方向箭头
        
        在第一个元件之前绘制一个箭头，表示入射光的方向。
        """
        # 获取初始光轴状态
        initial_state = tracker.get_state_at_distance(0.0)
        
        # 入射光方向
        dir_arr = initial_state.direction.to_array()
        
        # 计算箭头的起点和终点（在 YZ 平面投影）
        # 箭头长度根据系统尺寸自动调整
        if self._system.elements:
            first_elem = self._system.elements[0]
            arrow_length = min(first_elem.thickness * 0.5, 20.0)
            if arrow_length < 5.0:
                arrow_length = 10.0
        else:
            arrow_length = 10.0
        
        # 箭头起点（在第一个元件之前）
        start_z = initial_state.position.z - arrow_length * dir_arr[2]
        start_y = initial_state.position.y - arrow_length * dir_arr[1]
        
        # 箭头终点（第一个元件位置）
        end_z = initial_state.position.z
        end_y = initial_state.position.y
        
        # 绘制箭头
        ax.annotate(
            '',
            xy=(end_z, end_y),
            xytext=(start_z, start_y),
            arrowprops=dict(
                arrowstyle='->',
                color='green',
                lw=2.5,
                mutation_scale=15,
            ),
        )
        
        # 添加"入射光"标签
        label_z = start_z - 2
        label_y = start_y
        ax.annotate(
            'Incident',
            xy=(label_z, label_y),
            fontsize=9,
            color='green',
            ha='right',
            va='center',
        )
    
    def _draw_elements_spatial(self, ax, tracker) -> None:
        """绘制空间坐标中的光学元件（按实际面型和倾斜角度）"""
        from .coordinate_tracking import LocalCoordinateSystem
        
        colors = ['purple', 'brown', 'teal', 'orange', 'pink']
        element_positions = tracker.get_element_global_positions()
        
        for i, (element, position, dir_before, dir_after) in enumerate(element_positions):
            color = colors[i % len(colors)]
            name = element.name if element.name else f"{element.element_type}"
            
            z_pos = position.z
            y_pos = position.y
            aperture = element.semi_aperture
            
            # 计算表面法向量（考虑倾斜）
            local_cs = LocalCoordinateSystem(
                origin=position,
                z_axis=dir_before,
            )
            surface_normal = local_cs.get_surface_normal(
                tilt_x=element.tilt_x,
                tilt_y=element.tilt_y,
            )
            
            # 在 YZ 平面内，计算表面的切向量（垂直于法向量）
            normal_arr = surface_normal.to_array()
            tangent_y = normal_arr[2]   # 法向量 z 分量
            tangent_z = -normal_arr[1]  # 法向量 y 分量取负
            tangent_norm = np.sqrt(tangent_y**2 + tangent_z**2)
            
            if tangent_norm < 1e-6:
                tangent_y, tangent_z = 1.0, 0.0
            else:
                tangent_y /= tangent_norm
                tangent_z /= tangent_norm
            
            # 生成沿表面的局部坐标
            num_points = 50
            local_coords = np.linspace(-aperture, aperture, num_points)
            
            # 计算矢高
            sag = self._calculate_surface_sag(element, local_coords)
            
            # 转换到全局坐标
            z_coords = z_pos + local_coords * tangent_z + sag * normal_arr[2]
            y_coords = y_pos + local_coords * tangent_y + sag * normal_arr[1]
            
            # 绘制表面
            ax.plot(z_coords, y_coords, color=color, linewidth=3, label=name)
            
            # 标注元件名称
            ax.annotate(name, xy=(z_pos, y_pos), fontsize=8, color=color,
                        ha='center', va='bottom', xytext=(0, 5),
                        textcoords='offset points')
    
    def _calculate_surface_sag(self, element, local_coords: np.ndarray) -> np.ndarray:
        """计算元件表面的矢高（sag）"""
        element_type = element.element_type
        
        if element_type == "flat_mirror":
            return np.zeros_like(local_coords)
        
        elif element_type == "parabolic_mirror":
            f = element.parent_focal_length
            r = local_coords
            sag = r**2 / (4 * f)
            return sag
        
        elif element_type == "spherical_mirror":
            R = element.radius_of_curvature
            if np.isinf(R):
                return np.zeros_like(local_coords)
            r = local_coords
            r_clipped = np.clip(np.abs(r), 0, np.abs(R) * 0.99)
            sag = R - np.sign(R) * np.sqrt(R**2 - r_clipped**2)
            return sag
        
        else:
            return np.zeros_like(local_coords)
    
    def _draw_sampling_planes_spatial(self, ax, tracker) -> None:
        """绘制空间坐标中的采样面（作为垂直于光轴的平面）"""
        for plane in self._system.sampling_planes:
            # 获取采样面的光轴状态
            state = tracker.get_state_at_distance(plane.distance)
            
            z_pos = state.position.z
            y_pos = state.position.y
            
            # 采样面垂直于光轴
            dir_arr = state.direction.to_array()
            perp_y = -dir_arr[2]
            perp_z = dir_arr[1]
            perp_norm = np.sqrt(perp_y**2 + perp_z**2)
            if perp_norm > 1e-6:
                perp_y /= perp_norm
                perp_z /= perp_norm
            else:
                perp_y, perp_z = 1.0, 0.0
            
            # 采样面半长度（根据系统中最大孔径确定）
            max_aperture = 10.0
            if self._system.elements:
                max_aperture = max(e.semi_aperture for e in self._system.elements)
            length = max_aperture * 0.6
            
            # 绘制采样面（平面线段）
            z1 = z_pos + length * perp_z
            y1 = y_pos + length * perp_y
            z2 = z_pos - length * perp_z
            y2 = y_pos - length * perp_y
            
            ax.plot([z1, z2], [y1, y2], color='red', linewidth=2, 
                    linestyle='--', alpha=0.8)
            ax.plot(z_pos, y_pos, 'r*', markersize=12, alpha=0.9)
            
            # 标注名称
            label = plane.name if plane.name else f"d={plane.distance}"
            ax.annotate(label, xy=(z_pos, y_pos), fontsize=8, color='red',
                        ha='center', va='top', xytext=(0, -10),
                        textcoords='offset points')

    def _draw_beam_envelope(self, ax, total_path: float) -> None:
        """绘制光束包络"""
        from gaussian_beam_simulation.abcd_calculator import ABCDCalculator
        
        beam = self._system.source.to_gaussian_beam()
        calculator = ABCDCalculator(beam, self._system.elements)
        
        num_points = 200
        distances = np.linspace(0, total_path * 1.1, num_points)
        
        w_values = []
        for d in distances:
            try:
                result = calculator.propagate_distance(d)
                w_values.append(result.w)
            except Exception:
                w_values.append(np.nan)
        
        w_values = np.array(w_values)
        draw_gaussian_beam_2d(ax, distances, w_values, color='blue', alpha=0.2,
                              label='Beam Envelope', show_rays=True)
    
    def _draw_elements_realistic(self, ax) -> None:
        """绘制光学元件（实际面形）"""
        colors = ['purple', 'brown', 'teal', 'orange', 'pink']
        
        for i, element in enumerate(self._system.elements):
            path_length = element.path_length
            aperture = element.semi_aperture
            color = colors[i % len(colors)]
            name = element.name if element.name else f"{element.element_type}"
            
            if element.element_type == "parabolic_mirror":
                draw_parabolic_mirror(ax, path_length, aperture, 
                                      element.focal_length, color=color, label=name)
            elif element.element_type == "spherical_mirror":
                draw_spherical_mirror(ax, path_length, aperture,
                                      element.radius_of_curvature, color=color, label=name)
            elif element.element_type == "thin_lens":
                draw_thin_lens(ax, path_length, aperture,
                               element.focal_length, color=color, label=name)
            elif element.element_type == "flat_mirror":
                draw_flat_mirror(ax, path_length, aperture, color=color, label=name)
            else:
                ax.axvline(x=path_length, color=color, linestyle='--', linewidth=1.5)
    
    def _draw_sampling_planes(self, ax) -> None:
        """绘制采样面"""
        for plane in self._system.sampling_planes:
            ax.axvline(x=plane.distance, color='red', linestyle=':', linewidth=2, alpha=0.8)
            label = plane.name if plane.name else f"d={plane.distance}"
            y_pos = ax.get_ylim()[1] * 0.9
            ax.annotate(label, xy=(plane.distance, y_pos), fontsize=8, color='red',
                        ha='center', va='top')


def plot_sampling_results(results: "SimulationResults", 
                          figsize: Tuple[float, float] = (15, 10),
                          show: bool = True) -> Tuple["Figure", List["Axes"]]:
    """绘制所有采样面的强度和相位分布
    
    参数:
        results: 仿真结果
        figsize: 图形大小
        show: 是否显示
    
    返回:
        (fig, axes) 元组
    """
    import matplotlib.pyplot as plt
    
    n_planes = len(results)
    if n_planes == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No sampling planes", ha='center', va='center')
        return fig, [ax]
    
    fig, axes = plt.subplots(n_planes, 3, figsize=figsize)
    if n_planes == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        name = result.name if result.name else f"Plane {i}"
        
        # 强度分布
        intensity = result.amplitude**2
        extent = [-result.physical_size/2, result.physical_size/2,
                  -result.physical_size/2, result.physical_size/2]
        
        im0 = axes[i, 0].imshow(intensity, extent=extent, cmap='hot', origin='lower')
        axes[i, 0].set_title(f"{name}: Intensity")
        axes[i, 0].set_xlabel("X (mm)")
        axes[i, 0].set_ylabel("Y (mm)")
        plt.colorbar(im0, ax=axes[i, 0], label='I (a.u.)')
        
        # 相位分布
        phase = result.phase
        im1 = axes[i, 1].imshow(phase, extent=extent, cmap='twilight', origin='lower',
                                 vmin=-np.pi, vmax=np.pi)
        axes[i, 1].set_title(f"{name}: Phase")
        axes[i, 1].set_xlabel("X (mm)")
        axes[i, 1].set_ylabel("Y (mm)")
        plt.colorbar(im1, ax=axes[i, 1], label='Phase (rad)')
        
        # 截面曲线
        n = result.grid_size
        center = n // 2
        x_coords = np.linspace(-result.physical_size/2, result.physical_size/2, n)
        
        ax2 = axes[i, 2]
        ax2.plot(x_coords, intensity[center, :], 'r-', label='Intensity')
        ax2.set_xlabel("X (mm)")
        ax2.set_ylabel("Intensity", color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x_coords, phase[center, :], 'b--', label='Phase')
        ax2_twin.set_ylabel("Phase (rad)", color='b')
        ax2_twin.tick_params(axis='y', labelcolor='b')
        
        ax2.set_title(f"{name}: Cross-section (w={result.beam_radius:.3f}mm)")
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, axes
