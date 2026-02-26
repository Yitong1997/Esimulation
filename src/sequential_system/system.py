"""
序列光学系统核心类

本模块实现 SequentialOpticalSystem 类，提供类似 Zemax 序列模式的接口。

验证需求:
- Requirements 3.1-3.8: 系统构建
- Requirements 5.1-5.7: 仿真执行
- Requirements 6.1-6.5: ABCD 计算

作者：混合光学仿真项目
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .source import GaussianBeamSource
from .sampling import SamplingPlane, SamplingResult, SimulationResults
from .coordinate_tracking import OpticalAxisTracker, OpticalAxisState
from .exceptions import (
    SequentialSystemError,
    SurfaceConfigurationError,
    SimulationError,
    SamplingError,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


@dataclass
class ABCDResult:
    """ABCD 计算结果"""
    distance: float
    w: float
    R: float
    z_position: float


class SequentialOpticalSystem:
    """序列光学系统
    
    提供类似 Zemax 序列模式的接口，用于定义和仿真光学系统。
    结合 PROPER 物理光学衍射传播和 ABCD 矩阵计算，实现混合光学仿真。
    
    参数:
        source: 光源定义（GaussianBeamSource 对象）
        grid_size: 波前网格大小，默认 512
        beam_ratio: PROPER beam_ratio 参数，默认 0.5
        use_hybrid_propagation: 是否使用混合传播模式，默认 False
            - True: 在元件处使用 HybridElementPropagator 进行波前-光线-波前重建
            - False: 使用纯 PROPER 传播（透镜相位）
        hybrid_num_rays: 混合传播模式下的采样光线数量，默认 100
    
    属性:
        source: 光源对象
        grid_size: 网格大小
        beam_ratio: PROPER beam_ratio
        elements: 光学元件列表（只读副本）
        sampling_planes: 采样面列表（只读副本）
        total_path_length: 总光程长度
        use_hybrid_propagation: 是否使用混合传播模式
    
    示例:
        基本使用 - 创建简单的聚焦系统::
        
            from sequential_system import (
                SequentialOpticalSystem,
                GaussianBeamSource,
                SphericalMirror,
            )
            
            # 1. 定义光源（波长 633nm，束腰 1mm）
            source = GaussianBeamSource(
                wavelength=0.633,  # μm
                w0=1.0,            # mm
                z0=0.0,            # mm，束腰在光源位置
            )
            
            # 2. 创建系统
            system = SequentialOpticalSystem(source, grid_size=256)
            
            # 3. 添加凹面镜（焦距 100mm）
            system.add_surface(SphericalMirror(
                radius_of_curvature=200.0,  # mm，R = 2f
                thickness=150.0,            # mm，到采样面的距离
                semi_aperture=15.0,         # mm
            ))
            
            # 4. 添加采样面
            system.add_sampling_plane(distance=150.0, name="focus")
            
            # 5. 运行仿真
            results = system.run()
            
            # 6. 获取结果
            focus_result = results["focus"]
            print(f"焦点处光束半径: {focus_result.beam_radius:.3f} mm")
        
        使用混合传播模式::
        
            # 创建系统时启用混合传播
            system = SequentialOpticalSystem(
                source,
                grid_size=256,
                use_hybrid_propagation=True,
                hybrid_num_rays=100,
            )
        
        链式调用 - 构建多元件系统::
        
            system = (
                SequentialOpticalSystem(source)
                .add_surface(ThinLens(
                    focal_length_value=50.0,
                    thickness=100.0,
                    semi_aperture=10.0,
                ))
                .add_surface(ThinLens(
                    focal_length_value=-25.0,
                    thickness=50.0,
                    semi_aperture=10.0,
                ))
                .add_sampling_plane(distance=150.0, name="output")
            )
            results = system.run()
        
        ABCD 计算 - 获取任意位置的光束参数::
        
            # 获取距离 100mm 处的光束参数
            abcd_result = system.get_abcd_result(distance=100.0)
            print(f"光束半径: {abcd_result.w:.3f} mm")
            print(f"波前曲率半径: {abcd_result.R:.3f} mm")
        
        可视化 - 绘制光路图::
        
            # 绘制 2D 光路布局
            fig, ax = system.draw_layout(show=False)
            fig.savefig("layout.png")
        
        系统摘要::
        
            print(system.summary())
    
    验证需求:
        - Requirements 3.1-3.8: 系统构建
        - Requirements 5.1-5.7: 仿真执行
        - Requirements 6.1-6.5: ABCD 计算
        - Requirements 10.1-10.4: SequentialOpticalSystem 集成
    """
    
    def __init__(
        self,
        source: GaussianBeamSource,
        grid_size: int = 512,
        beam_ratio: float = 0.5,
        use_hybrid_propagation: bool = True,  # 默认开启混合传播模式
        hybrid_num_rays: int = 100,
    ) -> None:
        self._source = source
        self._grid_size = grid_size
        self._beam_ratio = beam_ratio
        self._use_hybrid_propagation = use_hybrid_propagation
        self._hybrid_num_rays = hybrid_num_rays
        self._elements: List = []
        self._sampling_planes: List[SamplingPlane] = []
        self._propagation_direction: int = 1
        self._current_path_length: float = 0.0
        self._current_z_position: float = 0.0
        
        # 光轴跟踪器，用于跟踪光轴在系统中的演变
        self._axis_tracker = OpticalAxisTracker()
    
    @property
    def source(self) -> GaussianBeamSource:
        return self._source
    
    @property
    def grid_size(self) -> int:
        return self._grid_size
    
    @property
    def beam_ratio(self) -> float:
        return self._beam_ratio
    
    @property
    def elements(self) -> List:
        return self._elements.copy()
    
    @property
    def sampling_planes(self) -> List[SamplingPlane]:
        return self._sampling_planes.copy()
    
    @property
    def total_path_length(self) -> float:
        return self._current_path_length
    
    @property
    def use_hybrid_propagation(self) -> bool:
        """是否使用混合传播模式"""
        return self._use_hybrid_propagation
    
    @property
    def axis_tracker(self) -> OpticalAxisTracker:
        """获取光轴跟踪器"""
        return self._axis_tracker

    def add_surface(self, element) -> "SequentialOpticalSystem":
        """添加光学面，支持链式调用"""
        element.z_position = self._current_z_position
        element.path_length = self._current_path_length
        self._elements.append(element)
        
        # 更新光轴跟踪器
        self._axis_tracker.add_element(element)
        
        thickness = element.thickness
        self._current_path_length += abs(thickness)
        
        if element.is_reflective:
            self._propagation_direction *= -1
        
        self._current_z_position += self._propagation_direction * thickness
        return self
    
    def add_sampling_plane(
        self,
        distance: float,
        name: Optional[str] = None,
    ) -> "SequentialOpticalSystem":
        """添加采样面，支持链式调用"""
        plane = SamplingPlane(distance=distance, name=name)
        self._sampling_planes.append(plane)
        return self
    
    def get_abcd_result(self, distance: float) -> ABCDResult:
        """获取指定距离的 ABCD 计算结果"""
        from gaussian_beam_simulation.abcd_calculator import ABCDCalculator
        
        beam = self._source.to_gaussian_beam()
        calculator = ABCDCalculator(beam, self._elements)
        
        result = calculator.propagate_distance(distance)
        
        return ABCDResult(
            distance=distance,
            w=result.w,
            R=result.R,
            z_position=result.z,
        )
    
    def _calculate_z_position(self, distance: float) -> float:
        """根据光程距离计算 z 位置"""
        z = 0.0
        path = 0.0
        direction = 1
        
        for element in self._elements:
            if path + element.thickness >= distance:
                remaining = distance - path
                z += direction * remaining
                return z
            
            path += element.thickness
            z += direction * element.thickness
            
            if element.is_reflective:
                direction *= -1
        
        remaining = distance - path
        z += direction * remaining
        return z

    def run(
        self,
        plot: bool = False,
        plot_mode: str = "spatial",
        save_plot: Optional[str] = None,
        show_plot: bool = True,
    ) -> SimulationResults:
        """执行仿真
        
        参数:
            plot: 是否自动绘制光路图，默认 False
            plot_mode: 绘图模式，"spatial"（空间坐标）或 "unfolded"（展开），默认 "spatial"
            save_plot: 保存图像的文件路径，如 "layout.png"，默认 None（不保存）
            show_plot: 是否显示图像，默认 True
        
        返回:
            SimulationResults 对象
        
        示例:
            # 简单运行
            results = system.run()
            
            # 运行并自动绘制空间光路图
            results = system.run(plot=True)
            
            # 运行并保存光路图
            results = system.run(plot=True, save_plot="my_system.png", show_plot=False)
        """
        try:
            results = self._run_simulation()
            
            # 自动绘图
            if plot:
                fig, ax = self.draw_layout(
                    mode=plot_mode,
                    show=show_plot,
                )
                if save_plot:
                    fig.savefig(save_plot, dpi=150, bbox_inches='tight')
                    if not show_plot:
                        import matplotlib.pyplot as plt
                        plt.close(fig)
            
            return results
        except Exception as e:
            if isinstance(e, SequentialSystemError):
                raise
            raise SimulationError(f"仿真执行失败：{str(e)}") from e
    
    def _run_simulation(self) -> SimulationResults:
        """内部仿真实现
        
        传播逻辑：
        1. 按光程距离排序所有采样面
        2. 遍历光学元件，在每个元件处：
           a. 先传播到元件位置
           b. 应用元件效果（透镜相位）
           c. 记录元件位置之后、下一元件之前的采样面
        3. 处理最后一个元件之后的采样面
        """
        import proper
        
        beam = self._source.to_gaussian_beam()
        wavelength_m = self._source.wavelength * 1e-6
        w0 = self._source.w0  # 束腰半径
        beam_diameter_m = 2 * w0 * 1e-3  # beam_diameter = 2 × w0（PROPER 固定用法）
        
        wfo = proper.prop_begin(
            beam_diameter_m, wavelength_m, self._grid_size, 0.5  # beam_diam_fraction = 0.5（PROPER 固定用法）
        )
        
        self._apply_initial_gaussian(wfo, beam)
        
        sampling_results: Dict[str, SamplingResult] = {}
        sorted_planes = sorted(self._sampling_planes, key=lambda p: p.distance)
        plane_index = 0
        current_path = 0.0  # 当前光程位置
        
        # 计算每个元件的光程位置
        element_positions = []
        pos = 0.0
        for element in self._elements:
            element_positions.append(pos)
            pos += element.thickness
        
        # 处理每个元件
        for elem_idx, element in enumerate(self._elements):
            elem_path = element_positions[elem_idx]
            next_elem_path = element_positions[elem_idx + 1] if elem_idx + 1 < len(self._elements) else float('inf')
            
            # 记录元件之前的采样面
            while plane_index < len(sorted_planes):
                plane = sorted_planes[plane_index]
                if plane.distance <= elem_path:
                    # 传播到采样面位置
                    prop_distance = plane.distance - current_path
                    if prop_distance > 0:
                        proper.prop_propagate(wfo, prop_distance * 1e-3)
                        current_path = plane.distance
                    
                    result = self._record_wavefront(wfo, plane)
                    key = plane.name if plane.name else f"plane_{plane_index}"
                    sampling_results[key] = result
                    plane.result = result
                    plane_index += 1
                else:
                    break
            
            # 传播到元件位置
            prop_distance = elem_path - current_path
            if prop_distance > 0:
                proper.prop_propagate(wfo, prop_distance * 1e-3)
                current_path = elem_path
            
            # 应用元件效果
            self._apply_element(wfo, element)
            
            # 记录元件之后、下一元件之前的采样面
            while plane_index < len(sorted_planes):
                plane = sorted_planes[plane_index]
                if plane.distance > elem_path and plane.distance <= next_elem_path:
                    # 传播到采样面位置
                    prop_distance = plane.distance - current_path
                    if prop_distance > 0:
                        proper.prop_propagate(wfo, prop_distance * 1e-3)
                        current_path = plane.distance
                    
                    result = self._record_wavefront(wfo, plane)
                    key = plane.name if plane.name else f"plane_{plane_index}"
                    sampling_results[key] = result
                    plane.result = result
                    plane_index += 1
                else:
                    break
        
        # 处理最后一个元件之后的采样面
        while plane_index < len(sorted_planes):
            plane = sorted_planes[plane_index]
            prop_distance = plane.distance - current_path
            if prop_distance > 0:
                proper.prop_propagate(wfo, prop_distance * 1e-3)
                current_path = plane.distance
            
            result = self._record_wavefront(wfo, plane)
            key = plane.name if plane.name else f"plane_{plane_index}"
            sampling_results[key] = result
            plane.result = result
            plane_index += 1
        
        return SimulationResults(
            sampling_results=sampling_results,
            source=self._source,
            surfaces=self._elements,
        )

    def _apply_initial_gaussian(self, wfo, beam) -> None:
        """应用初始高斯光束
        
        注意：PROPER 内部使用 FFT 坐标系，原点在 (0,0)。
        我们需要创建以中心为原点的高斯场，然后用 prop_shift_center 移到 FFT 原点。
        """
        import proper
        
        n = proper.prop_get_gridsize(wfo)
        sampling = proper.prop_get_sampling(wfo) * 1e3  # 转换为 mm
        
        # 创建以中心为原点的坐标网格
        half_size = sampling * n / 2
        coords = np.linspace(-half_size, half_size, n)
        X, Y = np.meshgrid(coords, coords)
        R_sq = X**2 + Y**2
        
        w_0 = beam.w(0.0)  # mm
        R_0 = beam.R(0.0)  # mm
        
        # 高斯振幅
        amplitude = np.exp(-R_sq / w_0**2)
        
        # 高斯相位（球面波前）
        if np.isinf(R_0):
            phase = np.zeros_like(R_sq)
        else:
            k = 2 * np.pi / (self._source.wavelength * 1e-3)  # 1/mm
            phase = -k * R_sq / (2 * R_0)
        
        # 创建高斯场（以中心为原点）
        gaussian_field_centered = amplitude * np.exp(1j * phase)
        
        # 将高斯场移到 FFT 原点（左上角），与 PROPER 内部坐标系匹配
        gaussian_field_fft = proper.prop_shift_center(gaussian_field_centered)
        
        # 应用到波前
        wfo.wfarr = wfo.wfarr * gaussian_field_fft
    
    def _apply_element(self, wfo, element) -> None:
        """应用光学元件效果
        
        支持两种模式：
        1. 纯 PROPER 模式（use_hybrid_propagation=False）：
           - 透镜/反射镜的聚焦效果（通过 prop_lens）
           - 倾斜效果（仅当 is_fold=False 时添加倾斜相位）
           - 偏心效果（通过坐标偏移）
        
        2. 混合传播模式（use_hybrid_propagation=True）：
           - 使用 HybridElementPropagator 进行波前-光线-波前重建
           - 更准确地处理元件处的波前变换
        
        关于倾斜的处理：
        - is_fold=False（默认）：使用完整光线追迹计算 OPD
          入射面和出射面垂直于各自的光轴，OPD 不包含整体倾斜
        - is_fold=True：跳过光线追迹，仅更新光轴方向
        
        参数:
            wfo: PROPER 波前对象
            element: 光学元件对象
        
        **Validates: Requirements 10.1, 10.2, 10.4**
        """
        import proper
        
        # 检查是否使用混合传播模式
        if self._use_hybrid_propagation:
            self._apply_element_hybrid(wfo, element)
            return
        
        # 以下是纯 PROPER 模式的实现
        n = proper.prop_get_gridsize(wfo)
        sampling = proper.prop_get_sampling(wfo)  # m/pixel
        sampling_mm = sampling * 1e3  # mm/pixel
        wavelength_m = wfo.lamda
        
        # 创建坐标网格（以中心为原点，单位 mm）
        half_size = sampling_mm * n / 2
        coords = np.linspace(-half_size, half_size, n)
        X, Y = np.meshgrid(coords, coords)
        
        # 1. 应用偏心效果（坐标偏移）
        if element.decenter_x != 0 or element.decenter_y != 0:
            X_shifted = X - element.decenter_x
            Y_shifted = Y - element.decenter_y
        else:
            X_shifted = X
            Y_shifted = Y
        
        # 2. 应用倾斜效果（仅当 is_fold=False 时）
        # 获取 is_fold 属性，默认为 False（使用光线追迹计算 OPD）
        is_fold = getattr(element, 'is_fold', False)
        
        if not is_fold and (element.tilt_x != 0 or element.tilt_y != 0):
            # 只有失调倾斜才引入波前倾斜
            # 倾斜引入的 OPD = x * sin(tilt_y) + y * sin(tilt_x)
            # 对于反射镜，OPD 加倍
            tilt_opd = (X_shifted * np.sin(element.tilt_y) + 
                        Y_shifted * np.sin(element.tilt_x))
            
            if element.is_reflective:
                tilt_opd *= 2  # 反射镜 OPD 加倍
            
            # 转换为相位（弧度）
            k = 2 * np.pi / (wavelength_m * 1e3)  # 1/mm
            tilt_phase = k * tilt_opd
            
            # 应用倾斜相位（需要移到 FFT 坐标系）
            tilt_field = np.exp(1j * tilt_phase)
            tilt_field_fft = proper.prop_shift_center(tilt_field)
            wfo.wfarr = wfo.wfarr * tilt_field_fft
        
        # 3. 离轴抛物面镜效应
        # 注意：对于轴上平行光入射的 OAP，不应添加彗差
        # 抛物面镜对轴上点源是无像差的，这是抛物面的定义特性
        # 彗差只在视场外点源时出现，当前仿真假设轴上平行光
        # 因此这里不添加离轴彗差
        
        # 4. 应用聚焦效果（透镜相位）
        if element.element_type == "thin_lens":
            proper.prop_lens(wfo, element.focal_length * 1e-3)
        else:
            # 反射镜和抛物面镜
            if not np.isinf(element.focal_length):
                proper.prop_lens(wfo, element.focal_length * 1e-3)
    
    def _apply_element_hybrid(self, wfo, element) -> None:
        """使用混合传播模式应用光学元件效果（重构版）
        
        本方法实现了混合光学仿真的核心功能：结合几何光线追迹（optiland）和
        物理光学传播（PROPER）来准确计算光学元件对波前的影响。
        
        核心设计原理：
        ==============
        
        1. OPD 计算完全依赖 ElementRaytracer（几何光线追迹）
           - 不使用 PROPER 的 prop_lens 函数计算相位
           - ElementRaytracer 提供真实的光程差（OPD），包含所有几何像差
           - 注意：ElementRaytracer 的 OPD 符号与 PROPER 相反，需要取反
           - 所有元件（包括平面镜、抛物面镜）都进行精确光线追迹
        
        2. 高斯光束参数更新
           - 使用 _update_gaussian_params_only 方法（复用 prop_lens 的参数更新逻辑）
           - 只更新高斯光束跟踪参数（z_w0, w0, z_Rayleigh 等）
           - 不修改波前数组 wfarr
           - 平面镜（焦距无穷大）不更新高斯光束参数
        
        3. 像差计算（参考面变换）
           - 像差 = 实际 OPD - 理想聚焦 OPD
           - 理想聚焦 OPD = r² / (2f) / λ（对于凹面镜，边缘光程长）
           - 对于平面镜（f = ∞），理想 OPD = 0
           - 这样计算的像差只包含元件引入的波前畸变，不包含理想聚焦效果
        
        与旧实现的区别：
        ================
        
        旧实现（纯 PROPER 模式）：
        - 使用 prop_lens 函数同时更新参数和应用相位
        - 相位计算基于理想薄透镜模型
        - 无法处理真实光学元件的像差
        
        新实现（混合传播模式）：
        - 参数更新和相位计算分离
        - 相位计算基于真实的几何光线追迹
        - 能够准确计算球差、彗差等几何像差
        - 支持复杂的光学元件（如离轴抛物面镜）
        - 所有元件都进行精确光线追迹，包括平面镜和抛物面镜
        
        处理流程：
        ==========
        
        1. 获取 SurfaceDefinition
           - 如果元件没有提供 SurfaceDefinition（如 ThinLens），使用 prop_lens
        
        2. 更新高斯光束参数（仅对有限焦距元件）
           - 调用 _update_gaussian_params_only 更新 PROPER 的光束跟踪参数
           - 这确保后续传播使用正确的参考球面
           - 平面镜不更新参数（焦距无穷大）
        
        3. 使用 ElementRaytracer 计算完整 OPD
           - 在整个采样面上创建均匀分布的采样光线
           - 追迹光线通过光学元件
           - 获取相对于主光线的 OPD（波长数）
        
        4. 计算像差并应用到波前
           - 像差 = 实际 OPD - 理想聚焦 OPD
           - 将像差转换为相位并插值到 PROPER 网格
           - 应用相位到波前数组
        
        参数:
            wfo: PROPER 波前对象，包含当前波前状态和高斯光束跟踪参数
            element: 光学元件对象，需要提供 get_surface_definition() 方法
        
        注意事项:
            - OPD 符号约定：ElementRaytracer 的 OPD 符号与 PROPER 相反，需要取反
            - 坐标单位：ElementRaytracer 使用 mm，PROPER 使用 m
            - 参考面更新时机：高斯光束参数必须在计算参考面相位之前更新
            - 大 OPD 值：对于短焦距元件，边缘 OPD 可能很大，需要正确处理
        
        **Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 10.1, 10.3**
        """
        import proper
        
        is_fold = getattr(element, 'is_fold', False)
        has_tilt = (element.tilt_x != 0 or element.tilt_y != 0)
        is_plane_mirror = np.isinf(element.focal_length)
        
        # 获取 SurfaceDefinition（用于 ElementRaytracer）
        surface_def = element.get_surface_definition()
        
        # 如果元件没有提供 SurfaceDefinition（如 ThinLens），使用 prop_lens
        # 这是向后兼容的处理方式
        if surface_def is None:
            if not is_plane_mirror:
                proper.prop_lens(wfo, element.focal_length * 1e-3)
            return
        
        # =====================================================================
        # 关于 is_fold 的处理：
        # 
        # is_fold=False（默认）：使用完整光线追迹计算 OPD
        # - 入射面和出射面垂直于各自的光轴
        # - OPD 不包含整体倾斜（因为参考面垂直于光轴）
        # - 适用于所有情况，包括大角度折叠镜
        # 
        # is_fold=True：跳过光线追迹中的倾斜处理
        # - 仅用于需要简化计算的特殊情况
        # - 使用不包含倾斜的 SurfaceDefinition 进行追迹
        # =====================================================================
        if is_fold and has_tilt:
            # 创建不包含倾斜的 SurfaceDefinition
            from wavefront_to_rays.element_raytracer import SurfaceDefinition
            surface_def_for_trace = SurfaceDefinition(
                surface_type=surface_def.surface_type,
                radius=surface_def.radius,
                thickness=surface_def.thickness,
                material=surface_def.material,
                semi_aperture=surface_def.semi_aperture,
                conic=surface_def.conic,
                tilt_x=0.0,  # 不包含倾斜
                tilt_y=0.0,
            )
        else:
            surface_def_for_trace = surface_def
        
        # =====================================================================
        # 步骤 1: 更新高斯光束参数（不修改 wfarr）
        # 
        # 这一步复用 PROPER prop_lens 的参数更新逻辑，更新：
        # - z_w0: 虚拟束腰位置（用于计算参考球面）
        # - w0: 束腰半径
        # - z_Rayleigh: 瑞利距离
        # - beam_type_old: 光束类型（INSIDE_ 或 OUTSIDE）
        # - reference_surface: 参考面类型（PLANAR 或 SPHERI）
        # 
        # 注意：此步骤不修改 wfarr，只更新跟踪参数
        # 平面镜（焦距无穷大）不更新高斯光束参数
        # =====================================================================
        if not is_plane_mirror:
            focal_length_m = element.focal_length * 1e-3
            self._update_gaussian_params_only(wfo, focal_length_m)
        
        # =====================================================================
        # 步骤 2: 使用 ElementRaytracer 计算完整 OPD
        # 
        # ElementRaytracer 使用 optiland 进行几何光线追迹，计算真实的 OPD
        # 包含所有几何像差（球差、彗差等）
        # 
        # 对于折叠倾斜，使用不包含倾斜的 surface_def_for_trace
        # =====================================================================
        # 延迟导入
        from wavefront_to_rays.element_raytracer import ElementRaytracer
        from optiland.rays import RealRays
        from scipy.interpolate import griddata
        
        # 获取波前参数
        n = proper.prop_get_gridsize(wfo)
        sampling_m = proper.prop_get_sampling(wfo)
        sampling_mm = sampling_m * 1e3
        wavelength_um = wfo.lamda * 1e6
        wavelength_mm = wfo.lamda * 1e3
        
        # =====================================================================
        # 采样范围和密度选择：
        # 
        # is_fold=False（默认）：使用完整光线追迹
        # - 入射面和出射面垂直于各自的光轴
        # - OPD 不包含整体倾斜
        # - 使用完整网格尺寸进行采样
        # 
        # is_fold=True：简化处理
        # - 使用完整网格尺寸，稀疏采样
        # =====================================================================
        if not is_fold and has_tilt:
            # 使用光束覆盖区域
            beam_radius_mm = self._source.w(0.0) * 1.5  # 1.5 倍光束腰半径
            element_aperture = surface_def.semi_aperture if surface_def.semi_aperture else 15.0
            half_size_mm = min(beam_radius_mm, element_aperture)
            
            # 对于 is_fold=False，使用更密集的采样
            # 采样点数量应该与 PROPER 网格大小相当，以避免插值导致的相位混叠
            # 但为了效率，限制最大采样点数
            n_rays_1d_dense = min(n // 4, 128)  # 最多 128x128 = 16384 光线
            use_dense_sampling = True
        else:
            # 使用完整网格尺寸
            half_size_mm = self._get_sampling_half_size_mm(wfo)
            use_dense_sampling = False
        
        # 创建采样光线
        if use_dense_sampling:
            # 使用密集采样（直接在规则网格上采样）
            ray_coords = np.linspace(-half_size_mm, half_size_mm, n_rays_1d_dense)
            ray_X, ray_Y = np.meshgrid(ray_coords, ray_coords)
            ray_x = ray_X.flatten()
            ray_y = ray_Y.flatten()
        else:
            ray_x, ray_y = self._create_sampling_rays(half_size_mm)
        n_rays = len(ray_x)
        
        if n_rays == 0:
            return
        
        # 创建平行光入射光线
        rays_in = RealRays(
            x=ray_x,
            y=ray_y,
            z=np.zeros(n_rays),
            L=np.zeros(n_rays),
            M=np.zeros(n_rays),
            N=np.ones(n_rays),
            intensity=np.ones(n_rays),
            wavelength=np.full(n_rays, wavelength_um),
        )
        
        # 光线追迹（使用不包含折叠倾斜的 surface_def_for_trace）
        raytracer = ElementRaytracer(
            surfaces=[surface_def_for_trace],
            wavelength=wavelength_um,
        )
        
        rays_out = raytracer.trace(rays_in)
        opd_waves = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask()
        
        # =====================================================================
        # 步骤 3: 计算像差（实际 OPD - 理想 OPD）
        # 
        # 像差计算原理：
        # - 实际 OPD：ElementRaytracer 计算的真实光程差（相对于主光线）
        # - 理想 OPD：使用精确的几何公式计算理想抛物面镜的 OPD
        # - 像差 = 实际 OPD - 理想 OPD
        # 
        # 核心设计原则：
        # - 入射面和出射面垂直于各自的光轴
        # - 因此 OPD 不包含整体波前倾斜
        # - 光线追迹计算的是从入射面到出射面的光程差
        # 
        # 对于平面镜（f = ∞），理想 OPD = 0
        # =====================================================================
        
        # =====================================================================
        # 检查是否为抛物面镜
        # 
        # 关键物理原理：
        # - 抛物面镜对轴上平行光入射是**无像差**的，这是抛物面的定义特性
        # - 无论倾斜角度如何（包括 45° 离轴使用），抛物面镜都不会引入像差
        # - 这与球面镜不同：球面镜倾斜会引入像散、彗差等
        # =====================================================================
        is_parabolic = abs(surface_def.conic + 1.0) < 1e-6  # conic = -1 表示抛物面
        
        if not is_fold and has_tilt and not is_parabolic:
            # =====================================================================
            # is_fold=False 的倾斜**球面**表面：使用差分方法计算像差
            # 
            # 物理意义：
            # - 对于球面镜，倾斜会引入像散和彗差
            # - 这是物理上正确的行为
            # 
            # 注意：此分支仅适用于球面镜（conic != -1）
            # 抛物面镜不会进入此分支，因为抛物面对轴上光是无像差的
            # 
            # 计算方法：
            # 1. 追迹不带倾斜的表面，获取参考 OPD
            # 2. 差分 OPD = 带倾斜的 OPD - 不带倾斜的 OPD
            # 3. 从差分 OPD 中去除波前倾斜分量（Zernike Z2, Z3）
            # 4. 剩余的就是真正的像差（像散、彗差等）
            # 
            # 重要说明：
            # - 波前倾斜由光束方向改变来表示，不应作为相位添加
            # - 只有真正的像差才应该作为相位添加到 PROPER 波前
            # - 对于平面镜，去除倾斜后像差应为 0
            # =====================================================================
            
            # 检查倾斜角度，如果大于 5°，发出提示信息
            import warnings
            max_tilt = max(abs(element.tilt_x), abs(element.tilt_y))
            if max_tilt > np.deg2rad(5.0):
                # 大角度倾斜的球面镜会引入真实的像差，这是物理上正确的行为
                pass  # 不再发出警告，因为 is_fold=False 是默认行为
            
            # 创建不带倾斜的 SurfaceDefinition
            from wavefront_to_rays.element_raytracer import SurfaceDefinition as SD
            surface_no_tilt = SD(
                surface_type=surface_def.surface_type,
                radius=surface_def.radius,
                thickness=surface_def.thickness,
                material=surface_def.material,
                semi_aperture=surface_def.semi_aperture,
                conic=surface_def.conic,
                tilt_x=0.0,
                tilt_y=0.0,
            )
            
            # 追迹不带倾斜的表面
            raytracer_ref = ElementRaytracer(
                surfaces=[surface_no_tilt],
                wavelength=wavelength_um,
            )
            
            # 创建相同的入射光线
            rays_in_ref = RealRays(
                x=ray_x.copy(),
                y=ray_y.copy(),
                z=np.zeros(n_rays),
                L=np.zeros(n_rays),
                M=np.zeros(n_rays),
                N=np.ones(n_rays),
                intensity=np.ones(n_rays),
                wavelength=np.full(n_rays, wavelength_um),
            )
            
            rays_out_ref = raytracer_ref.trace(rays_in_ref)
            opd_waves_ref = raytracer_ref.get_relative_opd_waves()
            valid_mask_ref = raytracer_ref.get_valid_ray_mask()
            
            # 像差 = 带倾斜的 OPD - 不带倾斜的 OPD
            # 注意：两者的主光线可能不同，需要对齐
            # 使用入射位置 (0, 0) 的光线作为参考
            center_idx = n_rays // 2  # 假设网格中心是 (0, 0)
            
            # 对齐到中心光线
            opd_waves_aligned = opd_waves - opd_waves[center_idx]
            opd_waves_ref_aligned = opd_waves_ref - opd_waves_ref[center_idx]
            
            # 计算差分 OPD
            diff_opd_waves = opd_waves_aligned - opd_waves_ref_aligned
            
            # 合并有效掩模
            valid_mask = valid_mask & valid_mask_ref
            
            # =====================================================================
            # 从差分 OPD 中去除波前倾斜分量
            # 
            # 波前倾斜由光束方向改变来表示，不应作为相位添加到 PROPER 波前。
            # 使用最小二乘法拟合并去除倾斜分量（Zernike Z2, Z3）。
            # =====================================================================
            valid_x = ray_x[valid_mask]
            valid_y = ray_y[valid_mask]
            valid_diff = diff_opd_waves[valid_mask]
            
            if len(valid_x) > 3:
                # 归一化坐标到 [-1, 1]
                max_r = max(np.max(np.abs(valid_x)), np.max(np.abs(valid_y)))
                if max_r > 0:
                    norm_x = valid_x / max_r
                    norm_y = valid_y / max_r
                else:
                    norm_x = valid_x
                    norm_y = valid_y
                
                # 最小二乘拟合：OPD = a0 + a1*x + a2*y
                A = np.column_stack([np.ones_like(norm_x), norm_x, norm_y])
                coeffs, _, _, _ = np.linalg.lstsq(A, valid_diff, rcond=None)
                
                # 计算倾斜分量
                tilt_component = coeffs[0] + coeffs[1] * (ray_x / max_r if max_r > 0 else ray_x) + \
                                 coeffs[2] * (ray_y / max_r if max_r > 0 else ray_y)
                
                # 去除倾斜，得到真正的像差
                aberration_waves = diff_opd_waves - tilt_component
            else:
                aberration_waves = diff_opd_waves
        else:
            # =====================================================================
            # is_fold=True 或无倾斜表面：使用理想 OPD 公式
            # =====================================================================
            ray_r_sq = ray_x**2 + ray_y**2
            
            if is_plane_mirror:
                # 平面镜：理想 OPD = 0
                ideal_opd_waves = np.zeros_like(ray_r_sq)
            else:
                # 使用精确公式计算理想抛物面镜的 OPD
                focal_length_mm = element.focal_length
                ideal_opd_mm = self._calculate_exact_mirror_opd(ray_r_sq, focal_length_mm)
                ideal_opd_waves = ideal_opd_mm / wavelength_mm
            
            # 像差 = 实际 OPD - 理想 OPD
            aberration_waves = opd_waves - ideal_opd_waves
        
        # 将无效光线的像差设为 0（避免插值问题）
        aberration_waves = np.where(valid_mask, aberration_waves, 0.0)
        
        # 检查像差是否足够小（用于验证光线追迹精度）
        valid_aberration = aberration_waves[valid_mask]
        if len(valid_aberration) > 0:
            aberration_rms = np.std(valid_aberration)
            aberration_pv = np.max(valid_aberration) - np.min(valid_aberration)
            
            # 如果像差很小（< 0.01 waves），说明光线追迹与理论公式一致
            # 这种情况下不需要应用像差相位
            if aberration_rms < 0.01:
                # 光线追迹精度验证通过，不应用像差
                return
        
        # =====================================================================
        # 步骤 4: 将像差转换为相位并应用
        # 
        # 相位转换公式：
        # - 像差相位 = -2π * 像差（波长数）
        # - 负号原因：正 OPD（光程长）对应负相位（波前滞后）
        # 
        # 插值说明：
        # - 使用三次插值（cubic）将稀疏采样点插值到 PROPER 网格
        # - 插值必须在 PROPER 网格的坐标系中进行
        # - 采样点范围外的区域填充为 0（不引入相位变化）
        # 
        # 重要修正（2024-01）：
        # 对于 is_fold=False 的情况，像差可能很大（多个波长），
        # 直接插值会导致边界处出现巨大的相位跳变。
        # 解决方案：使用 PROPER 波前的振幅作为掩模，只在有光的区域应用相位。
        # =====================================================================
        # 像差相位 = -2π * 像差（波长数）
        # 负号是因为：正 OPD（光程长）对应负相位（波前滞后）
        aberration_phase = -2 * np.pi * aberration_waves
        
        # 只使用有效光线进行插值
        valid_x = ray_x[valid_mask]
        valid_y = ray_y[valid_mask]
        valid_phase = aberration_phase[valid_mask]
        
        if len(valid_x) > 3:
            # 创建 PROPER 网格坐标（必须与 PROPER 波前的坐标系匹配）
            # PROPER 网格范围是 [-n/2 * sampling, n/2 * sampling]
            proper_half_size_mm = sampling_mm * n / 2
            coords_mm = np.linspace(-proper_half_size_mm, proper_half_size_mm, n)
            X_mm, Y_mm = np.meshgrid(coords_mm, coords_mm)
            
            # 插值
            # 注意：采样点范围是 [-half_size_mm, half_size_mm]
            # 使用 NaN 作为填充值，稍后用掩模处理
            points = np.column_stack([valid_x, valid_y])
            phase_grid = griddata(
                points,
                valid_phase,
                (X_mm, Y_mm),
                method='cubic',
                fill_value=np.nan,
            )
            
            # =====================================================================
            # 使用 PROPER 波前振幅作为掩模
            # 
            # 对于 is_fold=False 的大像差情况，采样范围外的区域不应该应用相位。
            # 使用波前振幅作为掩模，只在有光的区域应用相位。
            # 这避免了边界处的相位跳变问题。
            # 
            # 重要：使用采样范围作为掩模，而不是振幅阈值
            # 因为高斯光束的尾部延伸很远，振幅阈值掩模可能覆盖整个网格
            # =====================================================================
            # 创建采样范围掩模
            R_mm = np.sqrt(X_mm**2 + Y_mm**2)
            sampling_range_mask = R_mm <= half_size_mm
            
            # 在采样范围外，相位设为 0
            phase_grid = np.where(sampling_range_mask & ~np.isnan(phase_grid), phase_grid, 0.0)
            
            # 处理剩余的 NaN 值
            phase_grid = np.nan_to_num(phase_grid, nan=0.0)
            
            # 检查相位采样（只在采样范围内检查）
            phase_in_range = phase_grid.copy()
            phase_in_range[~sampling_range_mask] = np.nan
            self._check_phase_sampling(phase_in_range, sampling_mm)
            
            # 应用相位
            phase_field = np.exp(1j * phase_grid)
            phase_field_fft = proper.prop_shift_center(phase_field)
            wfo.wfarr = wfo.wfarr * phase_field_fft
    
    def _update_proper_gaussian_params(self, wfo, lens_fl: float) -> None:
        """更新 PROPER 的高斯光束跟踪参数（不修改 wfarr）
        
        本方法复用 PROPER prop_lens.py 中的算法，只更新高斯光束跟踪参数，
        不修改波前数组 wfarr。这是混合传播模式的核心设计之一。
        
        更新的参数：
        - z_w0: 虚拟束腰位置（用于计算参考球面曲率半径 R_ref = z - z_w0）
        - w0: 束腰半径
        - z_Rayleigh: 瑞利距离（z_R = π * w0² / λ）
        - beam_type_old: 光束类型
          - "INSIDE_": 在瑞利距离内，波前近似平面
          - "OUTSIDE": 在瑞利距离外，波前近似球面
        - reference_surface: 参考面类型
          - "PLANAR": 平面参考面（对应 INSIDE_ 光束）
          - "SPHERI": 球面参考面（对应 OUTSIDE 光束）
        - propagator_type: 传播器类型（用于选择传播算法）
        - current_fratio: 当前 F 数
        
        高斯光束变换原理：
        =================
        
        高斯光束通过薄透镜/反射镜的变换遵循 ABCD 矩阵法则。
        对于焦距为 f 的元件：
        
        1. 计算入射光束的曲率半径 R_in：
           R_in = (z - z_w0) + z_R² / (z - z_w0)
           
        2. 应用透镜变换：
           1/R_out = 1/R_in - 1/f
           
        3. 计算新的束腰位置和束腰半径：
           z_w0_new = z - R_out / (1 + (λ*R_out / (π*w²))²)
           w0_new = w / sqrt(1 + (π*w² / (λ*R_out))²)
        
        参数:
            wfo: PROPER 波前对象
            lens_fl: 透镜/反射镜焦距（单位：m）
                    - 正值：凹面镜/会聚透镜
                    - 负值：凸面镜/发散透镜
        
        注意：
            此函数不修改 wfarr，只更新高斯光束跟踪参数。
            这与 prop_lens 的区别在于：prop_lens 会同时更新参数和应用相位。
        
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
        """
        import proper
        
        rayleigh_factor = proper.rayleigh_factor
        
        # 计算当前表面处的束腰半径
        wfo.z_Rayleigh = np.pi * wfo.w0**2 / wfo.lamda
        w_at_surface = wfo.w0 * np.sqrt(1.0 + ((wfo.z - wfo.z_w0) / wfo.z_Rayleigh)**2)
        
        # 计算高斯光束曲率半径变换
        if (wfo.z - wfo.z_w0) != 0.0:
            # 透镜不在焦点或入瞳处
            gR_beam_old = (wfo.z - wfo.z_w0) + wfo.z_Rayleigh**2 / (wfo.z - wfo.z_w0)
            
            if gR_beam_old != lens_fl:
                gR_beam = 1.0 / (1.0 / gR_beam_old - 1.0 / lens_fl)
                gR_beam_inf = 0
            else:
                gR_beam_inf = 1
        else:
            # 在焦点或入瞳处，输入光束是平面的
            gR_beam = -lens_fl
            gR_beam_inf = 0
        
        # 更新束腰位置和束腰半径
        if not gR_beam_inf:
            wfo.z_w0 = -gR_beam / (1.0 + (wfo.lamda * gR_beam / (np.pi * w_at_surface**2))**2) + wfo.z
            wfo.w0 = w_at_surface / np.sqrt(1.0 + (np.pi * w_at_surface**2 / (wfo.lamda * gR_beam))**2)
        else:
            wfo.z_w0 = wfo.z
            wfo.w0 = w_at_surface  # 输出光束是平面的
        
        # 更新瑞利距离
        wfo.z_Rayleigh = np.pi * wfo.w0**2 / wfo.lamda
        
        # 确定新的光束类型
        if np.abs(wfo.z_w0 - wfo.z) < rayleigh_factor * wfo.z_Rayleigh:
            beam_type_new = "INSIDE_"
        else:
            beam_type_new = "OUTSIDE"
        
        # 更新传播器类型
        wfo.propagator_type = wfo.beam_type_old + "_to_" + beam_type_new
        
        # 更新参考面类型
        if beam_type_new == "INSIDE_":
            wfo.reference_surface = "PLANAR"
        else:
            wfo.reference_surface = "SPHERI"
        
        # 更新光束类型
        wfo.beam_type_old = beam_type_new
        
        # 更新当前 F 数
        wfo.current_fratio = np.abs(wfo.z_w0 - wfo.z) / (2.0 * w_at_surface)
    
    # 为 _update_proper_gaussian_params 创建别名，符合设计文档命名
    _update_gaussian_params_only = _update_proper_gaussian_params
    
    def _compute_reference_phase(
        self,
        wfo,
        x_mm: NDArray,
        y_mm: NDArray,
    ) -> NDArray:
        """计算参考面相位
        
        根据 PROPER 的参考面类型计算参考相位。这是混合传播模式中
        正确处理参考面变换的关键方法。
        
        参考面类型说明：
        ================
        
        PROPER 使用参考球面跟踪机制来处理高斯光束的传播：
        
        1. PLANAR 参考面（平面参考）
           - 当光束在瑞利距离内（beam_type = "INSIDE_"）时使用
           - 波前近似平面，参考相位为零
           - 返回值：全零数组
        
        2. SPHERI 参考面（球面参考）
           - 当光束在瑞利距离外（beam_type = "OUTSIDE"）时使用
           - 波前近似球面，参考球面曲率半径 R_ref = z - z_w0
           - 参考相位公式：φ_ref = +k * r² / (2 * R_ref)（正号！）
           - 其中 k = 2π/λ 是波数，r 是到光轴的距离
        
        物理意义：
        ==========
        
        参考相位表示理想高斯光束在当前位置的波前曲率。
        
        数据流：
        - 写入 PROPER (SPHERI): wfarr = 仿真复振幅 × exp(-i × φ_ref)
        - 读取 PROPER (SPHERI): 完整相位 = PROPER相位 + φ_ref
        
        参数:
            wfo: PROPER 波前对象，包含参考面类型和束腰位置信息
            x_mm: X 坐标数组（单位：mm）
            y_mm: Y 坐标数组（单位：mm）
        
        返回:
            参考面相位（弧度），与输入坐标数组形状相同
        
        **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
        """
        wavelength_m = wfo.lamda
        k = 2 * np.pi / wavelength_m  # 波数（1/m）
        
        # 转换坐标到米
        x_m = x_mm * 1e-3
        y_m = y_mm * 1e-3
        r_sq_m = x_m**2 + y_m**2
        
        if wfo.reference_surface == "PLANAR":
            # 平面参考面，参考相位为零
            return np.zeros_like(r_sq_m)
        else:
            # 球面参考面
            R_ref_m = wfo.z - wfo.z_w0
            
            if abs(R_ref_m) < 1e-10:
                # 参考球面曲率半径接近零，视为平面
                return np.zeros_like(r_sq_m)
            
            # 参考球面相位：φ_ref = +k * r² / (2 * R_ref)（正号！）
            phase_ref = k * r_sq_m / (2 * R_ref_m)
            return phase_ref
    
    def _compute_theory_curvature_phase(
        self,
        wfo,
        element,
    ) -> NDArray:
        """计算理论曲率相位
        
        计算光学元件的理论聚焦相位（不含倾斜分量）。此方法用于在像差复振幅
        重建后加回元件的理论聚焦效果。
        
        公式（需求 3.1）：
            φ_theory = -k × r² / (2f)
            
        其中：
            k = 2π/λ 为波数
            r² = x² + y² 为到光轴的距离平方
            f 为焦距
        
        物理意义：
        ==========
        
        理论曲率相位表示理想薄透镜/反射镜对波前的聚焦效果。
        - 凹面镜/会聚透镜（f > 0）：边缘相位为负（波前滞后）
        - 凸面镜/发散透镜（f < 0）：边缘相位为正（波前超前）
        
        与像差的关系：
        - 实际波前 = 理论曲率相位 + 像差相位
        - 在混合传播模式中，我们先计算像差（实际 OPD - 理想 OPD）
        - 然后加回理论曲率相位，得到完整的波前效果
        
        注意事项：
        ==========
        
        1. 此相位不包含倾斜分量（需求 3.3）
           - 入射面和出射面垂直于各自的光轴，不存在波前倾斜
           - is_fold=True 时跳过倾斜处理
           - is_fold=False 时（默认）使用完整光线追迹
        
        2. 对于平面镜（f = ∞），理论相位为 0（需求 3.2）
           - 平面镜不改变波前曲率
           - 返回全零数组
        
        3. 相位单位为弧度
        
        参数:
            wfo: PROPER 波前对象，用于获取网格大小、采样间隔和波长
            element: 光学元件对象，需要提供 focal_length 属性（单位：mm）
        
        返回:
            理论曲率相位网格（弧度），形状为 (n, n)
        
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
        """
        import proper
        
        # 获取网格参数
        n = proper.prop_get_gridsize(wfo)
        sampling_m = proper.prop_get_sampling(wfo)  # m/pixel
        wavelength_m = wfo.lamda  # m
        
        # 创建坐标网格（单位：m）
        half_size_m = sampling_m * n / 2
        coords_m = np.linspace(-half_size_m, half_size_m, n)
        X_m, Y_m = np.meshgrid(coords_m, coords_m)
        r_sq_m = X_m**2 + Y_m**2
        
        # 检查是否为平面镜（需求 3.2）
        if np.isinf(element.focal_length):
            return np.zeros((n, n))
        
        # 计算理论相位（需求 3.1）
        # 公式：φ_theory = -k × r² / (2f)
        focal_length_m = element.focal_length * 1e-3  # mm → m
        k = 2 * np.pi / wavelength_m  # 波数（1/m）
        
        # 理论曲率相位
        theory_phase = -k * r_sq_m / (2 * focal_length_m)
        
        return theory_phase
    
    def _check_phase_sampling(self, phase_grid: NDArray, sampling_mm: float) -> None:
        """检查相位采样是否充足
        
        如果相邻像素间相位差超过 π，发出警告。
        
        参数:
            phase_grid: 相位网格（弧度）
            sampling_mm: 采样间隔（mm/pixel）
        
        **Validates: Requirements 7.1, 7.2, 7.3**
        """
        import warnings
        
        # 计算相位梯度
        grad_x = np.diff(phase_grid, axis=1)
        grad_y = np.diff(phase_grid, axis=0)
        
        max_grad_x = np.nanmax(np.abs(grad_x))
        max_grad_y = np.nanmax(np.abs(grad_y))
        max_grad = max(max_grad_x, max_grad_y)
        
        if max_grad > np.pi:
            warnings.warn(
                f"相位采样不足：相邻像素间最大相位差为 {max_grad:.2f} 弧度 "
                f"（超过 π = {np.pi:.2f}）。\n"
                f"建议：增加网格大小或减小光束尺寸。",
                UserWarning,
            )
    
    def _get_sampling_half_size_mm(self, wfo) -> float:
        """获取采样面半尺寸
        
        使用 PROPER 网格的完整尺寸作为采样范围，
        避免基于光束强度计算导致的面积收缩问题。
        
        参数:
            wfo: PROPER 波前对象
        
        返回:
            采样面半尺寸（mm）
        
        **Validates: Requirements 1.1**
        """
        import proper
        
        n = proper.prop_get_gridsize(wfo)
        sampling_m = proper.prop_get_sampling(wfo)
        sampling_mm = sampling_m * 1e3
        
        # 使用完整网格尺寸
        half_size_mm = sampling_mm * n / 2
        
        return half_size_mm
    
    def _create_sampling_rays(
        self,
        half_size_mm: float,
    ) -> Tuple[NDArray, NDArray]:
        """创建采样光线
        
        在整个采样面上创建均匀分布的采样点。
        不基于光束强度限制采样范围，避免面积收缩。
        
        参数:
            half_size_mm: 采样面半尺寸（mm）
        
        返回:
            (ray_x, ray_y): 光线位置数组（mm）
        
        **Validates: Requirements 1.1**
        """
        n_rays_1d = int(np.sqrt(self._hybrid_num_rays))
        ray_coords = np.linspace(-half_size_mm, half_size_mm, n_rays_1d)
        ray_X, ray_Y = np.meshgrid(ray_coords, ray_coords)
        ray_x = ray_X.flatten()
        ray_y = ray_Y.flatten()
        
        return ray_x, ray_y
    
    def _calculate_exact_mirror_opd(
        self,
        r_sq: NDArray,
        focal_length_mm: float,
    ) -> NDArray:
        """计算反射镜的精确 OPD（相对于中心光线）
        
        使用精确的几何公式计算抛物面反射镜的 OPD，而不是近似公式 r²/(2f)。
        
        精确公式推导：
        ==============
        
        对于抛物面反射镜，表面矢高为：
            sag = r² / (4f)
        
        平行光（沿 +z 方向）入射后反射，光程计算：
        1. 入射光程：从 z=0 到表面 = sag（带符号）
        2. 反射光程：从表面回到 z=0 = -sag/rz（带符号）
        
        反射方向由反射定律决定：
        - 表面法向量：n = (-x/(2f), -y/(2f), 1) / |n|
        - 归一化因子：|n| = sqrt(1 + r²/(4f²))
        - 入射方向：d = (0, 0, 1)
        - 反射方向 z 分量：rz = 1 - 2/|n|²
        
        总光程 = sag + (-sag/rz) = sag * (1 - 1/rz)
        相对 OPD = 总光程（中心光程 = 0，因为 r=0 时 sag=0）
        
        符号约定（与 PROPER prop_lens 一致）：
        - 凹面镜（f > 0）：边缘 OPD > 0（边缘光程长）
        - 凸面镜（f < 0）：边缘 OPD < 0（边缘光程短）
        
        与近似公式的比较：
        - 近似公式：OPD ≈ r²/(2f)
        - 精确公式在大孔径时更准确
        - 小孔径（r/f < 0.1）时差异 < 0.3%
        - 大孔径（r/f ≈ 0.4）时差异可达 4%
        
        参数:
            r_sq: 到光轴距离的平方（mm²），可以是标量或数组
            focal_length_mm: 焦距（mm），正值为凹面镜，负值为凸面镜
        
        返回:
            OPD（mm），相对于中心光线
        
        **Validates: Requirements 1.1, 1.3**
        """
        f = focal_length_mm
        
        # 表面矢高（带符号）
        sag = r_sq / (4 * f)
        
        # 归一化因子的平方
        n_mag_sq = 1 + r_sq / (4 * f**2)
        
        # 反射方向 z 分量
        rz = 1 - 2 / n_mag_sq
        
        # 入射光程（带符号）
        incident_path = sag
        
        # 反射光程（带符号）
        reflected_path = -sag / rz
        
        # 总光程 = 相对 OPD（中心光程 = 0）
        opd = incident_path + reflected_path
        
        return opd
    
    def _apply_tilt_phase(self, wfo, element) -> None:
        """应用失调倾斜相位
        
        参数:
            wfo: PROPER 波前对象
            element: 光学元件对象
        """
        import proper
        
        n = proper.prop_get_gridsize(wfo)
        sampling_mm = proper.prop_get_sampling(wfo) * 1e3
        wavelength_m = wfo.lamda
        
        half_size = sampling_mm * n / 2
        coords = np.linspace(-half_size, half_size, n)
        X, Y = np.meshgrid(coords, coords)
        
        # 倾斜引入的 OPD
        tilt_opd = (X * np.sin(element.tilt_y) + Y * np.sin(element.tilt_x))
        if element.is_reflective:
            tilt_opd *= 2  # 反射镜 OPD 加倍
        
        k = 2 * np.pi / (wavelength_m * 1e3)  # 1/mm
        tilt_phase = k * tilt_opd
        
        tilt_field = np.exp(1j * tilt_phase)
        tilt_field_fft = proper.prop_shift_center(tilt_field)
        wfo.wfarr = wfo.wfarr * tilt_field_fft
    
    def _record_wavefront(self, wfo, plane: SamplingPlane) -> SamplingResult:
        """记录波前数据
        
        PROPER 内部使用参考球面跟踪机制：
        - reference_surface == "SPHERI"：球面参考，存储的是相对于参考球面的相位偏差
        - 参考球面会自动跟踪理想高斯光束的波前曲率
        - prop_lens() 会更新虚拟束腰位置 z_w0，从而更新参考球面
        
        对于理想高斯光束通过理想光学元件：
        - PROPER 存储的相对相位（相对于参考球面的偏差）应该接近零
        - 这个相对相位正是我们关心的"波前误差"（WFE）
        
        本方法直接使用 PROPER 存储的相对相位，不进行参考球面补偿。
        这样做的好处：
        1. 波前误差（WFE）直接可读
        2. 避免了 z_w0 在透镜后被修改导致的补偿错误
        3. 与 PROPER 的设计理念一致
        
        如果需要绝对相位（包含波前曲率），可以使用 ABCD 矩阵计算理论曲率半径，
        然后加上相应的二次相位项。
        
        同时记录采样面的光轴状态（位置和方向）。
        """
        import proper
        
        # 获取 PROPER 内部的振幅和相位（相对于参考球面）
        amplitude = proper.prop_get_amplitude(wfo).copy()
        phase_relative = proper.prop_get_phase(wfo).copy()
        sampling_m = proper.prop_get_sampling(wfo)  # m/pixel
        sampling_mm = sampling_m * 1e3  # mm/pixel
        
        # 直接使用 PROPER 存储的相对相位
        # 这是相对于参考球面的偏差，对于理想系统应该接近零
        # 这正是我们关心的波前误差（WFE）
        phase = phase_relative
        
        # 重建复振幅（使用相对相位）
        wavefront = amplitude * np.exp(1j * phase)
        
        # 计算光束半径
        beam_radius = self._calculate_beam_radius(amplitude, sampling_mm)
        z_position = self._calculate_z_position(plane.distance)
        
        # 获取采样面的光轴状态
        axis_state = self._axis_tracker.get_state_at_distance(plane.distance)
        
        # 记录 PROPER 内部的参考球面信息（用于调试和高级分析）
        # R_ref = z - z_w0 是 PROPER 跟踪的参考球面曲率半径
        R_ref_m = wfo.z - wfo.z_w0  # 单位：m
        
        return SamplingResult(
            distance=plane.distance,
            z_position=z_position,
            wavefront=wavefront,
            sampling=sampling_mm,
            beam_radius=beam_radius,
            name=plane.name,
            _wavelength=self._source.wavelength,
            axis_state=axis_state,
            _reference_radius=R_ref_m * 1e3 if abs(R_ref_m) > 1e-10 else float('inf'),  # mm
        )
    
    def _calculate_beam_radius(self, amplitude: NDArray, sampling: float) -> float:
        """从振幅分布计算光束半径"""
        intensity = amplitude**2
        total = np.sum(intensity)
        
        if total < 1e-15:
            return 0.0
        
        n = amplitude.shape[0]
        half_size = sampling * n / 2
        coords = np.linspace(-half_size, half_size, n)
        X, Y = np.meshgrid(coords, coords)
        
        x_var = np.sum(X**2 * intensity) / total
        y_var = np.sum(Y**2 * intensity) / total
        
        return float(np.sqrt(2 * (x_var + y_var)))

    def summary(self) -> str:
        """返回系统配置摘要"""
        lines = [
            "序列光学系统配置摘要",
            "=" * 40,
            "",
            "光源参数:",
            f"  波长: {self._source.wavelength} μm",
            f"  束腰半径: {self._source.w0} mm",
            f"  束腰位置: {self._source.z0} mm",
            f"  M² 因子: {self._source.m2}",
            f"  瑞利距离: {self._source.zR:.2f} mm",
            "",
            "仿真参数:",
            f"  网格大小: {self._grid_size}",
            f"  beam_ratio: {self._beam_ratio}",
            "",
            f"光学元件 ({len(self._elements)} 个):",
        ]
        
        for i, elem in enumerate(self._elements):
            # 基本信息
            line = f"  {i+1}. {elem.element_type}: f={elem.focal_length}mm, thickness={elem.thickness}mm"
            
            # 倾斜信息
            if elem.tilt_x != 0 or elem.tilt_y != 0:
                tilt_x_deg = np.degrees(elem.tilt_x)
                tilt_y_deg = np.degrees(elem.tilt_y)
                if elem.tilt_x != 0 and elem.tilt_y != 0:
                    line += f", tilt=({tilt_x_deg:.1f}°, {tilt_y_deg:.1f}°)"
                elif elem.tilt_x != 0:
                    line += f", tilt_x={tilt_x_deg:.1f}°"
                else:
                    line += f", tilt_y={tilt_y_deg:.1f}°"
            
            # 偏心信息
            if elem.decenter_x != 0 or elem.decenter_y != 0:
                line += f", decenter=({elem.decenter_x}, {elem.decenter_y})mm"
            
            lines.append(line)
        
        lines.extend([
            "",
            f"采样面 ({len(self._sampling_planes)} 个):",
        ])
        
        for i, plane in enumerate(self._sampling_planes):
            name = plane.name if plane.name else f"plane_{i}"
            lines.append(f"  {i+1}. {name}: distance={plane.distance}mm")
        
        lines.extend([
            "",
            f"总光程: {self._current_path_length:.2f} mm",
        ])
        
        return "\n".join(lines)
    
    def draw_layout(
        self,
        show: bool = True,
        figsize: Tuple[float, float] = (12, 6),
        mode: str = "spatial",
    ) -> Tuple["Figure", "Axes"]:
        """绘制 2D 光路图
        
        参数:
            show: 是否显示图形，默认 True
            figsize: 图形大小，默认 (12, 6)
            mode: 绘图模式
                - "spatial": 空间坐标模式，显示真实的折叠光路（默认）
                - "unfolded": 展开模式，沿光程距离展开
        
        返回:
            (fig, ax) 元组
        
        示例:
            # 绘制空间光路图
            fig, ax = system.draw_layout()
            
            # 绘制展开光路图
            fig, ax = system.draw_layout(mode="unfolded")
            
            # 保存图像
            fig, ax = system.draw_layout(show=False)
            fig.savefig("layout.png")
        """
        from .visualization import LayoutVisualizer
        
        visualizer = LayoutVisualizer(self)
        return visualizer.draw(figsize=figsize, show=show, mode=mode)
