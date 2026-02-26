"""
直接传播模式 (Direct Propagation Mode)

本模块实现理论上正确的元件处光传播仿真流程：
1. 入射复振幅在入射面被采样为光线（相位与强度）
2. 光线按照几何光线追迹，传播至真实面型，再传播至出射面
3. 在出射面，利用光线数据重建复振幅分布

定义：
- 入射面：元件顶点，垂直于入射光主光轴
- 出射面：元件顶点，垂直于出射光主光轴
- 切平面：元件顶点，与元器件表面相切（本模式不使用）

与原有 HybridElementPropagator 的区别：
- 原有模式：入射面 → [Tilted ASM] → 切平面 → 采样 → 追迹 → 重建 → [Tilted ASM] → 出射面
- 本模式：入射面 → 采样 → 追迹 → 出射面重建（无 Tilted ASM 步骤）

作者：混合光学仿真项目
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import warnings
import sys

# 添加项目路径
sys.path.insert(0, 'src')

from wavefront_to_rays.wavefront_sampler import WavefrontToRaysSampler
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from sequential_system.exceptions import SimulationError

from .pilot_beam import PilotBeamCalculator
from .phase_correction import PhaseCorrector
from .amplitude_reconstruction import AmplitudeReconstructor
from . import PilotBeamValidationResult, ValidationResult


@dataclass
class DirectPropagationResult:
    """直接传播结果
    
    属性:
        output_amplitude: 出射面复振幅
        exit_direction: 出射主光线方向 (L, M, N)
        exit_position: 出射面中心位置 (x, y, z)，单位 mm
        validation_result: Pilot Beam 验证结果（可选）
        intermediate: 中间结果字典（调试用）
    """
    output_amplitude: NDArray[np.complexfloating]
    exit_direction: Tuple[float, float, float]
    exit_position: Tuple[float, float, float]
    validation_result: Optional[PilotBeamValidationResult] = None
    intermediate: Optional[Dict[str, Any]] = None


class DirectElementPropagator:
    """直接元件传播器
    
    实现理论上正确的元件处光传播仿真流程：
    1. 在入射面直接采样复振幅为光线
    2. 光线追迹经过真实面型到出射面
    3. 在出射面重建复振幅
    
    参数:
        complex_amplitude: 入射面复振幅数组，形状 (N, N)
        element: 光学元件定义 (SurfaceDefinition)
        wavelength: 波长，单位 μm
        physical_size: 波前物理尺寸（直径），单位 mm
        grid_size: 输出网格大小，默认与输入相同
        num_rays: 采样光线数量，默认 100
        entrance_direction: 入射主光线方向 (L, M, N)，默认 (0, 0, 1)
        entrance_position: 入射面中心位置 (x, y, z)，单位 mm，默认 (0, 0, 0)
        use_pilot_beam: 是否使用 Pilot Beam 参考相位，默认 True
        pilot_beam_method: Pilot Beam 方法，'analytical' 或 'proper'
        debug: 是否启用调试模式
    
    示例:
        >>> import numpy as np
        >>> from wavefront_to_rays import SurfaceDefinition
        >>> 
        >>> # 创建输入复振幅（平面波）
        >>> grid_size = 64
        >>> amplitude = np.ones((grid_size, grid_size), dtype=complex)
        >>> 
        >>> # 定义光学元件（平面镜）
        >>> element = SurfaceDefinition(
        ...     surface_type='mirror',
        ...     radius=np.inf,
        ...     tilt_x=0.0,
        ...     tilt_y=0.0,
        ...     semi_aperture=25.0,
        ... )
        >>> 
        >>> # 创建传播器并执行传播
        >>> propagator = DirectElementPropagator(
        ...     complex_amplitude=amplitude,
        ...     element=element,
        ...     wavelength=0.633,
        ...     physical_size=50.0,
        ... )
        >>> result = propagator.propagate()
        >>> output_amplitude = result.output_amplitude
    """
    
    def __init__(
        self,
        complex_amplitude: NDArray[np.complexfloating],
        element: SurfaceDefinition,
        wavelength: float,
        physical_size: float,
        grid_size: Optional[int] = None,
        num_rays: int = 100,
        entrance_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        entrance_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        use_pilot_beam: bool = True,
        pilot_beam_method: str = 'analytical',
        debug: bool = False,
    ):
        """初始化直接元件传播器"""
        # 输入验证
        self._validate_inputs(
            complex_amplitude, element, wavelength, physical_size,
            grid_size, num_rays
        )
        
        # 存储参数
        self.complex_amplitude = complex_amplitude
        self.element = element
        self.wavelength = wavelength
        self.physical_size = physical_size
        self.grid_size = grid_size if grid_size is not None else complex_amplitude.shape[0]
        self.num_rays = num_rays
        self.entrance_direction = entrance_direction
        self.entrance_position = entrance_position
        self.use_pilot_beam = use_pilot_beam
        self.pilot_beam_method = pilot_beam_method
        self.debug = debug
        
        # 计算采样间隔
        self.dx = physical_size / complex_amplitude.shape[0]
        self.dy = physical_size / complex_amplitude.shape[1]
        
        # 波长转换为 mm
        self.wavelength_mm = wavelength * 1e-3
        
        # 中间结果存储
        self._intermediate_results: Dict[str, Any] = {}
        
        if self.debug:
            print(f"[DEBUG] DirectElementPropagator 初始化完成")
            print(f"  - 输入网格大小: {complex_amplitude.shape}")
            print(f"  - 输出网格大小: {self.grid_size}")
            print(f"  - 波长: {wavelength} μm")
            print(f"  - 物理尺寸: {physical_size} mm")
            print(f"  - 光线数量: {num_rays}")
            print(f"  - 使用 Pilot Beam: {use_pilot_beam}")
    
    def _validate_inputs(
        self,
        complex_amplitude: NDArray,
        element: SurfaceDefinition,
        wavelength: float,
        physical_size: float,
        grid_size: Optional[int],
        num_rays: int,
    ) -> None:
        """验证输入参数"""
        # 检查复振幅数组
        if complex_amplitude.ndim != 2:
            raise ValueError(
                f"复振幅数组必须是 2D 数组，当前维度: {complex_amplitude.ndim}"
            )
        
        if complex_amplitude.shape[0] != complex_amplitude.shape[1]:
            raise ValueError(
                f"复振幅数组必须是正方形，当前形状: {complex_amplitude.shape}"
            )
        
        # 检查波长
        if wavelength <= 0:
            raise ValueError(f"波长必须为正数，当前值: {wavelength}")
        
        # 检查物理尺寸
        if physical_size <= 0:
            raise ValueError(f"物理尺寸必须为正数，当前值: {physical_size}")
        
        # 检查网格大小
        if grid_size is not None and grid_size <= 0:
            raise ValueError(f"网格大小必须为正数，当前值: {grid_size}")
        
        # 检查光线数量
        if num_rays <= 0:
            raise ValueError(f"光线数量必须为正数，当前值: {num_rays}")
        
        # 检查元件定义
        if element is None:
            raise ValueError("元件定义不能为 None")
    
    def propagate(self) -> DirectPropagationResult:
        """执行直接传播
        
        流程：
        1. 在入射面采样复振幅为光线
        2. 光线追迹经过元件
        3. 在出射面重建复振幅
        
        返回:
            DirectPropagationResult 对象，包含出射面复振幅和相关信息
        """
        if self.debug:
            print("[DEBUG] 开始直接传播...")
        
        # =====================================================================
        # 步骤 1: 在入射面采样复振幅为光线
        # =====================================================================
        if self.debug:
            print("  [1] 在入射面采样复振幅为光线...")
        
        sampler = WavefrontToRaysSampler(
            wavefront_amplitude=self.complex_amplitude,
            wavelength=self.wavelength,
            physical_size=self.physical_size,
            num_rays=self.num_rays,
        )
        
        rays_in = sampler.get_output_rays()
        ray_x, ray_y = sampler.get_ray_positions()
        ray_amplitudes = np.sqrt(np.asarray(rays_in.i))  # 强度转振幅
        
        # 获取入射光线的相位（从波前采样得到）
        input_opd_waves = sampler.get_ray_opd()
        
        if self.debug:
            print(f"      采样光线数量: {len(ray_x)}")
            print(f"      入射能量: {np.sum(np.abs(self.complex_amplitude)**2):.6f}")
        
        self._intermediate_results['rays_in'] = rays_in
        self._intermediate_results['ray_positions'] = (ray_x, ray_y)
        self._intermediate_results['ray_amplitudes'] = ray_amplitudes
        self._intermediate_results['input_opd_waves'] = input_opd_waves
        
        # =====================================================================
        # 步骤 2: 光线追迹经过元件
        # =====================================================================
        if self.debug:
            print("  [2] 光线追迹经过元件...")
        
        raytracer = ElementRaytracer(
            surfaces=[self.element],
            wavelength=self.wavelength,
            chief_ray_direction=self.entrance_direction,
            entrance_position=self.entrance_position,
        )
        
        rays_out = raytracer.trace(rays_in)
        element_opd_waves = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask()
        
        # 获取出射主光线方向
        exit_direction = raytracer.get_exit_chief_ray_direction()
        
        if self.debug:
            valid_count = np.sum(valid_mask)
            print(f"      有效光线数量: {valid_count}/{len(valid_mask)}")
            print(f"      出射主光线方向: {exit_direction}")
        
        # 检查是否所有光线都无效
        if not np.any(valid_mask):
            raise SimulationError(
                "所有光线都无效，无法完成传播。"
                "可能的原因：元件半口径过小、光束尺寸过大、或元件位置不正确。"
            )
        
        self._intermediate_results['rays_out'] = rays_out
        self._intermediate_results['element_opd_waves'] = element_opd_waves
        self._intermediate_results['valid_mask'] = valid_mask
        self._intermediate_results['exit_direction'] = exit_direction
        
        # =====================================================================
        # 步骤 3: 计算总 OPD（入射相位 + 元件 OPD）
        # =====================================================================
        if self.debug:
            print("  [3] 计算总 OPD...")
        
        # 总 OPD = 入射波前 OPD + 元件引入的 OPD
        total_opd_waves = input_opd_waves + element_opd_waves
        
        self._intermediate_results['total_opd_waves'] = total_opd_waves
        
        # =====================================================================
        # 步骤 4: Pilot Beam 参考相位（可选）
        # =====================================================================
        reference_phase = None
        validation_result = None
        corrected_opd_waves = total_opd_waves
        
        if self.use_pilot_beam:
            if self.debug:
                print("  [4] 计算 Pilot Beam 参考相位...")
            
            reference_phase, validation_result, corrected_opd_waves = \
                self._apply_pilot_beam_correction(
                    ray_x, ray_y, total_opd_waves, valid_mask
                )
            
            self._intermediate_results['reference_phase'] = reference_phase
            self._intermediate_results['corrected_opd_waves'] = corrected_opd_waves
        else:
            if self.debug:
                print("  [4] 跳过 Pilot Beam（未启用）")
            # 不使用 Pilot Beam 时，参考相位为零
            reference_phase = np.zeros((self.grid_size, self.grid_size))
            corrected_opd_waves = total_opd_waves
        
        # =====================================================================
        # 步骤 5: 在出射面重建复振幅
        # =====================================================================
        if self.debug:
            print("  [5] 在出射面重建复振幅...")
        
        # 获取出射光线位置（在出射面局部坐标系中）
        exit_ray_x = np.asarray(rays_out.x)
        exit_ray_y = np.asarray(rays_out.y)
        
        # 获取出射光线强度
        exit_ray_intensity = np.asarray(rays_out.i)
        
        # 创建复振幅重建器
        reconstructor = AmplitudeReconstructor(
            grid_size=self.grid_size,
            physical_size=self.physical_size,
            wavelength=self.wavelength,
        )
        
        # 重建复振幅
        output_amplitude = reconstructor.reconstruct(
            ray_x=exit_ray_x,
            ray_y=exit_ray_y,
            ray_intensity=exit_ray_intensity,
            ray_opd_waves=corrected_opd_waves,
            reference_phase=reference_phase,
            valid_mask=valid_mask,
        )
        
        if self.debug:
            output_energy = np.sum(np.abs(output_amplitude)**2)
            print(f"      出射能量: {output_energy:.6f}")
            print("[DEBUG] 直接传播完成")
        
        self._intermediate_results['output_amplitude'] = output_amplitude
        
        # =====================================================================
        # 构建返回结果
        # =====================================================================
        
        # 出射面位置（对于单个元件，出射面在元件顶点）
        exit_position = self.entrance_position
        
        return DirectPropagationResult(
            output_amplitude=output_amplitude,
            exit_direction=exit_direction,
            exit_position=exit_position,
            validation_result=validation_result,
            intermediate=self._intermediate_results if self.debug else None,
        )
    
    def _apply_pilot_beam_correction(
        self,
        ray_x: NDArray,
        ray_y: NDArray,
        total_opd_waves: NDArray,
        valid_mask: NDArray,
    ) -> Tuple[NDArray, PilotBeamValidationResult, NDArray]:
        """应用 Pilot Beam 参考相位修正
        
        参数:
            ray_x: 光线 x 坐标
            ray_y: 光线 y 坐标
            total_opd_waves: 总 OPD（波长数）
            valid_mask: 有效光线掩模
        
        返回:
            (reference_phase, validation_result, corrected_opd_waves)
        """
        # 估计光束束腰（假设为物理尺寸的 1/4）
        beam_waist = self.physical_size / 4
        
        # 创建 Pilot Beam 计算器
        pilot_calculator = PilotBeamCalculator(
            wavelength=self.wavelength,
            beam_waist=beam_waist,
            beam_waist_position=0.0,
            element_focal_length=self._get_element_focal_length(),
            method=self.pilot_beam_method,
            grid_size=self.grid_size,
            physical_size=self.physical_size,
        )
        
        # 计算参考相位网格
        half_size = self.physical_size / 2
        x_grid = np.linspace(-half_size, half_size, self.grid_size)
        y_grid = np.linspace(-half_size, half_size, self.grid_size)
        
        reference_phase = pilot_calculator.compute_reference_phase(x_grid, y_grid)
        
        # 验证
        validation_result = pilot_calculator.validate(
            actual_beam_size=self.physical_size / 2
        )
        
        # 创建相位修正器
        corrector = PhaseCorrector(reference_phase, x_grid, y_grid)
        
        # 执行修正
        corrected_opd_waves, residual_phase, warnings_list = corrector.correct_rays(
            ray_x, ray_y, total_opd_waves, self.wavelength
        )
        
        # 发出警告
        for warning_msg in warnings_list:
            warnings.warn(warning_msg, UserWarning)
        
        self._intermediate_results['residual_phase'] = residual_phase
        
        return reference_phase, validation_result, corrected_opd_waves
    
    def _get_element_focal_length(self) -> float:
        """获取元件焦距"""
        if self.element.surface_type == 'mirror':
            if np.isinf(self.element.radius):
                return float('inf')  # 平面镜
            else:
                return self.element.radius / 2  # 球面镜焦距 = R/2
        else:
            return float('inf')  # 其他类型暂时返回无穷大
    
    def get_intermediate_results(self) -> Dict[str, Any]:
        """获取中间结果
        
        返回:
            包含中间计算结果的字典
        """
        return self._intermediate_results.copy()


# =============================================================================
# 便捷函数
# =============================================================================

def propagate_through_element(
    complex_amplitude: NDArray[np.complexfloating],
    element: SurfaceDefinition,
    wavelength: float,
    physical_size: float,
    num_rays: int = 100,
    use_pilot_beam: bool = True,
    debug: bool = False,
) -> DirectPropagationResult:
    """便捷函数：通过单个元件传播复振幅
    
    参数:
        complex_amplitude: 入射面复振幅数组
        element: 光学元件定义
        wavelength: 波长，单位 μm
        physical_size: 物理尺寸（直径），单位 mm
        num_rays: 采样光线数量
        use_pilot_beam: 是否使用 Pilot Beam
        debug: 是否启用调试模式
    
    返回:
        DirectPropagationResult 对象
    
    示例:
        >>> result = propagate_through_element(
        ...     complex_amplitude=amplitude,
        ...     element=mirror,
        ...     wavelength=0.633,
        ...     physical_size=50.0,
        ... )
        >>> output = result.output_amplitude
    """
    propagator = DirectElementPropagator(
        complex_amplitude=complex_amplitude,
        element=element,
        wavelength=wavelength,
        physical_size=physical_size,
        num_rays=num_rays,
        use_pilot_beam=use_pilot_beam,
        debug=debug,
    )
    return propagator.propagate()
