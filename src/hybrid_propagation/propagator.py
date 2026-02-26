"""
混合元件传播器主模块

本模块实现 HybridElementPropagator 主类，协调整个混合传播流程。

核心数据流：
    入射面复振幅 → [Tilted ASM] → 切平面复振幅 → [波前采样] → 光线
    → [元件追迹] → 出射光线 → [Pilot Beam 修正] → [复振幅重建]
    → 切平面复振幅 → [Tilted ASM] → 出射面复振幅

作者：混合光学仿真项目
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Dict, Any
import warnings
import sys

# 添加项目路径
sys.path.insert(0, 'src')

from .tilted_propagation import TiltedPropagation
from .pilot_beam import PilotBeamCalculator, PilotBeamValidator
from .phase_correction import PhaseCorrector
from .amplitude_reconstruction import AmplitudeReconstructor
from . import (
    ValidationResult,
    PilotBeamValidationResult,
    PropagationInput,
    PropagationOutput,
    IntermediateResults,
)

# 导入现有模块
from wavefront_to_rays.wavefront_sampler import WavefrontToRaysSampler
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition

# 导入异常类
from sequential_system.exceptions import SimulationError, PilotBeamWarning


class HybridElementPropagator:
    """混合元件传播器
    
    在光学元件处执行完整的波前-光线-波前重建流程。
    
    参数:
        complex_amplitude: 入射面复振幅数组，形状 (N, N)
        element: 光学元件定义 (SurfaceDefinition)
        wavelength: 波长，单位 μm
        physical_size: 波前物理尺寸（直径），单位 mm
        grid_size: 输出网格大小，默认与输入相同
        num_rays: 采样光线数量，默认 100
        pilot_beam_method: Pilot Beam 方法，'proper' 或 'analytical'
        entrance_direction: 入射主光线方向 (L, M, N)，默认 (0, 0, 1)
        entrance_position: 入射面中心位置 (x, y, z)，单位 mm，默认 (0, 0, 0)
        debug: 是否启用调试模式
    
    示例:
        >>> import numpy as np
        >>> from wavefront_to_rays import SurfaceDefinition
        >>> 
        >>> # 创建输入复振幅
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
        >>> propagator = HybridElementPropagator(
        ...     complex_amplitude=amplitude,
        ...     element=element,
        ...     wavelength=0.633,
        ...     physical_size=50.0,
        ... )
        >>> output_amplitude = propagator.propagate()
    
    **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 11.1, 11.2, 11.3, 11.4**
    """
    
    def __init__(
        self,
        complex_amplitude: NDArray[np.complexfloating],
        element: SurfaceDefinition,
        wavelength: float,
        physical_size: float,
        grid_size: Optional[int] = None,
        num_rays: int = 100,
        pilot_beam_method: str = 'analytical',
        entrance_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        entrance_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        debug: bool = False,
    ):
        """初始化混合元件传播器
        
        **Validates: Requirements 9.2, 11.1, 11.2, 11.3**
        """
        # 输入验证
        self._validate_inputs(
            complex_amplitude, element, wavelength, physical_size,
            grid_size, num_rays, pilot_beam_method
        )
        
        # 存储参数
        self.complex_amplitude = complex_amplitude
        self.element = element
        self.wavelength = wavelength
        self.physical_size = physical_size
        self.grid_size = grid_size if grid_size is not None else complex_amplitude.shape[0]
        self.num_rays = num_rays
        self.pilot_beam_method = pilot_beam_method
        self.entrance_direction = entrance_direction
        self.entrance_position = entrance_position
        self.debug = debug
        
        # 计算采样间隔
        self.dx = physical_size / complex_amplitude.shape[0]
        self.dy = physical_size / complex_amplitude.shape[1]
        
        # 波长转换为 mm
        self.wavelength_mm = wavelength * 1e-3
        
        # 初始化子模块
        self._init_submodules()
        
        # 中间结果存储
        self._intermediate_results: Optional[IntermediateResults] = None
        self._validation_result: Optional[PilotBeamValidationResult] = None
        
        if self.debug:
            print(f"[DEBUG] HybridElementPropagator 初始化完成")
            print(f"  - 输入网格大小: {complex_amplitude.shape}")
            print(f"  - 输出网格大小: {self.grid_size}")
            print(f"  - 波长: {wavelength} μm")
            print(f"  - 物理尺寸: {physical_size} mm")
            print(f"  - 光线数量: {num_rays}")
    
    def _validate_inputs(
        self,
        complex_amplitude: NDArray,
        element: SurfaceDefinition,
        wavelength: float,
        physical_size: float,
        grid_size: Optional[int],
        num_rays: int,
        pilot_beam_method: str,
    ) -> None:
        """验证输入参数
        
        **Validates: Requirements 11.1, 11.2, 11.3**
        """
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
        
        # 检查 Pilot Beam 方法
        if pilot_beam_method not in ['proper', 'analytical']:
            raise ValueError(
                f"Pilot Beam 方法必须是 'proper' 或 'analytical'，"
                f"当前值: {pilot_beam_method}"
            )
        
        # 检查元件定义
        if element is None:
            raise ValueError("元件定义不能为 None")
    
    def _init_submodules(self) -> None:
        """初始化子模块"""
        # 倾斜传播模块
        self._tilted_propagation = TiltedPropagation(
            wavelength=self.wavelength_mm,
            dx=self.dx,
            dy=self.dy,
        )
        
        # Pilot Beam 计算器
        # 估计光束束腰（假设为物理尺寸的一半）
        beam_waist = self.physical_size / 4
        self._pilot_beam_calculator = PilotBeamCalculator(
            wavelength=self.wavelength,
            beam_waist=beam_waist,
            beam_waist_position=0.0,
            element_focal_length=self._get_element_focal_length(),
            method=self.pilot_beam_method,
            grid_size=self.grid_size,
            physical_size=self.physical_size,
        )
        
        # 复振幅重建器
        self._amplitude_reconstructor = AmplitudeReconstructor(
            grid_size=self.grid_size,
            physical_size=self.physical_size,
            wavelength=self.wavelength,
        )
    
    def _get_element_focal_length(self) -> float:
        """获取元件焦距"""
        if self.element.surface_type == 'mirror':
            if np.isinf(self.element.radius):
                return float('inf')  # 平面镜
            else:
                return self.element.radius / 2  # 球面镜焦距 = R/2
        else:
            return float('inf')  # 其他类型暂时返回无穷大
    
    def propagate(self) -> NDArray[np.complexfloating]:
        """执行完整的混合传播
        
        内部流程：
        1. 使用 TiltedPropagation 从入射面传播到切平面
        2. 使用 WavefrontToRaysSampler 采样为光线
        3. 使用 ElementRaytracer 追迹光线
        4. 使用 PilotBeamCalculator 计算参考相位
        5. 使用 PhaseCorrector 修正相位
        6. 使用 AmplitudeReconstructor 重建复振幅
        7. 使用 TiltedPropagation 从切平面传播到出射面
        
        返回:
            出射面复振幅数组，形状 (grid_size, grid_size)
        
        **Validates: Requirements 9.1, 9.3, 11.4**
        """
        if self.debug:
            print("[DEBUG] 开始混合传播...")
        
        # 1. 从入射面传播到切平面
        tangent_amplitude_in = self._propagate_to_tangent_plane()
        
        if self.debug:
            print(f"  [1] 切平面输入复振幅: 形状={tangent_amplitude_in.shape}, "
                  f"能量={np.sum(np.abs(tangent_amplitude_in)**2):.6f}")
        
        # 2. 波前采样为光线
        rays_in, ray_positions, ray_amplitudes = self._sample_wavefront(tangent_amplitude_in)
        
        if self.debug:
            print(f"  [2] 采样光线数量: {len(ray_positions[0])}")
        
        # 3. 元件光线追迹
        rays_out, ray_opd_waves, valid_mask = self._trace_rays(rays_in)
        
        if self.debug:
            valid_count = np.sum(valid_mask)
            print(f"  [3] 有效光线数量: {valid_count}/{len(valid_mask)}")
        
        # 检查是否所有光线都无效
        if not np.any(valid_mask):
            raise SimulationError(
                "所有光线都无效，无法完成传播。"
                "可能的原因：元件半口径过小、光束尺寸过大、或元件位置不正确。"
            )
        
        # 4. 计算 Pilot Beam 参考相位
        reference_phase, validation_result = self._compute_reference_phase()
        self._validation_result = validation_result
        
        if self.debug:
            print(f"  [4] Pilot Beam 验证: is_valid={validation_result.is_valid}")
        
        # 发出警告（如果有）
        if not validation_result.is_valid:
            for warning_msg in validation_result.warnings:
                warnings.warn(warning_msg, PilotBeamWarning)
        
        # 5. 相位修正
        corrected_opd, residual_phase = self._correct_phase(
            ray_positions, ray_opd_waves, reference_phase, valid_mask
        )
        
        if self.debug:
            print(f"  [5] 相位修正完成: 残差相位范围=[{np.nanmin(residual_phase):.4f}, "
                  f"{np.nanmax(residual_phase):.4f}] rad")
        
        # 6. 复振幅重建
        tangent_amplitude_out = self._reconstruct_amplitude(
            ray_positions, ray_amplitudes, corrected_opd, reference_phase, valid_mask
        )
        
        if self.debug:
            print(f"  [6] 切平面输出复振幅: 能量={np.sum(np.abs(tangent_amplitude_out)**2):.6f}")
        
        # 7. 从切平面传播到出射面
        output_amplitude = self._propagate_from_tangent_plane(tangent_amplitude_out)
        
        if self.debug:
            print(f"  [7] 出射面复振幅: 能量={np.sum(np.abs(output_amplitude)**2):.6f}")
            print("[DEBUG] 混合传播完成")
        
        # 存储中间结果
        self._intermediate_results = IntermediateResults(
            tangent_amplitude_in=tangent_amplitude_in,
            rays_in=rays_in,
            rays_out=rays_out,
            pilot_phase=reference_phase,
            residual_phase=residual_phase,
            tangent_amplitude_out=tangent_amplitude_out,
        )
        
        return output_amplitude
    
    def _propagate_to_tangent_plane(self) -> NDArray[np.complexfloating]:
        """从入射面传播到切平面"""
        return self._tilted_propagation.propagate_to_tangent_plane(
            self.complex_amplitude,
            self.element.tilt_x,
            self.element.tilt_y,
        )
    
    def _propagate_from_tangent_plane(
        self,
        tangent_amplitude: NDArray[np.complexfloating],
    ) -> NDArray[np.complexfloating]:
        """从切平面传播到出射面"""
        is_reflective = self.element.surface_type == 'mirror'
        return self._tilted_propagation.propagate_from_tangent_plane(
            tangent_amplitude,
            self.element.tilt_x,
            self.element.tilt_y,
            is_reflective,
        )
    
    def _sample_wavefront(
        self,
        amplitude: NDArray[np.complexfloating],
    ) -> Tuple[Any, Tuple[NDArray, NDArray], NDArray]:
        """波前采样为光线"""
        # 创建波前采样器
        sampler = WavefrontToRaysSampler(
            wavefront_amplitude=amplitude,
            wavelength=self.wavelength,
            physical_size=self.physical_size,
            num_rays=self.num_rays,
        )
        
        # 获取光线
        rays = sampler.get_output_rays()
        
        # 获取光线位置
        ray_x, ray_y = sampler.get_ray_positions()
        
        # 获取光线振幅
        ray_amplitudes = np.sqrt(np.asarray(rays.i))  # 强度转振幅
        
        return rays, (ray_x, ray_y), ray_amplitudes
    
    def _trace_rays(
        self,
        rays_in: Any,
    ) -> Tuple[Any, NDArray, NDArray]:
        """元件光线追迹"""
        # 创建光线追迹器
        raytracer = ElementRaytracer(
            surfaces=[self.element],  # 将单个元件包装为列表
            wavelength=self.wavelength,
        )
        
        # 追迹光线
        rays_out = raytracer.trace(rays_in)
        
        # 获取 OPD
        ray_opd_waves = raytracer.get_relative_opd_waves()
        
        # 获取有效光线掩模
        valid_mask = raytracer.get_valid_ray_mask()
        
        return rays_out, ray_opd_waves, valid_mask
    
    def _compute_reference_phase(
        self,
    ) -> Tuple[NDArray, PilotBeamValidationResult]:
        """计算 Pilot Beam 参考相位"""
        # 计算参考相位网格
        half_size = self.physical_size / 2
        x = np.linspace(-half_size, half_size, self.grid_size)
        y = np.linspace(-half_size, half_size, self.grid_size)
        
        reference_phase = self._pilot_beam_calculator.compute_reference_phase(x, y)
        
        # 验证
        validation_result = self._pilot_beam_calculator.validate(
            actual_beam_size=self.physical_size / 2
        )
        
        return reference_phase, validation_result
    
    def _correct_phase(
        self,
        ray_positions: Tuple[NDArray, NDArray],
        ray_opd_waves: NDArray,
        reference_phase: NDArray,
        valid_mask: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """相位修正"""
        ray_x, ray_y = ray_positions
        
        # 创建相位修正器
        half_size = self.physical_size / 2
        x = np.linspace(-half_size, half_size, reference_phase.shape[1])
        y = np.linspace(-half_size, half_size, reference_phase.shape[0])
        
        corrector = PhaseCorrector(reference_phase, x, y)
        
        # 执行修正
        corrected_opd, residual_phase, warnings_list = corrector.correct_rays(
            ray_x, ray_y, ray_opd_waves, self.wavelength
        )
        
        # 发出警告
        for warning_msg in warnings_list:
            warnings.warn(warning_msg, PilotBeamWarning)
        
        return corrected_opd, residual_phase
    
    def _reconstruct_amplitude(
        self,
        ray_positions: Tuple[NDArray, NDArray],
        ray_amplitudes: NDArray,
        corrected_opd: NDArray,
        reference_phase: NDArray,
        valid_mask: NDArray,
    ) -> NDArray[np.complexfloating]:
        """复振幅重建"""
        ray_x, ray_y = ray_positions
        ray_intensity = ray_amplitudes ** 2
        
        return self._amplitude_reconstructor.reconstruct(
            ray_x, ray_y, ray_intensity, corrected_opd,
            reference_phase, valid_mask
        )
    
    def get_intermediate_results(self) -> Dict[str, Any]:
        """获取中间结果
        
        返回:
            包含以下键的字典:
            - 'tangent_amplitude_in': 切平面输入复振幅
            - 'rays_in': 输入光线
            - 'rays_out': 输出光线
            - 'pilot_phase': Pilot Beam 参考相位
            - 'residual_phase': 残差相位
            - 'tangent_amplitude_out': 切平面输出复振幅
            - 'validation_result': Pilot Beam 验证结果
        
        **Validates: Requirements 9.4**
        """
        if self._intermediate_results is None:
            return {}
        
        return {
            'tangent_amplitude_in': self._intermediate_results.tangent_amplitude_in,
            'rays_in': self._intermediate_results.rays_in,
            'rays_out': self._intermediate_results.rays_out,
            'pilot_phase': self._intermediate_results.pilot_phase,
            'residual_phase': self._intermediate_results.residual_phase,
            'tangent_amplitude_out': self._intermediate_results.tangent_amplitude_out,
            'validation_result': self._validation_result,
        }
