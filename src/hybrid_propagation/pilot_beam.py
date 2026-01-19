"""
Pilot Beam 模块

本模块实现 Pilot Beam 参考相位计算和验证功能。

Pilot Beam 方法用于计算参考相位，以便从光线追迹的 OPD 中提取残差相位。

简化设计（无需 BeamEstimator）：
- 由于初始高斯光束参数已知，使用 ABCD 矩阵法追踪光束在整个系统中的演变
- 在每个元件的入射面和出射面都可以精确计算光束参数
- 参考相位直接从 ABCD 计算的光束参数得到，无需从波前拟合估计

主要类：
- PilotBeamValidator: 验证 Pilot Beam 方法的适用性
- PilotBeamCalculator: 计算 Pilot Beam 参考相位（使用 ABCD 方法）

作者：混合光学仿真项目
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Tuple, TYPE_CHECKING
import warnings

from . import ValidationResult, PilotBeamValidationResult

if TYPE_CHECKING:
    from gaussian_beam_simulation import GaussianBeam, ABCDCalculator
    from gaussian_beam_simulation.optical_elements import OpticalElement


class PilotBeamValidator:
    """Pilot Beam 适用性验证器
    
    检测 Pilot Beam 方法的适用性，包括：
    - 相位采样是否充足（相邻像素相位差不超过 π/2）
    - 光束发散角是否过大
    - Pilot Beam 与实际光束尺寸是否匹配
    
    参数:
        phase_grid: 参考相位网格，形状 (N, N)，单位弧度
        dx: x 方向采样间隔，单位 mm
        dy: y 方向采样间隔，单位 mm
        pilot_beam_size: Pilot Beam 光束尺寸（直径），单位 mm
        actual_beam_size: 实际光束尺寸（直径），单位 mm
        beam_divergence: 光束发散角，单位弧度
    
    示例:
        >>> import numpy as np
        >>> # 创建测试相位网格
        >>> n = 64
        >>> x = np.linspace(-10, 10, n)
        >>> X, Y = np.meshgrid(x, x)
        >>> phase = 0.1 * (X**2 + Y**2)  # 简单的二次相位
        >>> 
        >>> # 创建验证器
        >>> validator = PilotBeamValidator(
        ...     phase_grid=phase,
        ...     dx=20.0/n,
        ...     dy=20.0/n,
        ...     pilot_beam_size=15.0,
        ...     actual_beam_size=12.0,
        ...     beam_divergence=0.01,
        ... )
        >>> 
        >>> # 执行验证
        >>> result = validator.validate_all()
        >>> print(f"验证通过: {result.is_valid}")
    """
    
    # 默认阈值
    DEFAULT_PHASE_THRESHOLD = np.pi / 2  # 相邻像素最大相位差阈值
    DEFAULT_DIVERGENCE_THRESHOLD = 0.1   # 最大发散角阈值（弧度）
    DEFAULT_SIZE_RATIO_THRESHOLD = 1.5   # 最大尺寸比例阈值
    
    def __init__(
        self,
        phase_grid: Optional[NDArray[np.floating]] = None,
        dx: float = 1.0,
        dy: float = 1.0,
        pilot_beam_size: Optional[float] = None,
        actual_beam_size: Optional[float] = None,
        beam_divergence: Optional[float] = None,
    ):
        """初始化验证器
        
        参数:
            phase_grid: 参考相位网格，形状 (N, N)，单位弧度
            dx: x 方向采样间隔，单位 mm
            dy: y 方向采样间隔，单位 mm
            pilot_beam_size: Pilot Beam 光束尺寸（直径），单位 mm
            actual_beam_size: 实际光束尺寸（直径），单位 mm
            beam_divergence: 光束发散角，单位弧度
        """
        self.phase_grid = phase_grid
        self.dx = dx
        self.dy = dy
        self.pilot_beam_size = pilot_beam_size
        self.actual_beam_size = actual_beam_size
        self.beam_divergence = beam_divergence
        
        # 缓存计算结果
        self._phase_gradient_x: Optional[NDArray] = None
        self._phase_gradient_y: Optional[NDArray] = None
        self._max_phase_gradient: Optional[float] = None
        self._mean_phase_gradient: Optional[float] = None
    
    def _compute_phase_gradients(self) -> None:
        """计算相位梯度
        
        使用中心差分计算相位梯度，边界使用前向/后向差分。
        """
        if self.phase_grid is None:
            return
        
        # 计算 x 方向梯度（相邻像素相位差）
        self._phase_gradient_x = np.diff(self.phase_grid, axis=1)
        
        # 计算 y 方向梯度（相邻像素相位差）
        self._phase_gradient_y = np.diff(self.phase_grid, axis=0)
        
        # 将相位差包裹到 [-π, π] 范围
        self._phase_gradient_x = np.angle(np.exp(1j * self._phase_gradient_x))
        self._phase_gradient_y = np.angle(np.exp(1j * self._phase_gradient_y))
        
        # 计算梯度幅值
        # 注意：x 和 y 方向梯度数组大小不同，需要分别处理
        max_grad_x = np.max(np.abs(self._phase_gradient_x))
        max_grad_y = np.max(np.abs(self._phase_gradient_y))
        self._max_phase_gradient = max(max_grad_x, max_grad_y)
        
        mean_grad_x = np.mean(np.abs(self._phase_gradient_x))
        mean_grad_y = np.mean(np.abs(self._phase_gradient_y))
        self._mean_phase_gradient = (mean_grad_x + mean_grad_y) / 2
    
    def check_phase_sampling(
        self,
        threshold: float = DEFAULT_PHASE_THRESHOLD,
    ) -> ValidationResult:
        """检查相位采样是否充足
        
        检查相邻像素间的相位差是否超过阈值（默认 π/2）。
        如果超过阈值，说明采样不足，可能导致相位混叠。
        
        参数:
            threshold: 相位差阈值，单位弧度，默认 π/2
        
        返回:
            ValidationResult: 验证结果
        
        **Validates: Requirements 5.1, 5.2, 5.7**
        """
        if self.phase_grid is None:
            return ValidationResult(
                passed=False,
                message="未提供相位网格数据",
                value=None,
                threshold=threshold,
            )
        
        # 确保已计算梯度
        if self._max_phase_gradient is None:
            self._compute_phase_gradients()
        
        max_gradient = self._max_phase_gradient
        passed = max_gradient <= threshold
        
        if passed:
            message = f"相位采样充足：最大相位梯度 {max_gradient:.4f} rad ≤ 阈值 {threshold:.4f} rad"
        else:
            message = (
                f"相位采样不足警告：最大相位梯度 {max_gradient:.4f} rad > 阈值 {threshold:.4f} rad，"
                "可能导致相位混叠"
            )
        
        return ValidationResult(
            passed=passed,
            message=message,
            value=max_gradient,
            threshold=threshold,
        )
    
    def check_beam_divergence(
        self,
        max_divergence: float = DEFAULT_DIVERGENCE_THRESHOLD,
    ) -> ValidationResult:
        """检查光束发散角是否过大
        
        如果光束发散角过大，Pilot Beam 方法可能不适用。
        
        参数:
            max_divergence: 最大允许发散角，单位弧度，默认 0.1 rad
        
        返回:
            ValidationResult: 验证结果
        
        **Validates: Requirements 5.3, 5.4**
        """
        if self.beam_divergence is None:
            return ValidationResult(
                passed=True,
                message="未提供光束发散角数据，跳过检查",
                value=None,
                threshold=max_divergence,
            )
        
        passed = self.beam_divergence <= max_divergence
        
        if passed:
            message = (
                f"光束发散角正常：{self.beam_divergence:.4f} rad ≤ "
                f"阈值 {max_divergence:.4f} rad"
            )
        else:
            message = (
                f"光束发散角过大警告：{self.beam_divergence:.4f} rad > "
                f"阈值 {max_divergence:.4f} rad，Pilot Beam 方法可能不适用"
            )
        
        return ValidationResult(
            passed=passed,
            message=message,
            value=self.beam_divergence,
            threshold=max_divergence,
        )
    
    def check_beam_size_match(
        self,
        max_ratio: float = DEFAULT_SIZE_RATIO_THRESHOLD,
    ) -> ValidationResult:
        """检查 Pilot Beam 与实际光束尺寸匹配度
        
        如果 Pilot Beam 尺寸与实际光束尺寸差异超过 50%（比例超过 1.5），
        发出警告。
        
        参数:
            max_ratio: 最大允许尺寸比例，默认 1.5
        
        返回:
            ValidationResult: 验证结果
        
        **Validates: Requirements 5.5, 5.6**
        """
        if self.pilot_beam_size is None or self.actual_beam_size is None:
            return ValidationResult(
                passed=True,
                message="未提供光束尺寸数据，跳过检查",
                value=None,
                threshold=max_ratio,
            )
        
        # 计算尺寸比例（取较大值/较小值）
        if self.actual_beam_size > 0 and self.pilot_beam_size > 0:
            ratio = max(
                self.pilot_beam_size / self.actual_beam_size,
                self.actual_beam_size / self.pilot_beam_size,
            )
        else:
            ratio = float('inf')
        
        passed = ratio <= max_ratio
        
        if passed:
            message = (
                f"光束尺寸匹配良好：Pilot Beam {self.pilot_beam_size:.2f} mm，"
                f"实际光束 {self.actual_beam_size:.2f} mm，比例 {ratio:.2f} ≤ {max_ratio:.2f}"
            )
        else:
            message = (
                f"光束尺寸不匹配警告：Pilot Beam {self.pilot_beam_size:.2f} mm，"
                f"实际光束 {self.actual_beam_size:.2f} mm，比例 {ratio:.2f} > {max_ratio:.2f}，"
                "差异超过 50%"
            )
        
        return ValidationResult(
            passed=passed,
            message=message,
            value=ratio,
            threshold=max_ratio,
        )
    
    def validate_all(self) -> PilotBeamValidationResult:
        """执行所有验证检查
        
        返回:
            PilotBeamValidationResult: 完整的验证结果
        
        **Validates: Requirements 5.1-5.8**
        """
        # 执行各项检查
        phase_sampling = self.check_phase_sampling()
        beam_divergence = self.check_beam_divergence()
        beam_size_match = self.check_beam_size_match()
        
        # 确保已计算梯度
        if self._max_phase_gradient is None:
            self._compute_phase_gradients()
        
        # 收集警告信息
        warnings_list: List[str] = []
        if not phase_sampling.passed:
            warnings_list.append(phase_sampling.message)
        if not beam_divergence.passed:
            warnings_list.append(beam_divergence.message)
        if not beam_size_match.passed:
            warnings_list.append(beam_size_match.message)
        
        # 整体验证结果（所有检查都通过才算通过）
        is_valid = phase_sampling.passed and beam_divergence.passed and beam_size_match.passed
        
        return PilotBeamValidationResult(
            is_valid=is_valid,
            phase_sampling=phase_sampling,
            beam_divergence=beam_divergence,
            beam_size_match=beam_size_match,
            max_phase_gradient=self._max_phase_gradient if self._max_phase_gradient is not None else 0.0,
            mean_phase_gradient=self._mean_phase_gradient if self._mean_phase_gradient is not None else 0.0,
            warnings=warnings_list,
        )


class PilotBeamCalculator:
    """Pilot Beam 参考相位计算器（基于 ABCD 矩阵法）
    
    使用 ABCD 矩阵法计算高斯光束在光学系统中的传播，
    并在指定位置计算参考相位。
    
    简化设计：
    - 由于初始高斯光束参数已知，无需从波前拟合估计
    - 使用 ABCDCalculator 追踪光束在整个系统中的演变
    - 在每个元件的入射面和出射面都可以精确计算光束参数
    
    有两种创建方式：
    
    1. 新接口（推荐）：使用 GaussianBeam 和元件列表
        >>> from gaussian_beam_simulation import GaussianBeam, ParabolicMirror
        >>> beam = GaussianBeam(wavelength=0.633, w0=5.0, z0=0.0, z_init=0.0)
        >>> mirror = ParabolicMirror(thickness=100.0, semi_aperture=15.0,
        ...                          parent_focal_length=100.0)
        >>> calculator = PilotBeamCalculator(
        ...     beam=beam,
        ...     elements=[mirror],
        ...     initial_distance=200.0,
        ... )
    
    2. 旧接口（向后兼容）：使用波长、束腰等参数
        >>> calculator = PilotBeamCalculator(
        ...     wavelength=0.633,
        ...     beam_waist=5.0,
        ...     beam_waist_position=0.0,
        ...     element_focal_length=100.0,
        ... )
    """
    
    def __init__(
        self,
        beam: Optional['GaussianBeam'] = None,
        elements: Optional[List['OpticalElement']] = None,
        initial_distance: float = 0.0,
        grid_size: int = 64,
        physical_size: float = 20.0,
        # 旧接口参数（向后兼容）
        wavelength: Optional[float] = None,
        beam_waist: Optional[float] = None,
        beam_waist_position: float = 0.0,
        element_focal_length: float = float('inf'),
        method: str = 'analytical',
    ):
        """初始化计算器
        
        新接口参数:
            beam: 高斯光束对象
            elements: 光学元件列表
            initial_distance: 从光束初始面到第一个元件的距离（mm）
            grid_size: 网格大小
            physical_size: 物理尺寸（直径，mm）
        
        旧接口参数（向后兼容）:
            wavelength: 波长（μm）
            beam_waist: 束腰半径（mm）
            beam_waist_position: 束腰位置（mm）
            element_focal_length: 元件焦距（mm）
            method: 计算方法（'analytical' 或 'proper'）
        """
        self.grid_size = grid_size
        self.physical_size = physical_size
        self._validator: Optional[PilotBeamValidator] = None
        
        # 检测使用哪种接口
        if beam is not None:
            # 新接口
            self._use_new_interface = True
            self.beam = beam
            self.elements = elements or []
            self.initial_distance = initial_distance
            
            # 创建 ABCD 计算器
            from gaussian_beam_simulation import ABCDCalculator
            self._abcd_calculator = ABCDCalculator(
                beam=beam,
                elements=self.elements,
                initial_distance=initial_distance,
            )
        elif wavelength is not None and beam_waist is not None:
            # 旧接口（向后兼容）
            self._use_new_interface = False
            self._wavelength = wavelength
            self._beam_waist = beam_waist
            self._beam_waist_position = beam_waist_position
            self._element_focal_length = element_focal_length
            self._method = method
            
            # 计算波长（转换为 mm）
            self._wavelength_mm = wavelength * 1e-3
            
            # 计算瑞利长度
            self._rayleigh_length = np.pi * beam_waist**2 / self._wavelength_mm
            
            # 不创建 ABCD 计算器
            self._abcd_calculator = None
            self.beam = None
            self.elements = []
        else:
            raise ValueError(
                "必须提供 beam 参数（新接口）或 wavelength 和 beam_waist 参数（旧接口）"
            )
    
    # ========== 属性（向后兼容） ==========
    
    @property
    def wavelength(self) -> float:
        """波长（μm）"""
        if self._use_new_interface:
            return self.beam.wavelength
        return self._wavelength
    
    @property
    def wavelength_mm(self) -> float:
        """波长（mm）"""
        if self._use_new_interface:
            return self.beam.wavelength * 1e-3
        return self._wavelength_mm
    
    @property
    def beam_waist(self) -> float:
        """束腰半径（mm）"""
        if self._use_new_interface:
            return self.beam.w0
        return self._beam_waist
    
    @property
    def beam_waist_position(self) -> float:
        """束腰位置（mm）"""
        if self._use_new_interface:
            return self.beam.z0
        return self._beam_waist_position
    
    @property
    def element_focal_length(self) -> float:
        """元件焦距（mm）"""
        if self._use_new_interface:
            if self.elements:
                return self.elements[0].focal_length
            return float('inf')
        return self._element_focal_length
    
    @property
    def method(self) -> str:
        """计算方法"""
        if self._use_new_interface:
            return 'abcd'
        return self._method
    
    @property
    def rayleigh_length(self) -> float:
        """瑞利长度（mm）"""
        if self._use_new_interface:
            return self.beam.zR
        return self._rayleigh_length
    
    # ========== 新接口方法 ==========
    
    def get_beam_params_at_element(
        self,
        element_index: int,
        position: str = 'exit',
    ):
        """获取指定元件处的光束参数
        
        参数:
            element_index: 元件索引
            position: 'entrance' 或 'exit'
        
        返回:
            ABCDResult 对象
        """
        if not self._use_new_interface:
            raise RuntimeError("此方法仅在新接口模式下可用")
        return self._abcd_calculator.get_beam_at_element(element_index, position)
    
    def compute_reference_phase(
        self,
        x_coords: NDArray[np.floating],
        y_coords: NDArray[np.floating],
        element_index: int = 0,
        position: str = 'exit',
    ) -> NDArray[np.floating]:
        """计算参考相位
        
        参数:
            x_coords: x 坐标数组（mm）
            y_coords: y 坐标数组（mm）
            element_index: 元件索引（新接口）
            position: 'entrance' 或 'exit'（新接口）
        
        返回:
            参考相位数组（弧度）
        """
        if self._use_new_interface:
            return self._abcd_calculator.compute_reference_phase_at_position(
                x_coords, y_coords, element_index, position
            )
        else:
            # 旧接口
            if self._method == 'proper':
                return self.compute_reference_phase_proper(x_coords, y_coords)
            elif self._method == 'analytical':
                return self.compute_reference_phase_analytical(x_coords, y_coords)
            else:
                raise ValueError(f"未知的计算方法: {self._method}，支持 'proper' 或 'analytical'")
    
    def compute_reference_phase_grid(
        self,
        element_index: int = 0,
        position: str = 'exit',
    ) -> NDArray[np.floating]:
        """在网格上计算参考相位
        
        参数:
            element_index: 元件索引
            position: 'entrance' 或 'exit'
        
        返回:
            参考相位网格（弧度），形状 (grid_size, grid_size)
        """
        if self._use_new_interface:
            return self._abcd_calculator.compute_reference_phase_grid(
                self.grid_size, self.physical_size, element_index, position
            )
        else:
            # 旧接口
            half_size = self.physical_size / 2
            x = np.linspace(-half_size, half_size, self.grid_size)
            y = np.linspace(-half_size, half_size, self.grid_size)
            return self.compute_reference_phase(x, y)
    
    def compute_reference_phase_at_rays(
        self,
        ray_x: NDArray[np.floating],
        ray_y: NDArray[np.floating],
        element_index: int = 0,
        position: str = 'exit',
    ) -> NDArray[np.floating]:
        """在光线位置计算参考相位"""
        return self.compute_reference_phase(ray_x, ray_y, element_index, position)
    
    # ========== 旧接口方法（向后兼容） ==========
    
    def compute_reference_phase_proper(
        self,
        x_coords: NDArray[np.floating],
        y_coords: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """使用 PROPER 计算参考相位（旧接口）"""
        try:
            import proper
        except ImportError:
            warnings.warn(
                "PROPER 库未安装，回退到解析方法",
                UserWarning,
            )
            return self.compute_reference_phase_analytical(x_coords, y_coords)
        
        # 创建网格
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # 计算到束腰的距离
        z = -self._beam_waist_position
        
        # 使用 PROPER 创建高斯光束
        beam_diameter_m = self._beam_waist * 2 * 1e-3
        wavelength_m = self._wavelength * 1e-6
        
        wfo = proper.prop_begin(
            beam_diameter_m,
            wavelength_m,
            len(x_coords),
            beam_diam_fraction=0.5,
        )
        
        if np.isfinite(self._element_focal_length):
            focal_length_m = self._element_focal_length * 1e-3
            proper.prop_lens(wfo, focal_length_m)
        
        phase = proper.prop_get_phase(wfo)
        
        return phase.astype(np.float64)
    
    def compute_reference_phase_analytical(
        self,
        x_coords: NDArray[np.floating],
        y_coords: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """解析计算参考相位（旧接口）"""
        # 创建网格
        X, Y = np.meshgrid(x_coords, y_coords)
        r_squared = X**2 + Y**2
        
        # 计算到束腰的距离
        z = -self._beam_waist_position
        
        # 波数
        k = 2 * np.pi / self._wavelength_mm
        
        # 瑞利长度
        z_R = self._rayleigh_length
        
        # 初始化相位
        phase = np.zeros_like(r_squared)
        
        if abs(z) < 1e-10:
            pass
        else:
            R_z = z * (1 + (z_R / z)**2)
            phase = k * r_squared / (2 * R_z)
        
        if np.isfinite(self._element_focal_length):
            lens_phase = -k * r_squared / (2 * self._element_focal_length)
            phase = phase + lens_phase
        
        return phase.astype(np.float64)
    
    def validate(
        self,
        element_index: int = 0,
        position: str = 'exit',
        actual_beam_size: Optional[float] = None,
    ) -> PilotBeamValidationResult:
        """验证 Pilot Beam 适用性"""
        if self._use_new_interface:
            # 新接口
            phase_grid = self.compute_reference_phase_grid(element_index, position)
            beam_params = self.get_beam_params_at_element(element_index, position)
            pilot_beam_size = 2 * beam_params.w
            beam_divergence = self.beam.divergence
        else:
            # 旧接口
            half_size = self.physical_size / 2
            x = np.linspace(-half_size, half_size, self.grid_size)
            y = np.linspace(-half_size, half_size, self.grid_size)
            phase_grid = self.compute_reference_phase(x, y)
            pilot_beam_size = self._beam_waist * 2
            beam_divergence = self._wavelength_mm / (np.pi * self._beam_waist)
        
        dx = self.physical_size / self.grid_size
        dy = self.physical_size / self.grid_size
        
        self._validator = PilotBeamValidator(
            phase_grid=phase_grid,
            dx=dx,
            dy=dy,
            pilot_beam_size=pilot_beam_size,
            actual_beam_size=actual_beam_size,
            beam_divergence=beam_divergence,
        )
        
        return self._validator.validate_all()
