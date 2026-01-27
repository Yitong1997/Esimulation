# -*- coding: utf-8 -*-
"""
圆形光阑实现

本模块实现圆形光阑类，支持四种振幅透过率设置方法：
- HARD_EDGE: 硬边光阑，使用 PROPER 的 prop_circular_aperture 实现
- GAUSSIAN: 高斯光阑，透过率按 T(r) = exp(-0.5 × (r/σ)²) 分布
- SUPER_GAUSSIAN: 超高斯/软边光阑，透过率按 T(r) = exp(-(r/r₀)ⁿ) 分布
- EIGHTH_ORDER: 8 阶软边光阑，使用 PROPER 的 prop_8th_order_mask 实现

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 4.11
"""

from typing import Optional, TYPE_CHECKING
import numpy as np

from .data_models import ApertureType, PowerTransmissionResult
from .exceptions import InvalidInputError

if TYPE_CHECKING:
    import proper


class CircularAperture:
    """圆形光阑
    
    支持四种振幅透过率设置方法：
    - HARD_EDGE: 硬边光阑
    - GAUSSIAN: 高斯光阑
    - SUPER_GAUSSIAN: 超高斯/软边光阑
    - EIGHTH_ORDER: 8 阶软边光阑
    
    Attributes:
        aperture_type: 光阑类型
        radius: 光阑半径 (m)，或归一化半径
        normalized: 是否使用归一化半径
        center_x: 光阑中心 X 坐标 (m)
        center_y: 光阑中心 Y 坐标 (m)
        gaussian_sigma: 高斯光阑的 σ 参数 (m)
        super_gaussian_order: 超高斯光阑的阶数 n
        min_transmission: 8 阶光阑的最小透过率
        max_transmission: 8 阶光阑的最大透过率
    
    Requirements: 4.5, 4.6, 4.7
    """
    
    def __init__(
        self,
        aperture_type: ApertureType,
        radius: float,
        normalized: bool = False,
        center_x: float = 0.0,
        center_y: float = 0.0,
        # 高斯光阑参数
        gaussian_sigma: Optional[float] = None,
        # 超高斯光阑参数
        super_gaussian_order: int = 2,
        # 8 阶光阑参数
        min_transmission: float = 0.0,
        max_transmission: float = 1.0,
    ):
        """初始化圆形光阑
        
        参数:
            aperture_type: 光阑类型
            radius: 光阑半径 (m)，或归一化半径（如果 normalized=True）
            normalized: 是否使用归一化半径（相对于光束半径）
            center_x: 光阑中心 X 坐标 (m)
            center_y: 光阑中心 Y 坐标 (m)
            gaussian_sigma: 高斯光阑的 σ 参数 (m)，如果为 None 则默认等于 radius
            super_gaussian_order: 超高斯光阑的阶数 n，必须 >= 1
            min_transmission: 8 阶光阑的最小透过率，范围 [0, 1]
            max_transmission: 8 阶光阑的最大透过率，范围 [0, 1]
        
        Raises:
            InvalidInputError: 当参数无效时抛出
        
        Requirements: 4.5, 4.6, 4.7, 4.8, 4.9
        """
        # 验证 aperture_type
        if not isinstance(aperture_type, ApertureType):
            raise InvalidInputError(
                f"aperture_type 必须是 ApertureType 枚举类型，"
                f"收到: {type(aperture_type).__name__}"
            )
        
        # 验证 radius
        if radius <= 0:
            raise InvalidInputError(
                f"radius 必须为正数，收到: {radius}"
            )
        
        # 验证 super_gaussian_order
        if super_gaussian_order < 1:
            raise InvalidInputError(
                f"super_gaussian_order 必须 >= 1，收到: {super_gaussian_order}"
            )

        # 验证 min_transmission 和 max_transmission
        if not (0.0 <= min_transmission <= 1.0):
            raise InvalidInputError(
                f"min_transmission 必须在 [0, 1] 范围内，收到: {min_transmission}"
            )
        if not (0.0 <= max_transmission <= 1.0):
            raise InvalidInputError(
                f"max_transmission 必须在 [0, 1] 范围内，收到: {max_transmission}"
            )
        if min_transmission > max_transmission:
            raise InvalidInputError(
                f"min_transmission ({min_transmission}) 不能大于 "
                f"max_transmission ({max_transmission})"
            )
        
        # 验证高斯光阑参数
        if aperture_type == ApertureType.GAUSSIAN:
            if gaussian_sigma is not None and gaussian_sigma <= 0:
                raise InvalidInputError(
                    f"gaussian_sigma 必须为正数，收到: {gaussian_sigma}"
                )
        
        # 存储参数
        self.aperture_type = aperture_type
        self.radius = radius
        self.normalized = normalized
        self.center_x = center_x
        self.center_y = center_y
        
        # 高斯光阑参数：如果未指定 sigma，默认等于 radius
        self.gaussian_sigma = gaussian_sigma if gaussian_sigma is not None else radius
        
        # 超高斯光阑参数
        self.super_gaussian_order = super_gaussian_order
        
        # 8 阶光阑参数
        self.min_transmission = min_transmission
        self.max_transmission = max_transmission
    
    def _get_actual_radius(self, wfo: "proper.WaveFront") -> float:
        """获取实际光阑半径（米）
        
        如果使用归一化半径，则根据光束半径计算实际半径。
        
        参数:
            wfo: PROPER 波前对象
        
        返回:
            实际光阑半径 (m)
        
        Requirements: 4.5
        """
        if self.normalized:
            # 归一化半径：相对于光束半径
            # 从 PROPER 波前对象获取光束半径
            import proper
            beam_radius = proper.prop_get_beamradius(wfo)
            return self.radius * beam_radius
        else:
            return self.radius
    
    def _get_actual_sigma(self, wfo: "proper.WaveFront") -> float:
        """获取实际高斯 sigma 参数（米）
        
        如果使用归一化半径，则根据光束半径计算实际 sigma。
        
        参数:
            wfo: PROPER 波前对象
        
        返回:
            实际 sigma (m)
        """
        if self.normalized:
            import proper
            beam_radius = proper.prop_get_beamradius(wfo)
            return self.gaussian_sigma * beam_radius
        else:
            return self.gaussian_sigma
    
    def apply(self, wfo: "proper.WaveFront") -> np.ndarray:
        """将光阑应用到 PROPER 波前
        
        根据光阑类型调用对应的私有方法应用光阑。
        
        参数:
            wfo: PROPER 波前对象
        
        返回:
            光阑透过率掩模数组
        
        Requirements: 4.1, 4.2, 4.3, 4.4, 4.7
        """
        if self.aperture_type == ApertureType.HARD_EDGE:
            return self._apply_hard_edge(wfo)
        elif self.aperture_type == ApertureType.GAUSSIAN:
            return self._apply_gaussian(wfo)
        elif self.aperture_type == ApertureType.SUPER_GAUSSIAN:
            return self._apply_super_gaussian(wfo)
        elif self.aperture_type == ApertureType.EIGHTH_ORDER:
            return self._apply_eighth_order(wfo)
        else:
            raise InvalidInputError(
                f"未知的光阑类型: {self.aperture_type}"
            )
    
    def _apply_hard_edge(self, wfo: "proper.WaveFront") -> np.ndarray:
        """应用硬边光阑
        
        使用 PROPER 的 prop_circular_aperture 函数应用抗锯齿硬边光阑。
        
        参数:
            wfo: PROPER 波前对象
        
        返回:
            光阑透过率掩模数组
        
        Requirements: 4.1
        """
        import proper
        
        # 获取实际光阑半径
        actual_radius = self._get_actual_radius(wfo)
        
        # 调用 PROPER 的 prop_circular_aperture
        # 注意：prop_circular_aperture 会直接修改 wfo.wfarr
        # 参数：wf, radius, xc=0.0, yc=0.0, NORM=False
        proper.prop_circular_aperture(
            wfo,
            actual_radius,
            self.center_x,
            self.center_y,
        )
        
        # 返回透过率掩模（用于后续分析）
        # 创建一个与波前相同大小的掩模数组
        sampling = proper.prop_get_sampling(wfo)
        grid_size = wfo.wfarr.shape[0]
        
        # 创建坐标网格
        x = (np.arange(grid_size) - grid_size / 2) * sampling
        y = (np.arange(grid_size) - grid_size / 2) * sampling
        X, Y = np.meshgrid(x, y)
        
        # 计算到光阑中心的距离
        R = np.sqrt((X - self.center_x)**2 + (Y - self.center_y)**2)
        
        # 创建硬边掩模（透过率为 0 或 1）
        mask = (R <= actual_radius).astype(float)
        
        return mask
    
    def _apply_gaussian(self, wfo: "proper.WaveFront") -> np.ndarray:
        """应用高斯光阑
        
        应用透过率函数 T(r) = exp(-0.5 × (r/σ)²)。
        
        参数:
            wfo: PROPER 波前对象
        
        返回:
            光阑透过率掩模数组（物理坐标系，中心在数组中心）
        
        Requirements: 4.2
        """
        import proper
        
        # 获取实际 sigma 参数
        actual_sigma = self._get_actual_sigma(wfo)
        
        # 获取网格参数
        sampling = proper.prop_get_sampling(wfo)
        grid_size = wfo.wfarr.shape[0]
        
        # 创建坐标网格（物理坐标系，中心在数组中心）
        x = (np.arange(grid_size) - grid_size / 2) * sampling
        y = (np.arange(grid_size) - grid_size / 2) * sampling
        X, Y = np.meshgrid(x, y)
        
        # 计算到光阑中心的距离
        R = np.sqrt((X - self.center_x)**2 + (Y - self.center_y)**2)
        
        # 计算高斯透过率掩模
        # T(r) = exp(-0.5 × (r/σ)²)
        mask = np.exp(-0.5 * (R / actual_sigma)**2)
        
        # 应用掩模到波前
        # 注意：wfarr 使用 FFT 坐标系（零频在角落），需要使用 prop_shift_center 转换
        wfo.wfarr *= proper.prop_shift_center(mask)
        
        return mask
    
    def _apply_super_gaussian(self, wfo: "proper.WaveFront") -> np.ndarray:
        """应用超高斯光阑
        
        应用透过率函数 T(r) = exp(-(r/r₀)ⁿ)，其中 n 为超高斯阶数。
        
        注意：
        - n=2 时为类高斯分布（但公式与标准高斯光阑略有不同）
        - n→∞ 时趋近硬边光阑
        
        参数:
            wfo: PROPER 波前对象
        
        返回:
            光阑透过率掩模数组（物理坐标系，中心在数组中心）
        
        Requirements: 4.3, 4.8
        """
        import proper
        
        # 获取实际光阑半径（特征半径 r₀）
        actual_radius = self._get_actual_radius(wfo)
        
        # 获取网格参数
        sampling = proper.prop_get_sampling(wfo)
        grid_size = wfo.wfarr.shape[0]
        
        # 创建坐标网格（物理坐标系，中心在数组中心）
        x = (np.arange(grid_size) - grid_size / 2) * sampling
        y = (np.arange(grid_size) - grid_size / 2) * sampling
        X, Y = np.meshgrid(x, y)
        
        # 计算到光阑中心的距离
        R = np.sqrt((X - self.center_x)**2 + (Y - self.center_y)**2)
        
        # 计算超高斯透过率掩模
        # T(r) = exp(-(r/r₀)ⁿ)
        n = self.super_gaussian_order
        mask = np.exp(-(R / actual_radius)**n)
        
        # 应用掩模到波前
        # 注意：wfarr 使用 FFT 坐标系（零频在角落），需要使用 prop_shift_center 转换
        wfo.wfarr *= proper.prop_shift_center(mask)
        
        return mask
    
    def _apply_eighth_order(self, wfo: "proper.WaveFront") -> np.ndarray:
        """应用 8 阶软边光阑
        
        使用 PROPER 的 prop_8th_order_mask 函数的算法应用 8 阶软边光阑。
        
        注意：PROPER 的 prop_8th_order_mask 原本是一个遮挡器（occulter），
        中心透过率低，边缘透过率高。我们需要将其反转为光阑行为：
        中心透过率高，边缘透过率低。
        
        8 阶光阑基于 sinc 函数的 8 阶透过率分布，具有以下特点：
        - 中心透过率为 max_transmission
        - 在 HWHM（半高半宽）处透过率为 0.5
        - 边缘透过率趋近于 min_transmission
        - 相比硬边光阑，具有更平滑的过渡，减少衍射效应
        
        参数:
            wfo: PROPER 波前对象
        
        返回:
            光阑透过率掩模数组（物理坐标系，中心在数组中心）
        
        Requirements: 4.4, 4.9
        """
        import proper
        
        # 获取实际光阑半径（HWHM - 半高半宽）
        actual_radius = self._get_actual_radius(wfo)
        
        # 获取网格参数
        sampling = proper.prop_get_sampling(wfo)
        grid_size = wfo.wfarr.shape[0]
        fratio = proper.prop_get_fratio(wfo)
        wavelength = proper.prop_get_wavelength(wfo)
        
        # 计算 e 参数（与 PROPER 内部计算一致）
        # hwhm 转换为 lambda/D 单位
        hwhm_nlamd = actual_radius / (fratio * wavelength)
        e = 1.788 / hwhm_nlamd
        
        # 创建坐标网格（物理坐标系，中心在数组中心）
        x = (np.arange(grid_size) - grid_size / 2) * sampling
        y = (np.arange(grid_size) - grid_size / 2) * sampling
        X, Y = np.meshgrid(x, y)
        
        # 计算到光阑中心的距离（考虑偏移）
        R = np.sqrt((X - self.center_x)**2 + (Y - self.center_y)**2)
        
        # 转换为 lambda/D 单位
        c = sampling / (fratio * wavelength)
        r_lamd = R / sampling * c
        
        # 计算 8 阶遮挡器掩模（与 PROPER 内部计算一致）
        # 这是 PROPER prop_8th_order_mask 的核心算法
        ll = 3.0
        mm = 1.0
        occulter_amp = (ll - mm) / ll - proper.prop_sinc(r_lamd * (np.pi * e / ll))**ll + \
                       mm / ll * proper.prop_sinc(r_lamd * (np.pi * e / mm))**mm
        
        # 转换为强度并归一化
        occulter_int = occulter_amp**2
        occulter_int -= np.min(occulter_int)
        occulter_int /= np.max(occulter_int)
        
        # 反转遮挡器为光阑：aperture = 1 - occulter
        # 这样中心透过率高，边缘透过率低
        aperture_int = 1.0 - occulter_int
        
        # 应用透过率范围
        # min_transmission 对应边缘，max_transmission 对应中心
        aperture_int = aperture_int * (self.max_transmission - self.min_transmission) + self.min_transmission
        
        # 转回振幅
        mask = np.sqrt(aperture_int)
        
        # 应用掩模到波前
        # 注意：wfarr 使用 FFT 坐标系（零频在角落），需要使用 prop_shift_center 转换
        wfo.wfarr *= proper.prop_shift_center(mask)
        
        return mask
    
    def calculate_power_transmission(
        self,
        wfo: "proper.WaveFront",
        beam_radius: float,
    ) -> PowerTransmissionResult:
        """计算高斯光束通过光阑后的能量透过率
        
        计算流程：
        1. 获取应用光阑前的输入功率（从波前振幅计算）
        2. 获取实际光阑半径（在应用光阑前获取，因为归一化半径依赖光束参数）
        3. 应用光阑（调用 self.apply()）
        4. 获取应用光阑后的输出功率
        5. 计算实际透过率 = output_power / input_power
        6. 根据光阑类型计算理论透过率
        7. 计算相对误差
        8. 返回 PowerTransmissionResult
        
        参数:
            wfo: PROPER 波前对象
            beam_radius: 光束半径 (m)，用于计算理论透过率
        
        返回:
            PowerTransmissionResult 对象，包含：
            - actual_transmission: 实际透过率
            - theoretical_transmission: 理论透过率
            - relative_error: 相对误差
            - input_power: 输入功率
            - output_power: 输出功率
        
        Requirements: 4.10, 4.11
        """
        import proper
        
        # 获取采样间隔
        sampling = proper.prop_get_sampling(wfo)
        
        # 1. 获取应用光阑前的输入功率
        # 功率计算：P = Σ|E|² × dx²，其中 E 为复振幅，dx 为采样间隔
        amplitude_before = proper.prop_get_amplitude(wfo)
        input_power = np.sum(amplitude_before**2) * sampling**2
        
        # 2. 获取实际光阑半径（在应用光阑前获取，因为归一化半径依赖光束参数）
        actual_radius = self._get_actual_radius(wfo)
        
        # 3. 应用光阑
        self.apply(wfo)
        
        # 4. 获取应用光阑后的输出功率
        amplitude_after = proper.prop_get_amplitude(wfo)
        output_power = np.sum(amplitude_after**2) * sampling**2
        
        # 5. 计算实际透过率
        if input_power > 0:
            actual_transmission = output_power / input_power
        else:
            actual_transmission = 0.0
        
        # 6. 根据光阑类型计算理论透过率
        theoretical_transmission = self._calculate_theoretical_transmission(
            actual_radius, beam_radius
        )
        
        # 7. 计算相对误差
        if theoretical_transmission > 0:
            relative_error = abs(actual_transmission - theoretical_transmission) / theoretical_transmission
        else:
            relative_error = 0.0 if actual_transmission == 0 else float('inf')
        
        # 8. 返回 PowerTransmissionResult
        return PowerTransmissionResult(
            actual_transmission=actual_transmission,
            theoretical_transmission=theoretical_transmission,
            relative_error=relative_error,
            input_power=input_power,
            output_power=output_power,
        )
    
    def _calculate_theoretical_transmission(
        self,
        aperture_radius: float,
        beam_radius: float,
    ) -> float:
        """根据光阑类型计算理论透过率
        
        参数:
            aperture_radius: 光阑半径 (m)
            beam_radius: 光束半径 (m)
        
        返回:
            理论透过率
        
        Requirements: 4.10, 4.11
        """
        if self.aperture_type == ApertureType.HARD_EDGE:
            return self._theoretical_transmission_hard_edge(aperture_radius, beam_radius)
        elif self.aperture_type == ApertureType.GAUSSIAN:
            # 对于高斯光阑，使用实际的 sigma 参数
            actual_sigma = self.gaussian_sigma
            if self.normalized:
                actual_sigma = self.gaussian_sigma * beam_radius
            return self._theoretical_transmission_gaussian(actual_sigma, beam_radius)
        elif self.aperture_type == ApertureType.SUPER_GAUSSIAN:
            return self._theoretical_transmission_super_gaussian(
                aperture_radius, beam_radius, self.super_gaussian_order
            )
        elif self.aperture_type == ApertureType.EIGHTH_ORDER:
            return self._theoretical_transmission_eighth_order(
                aperture_radius, beam_radius
            )
        else:
            # 未知类型，返回 1.0（无损耗）
            return 1.0
    
    def _theoretical_transmission_super_gaussian(
        self,
        aperture_radius: float,
        beam_radius: float,
        order: int,
    ) -> float:
        """计算超高斯光阑的理论透过率
        
        对于超高斯光阑 T(r) = exp(-(r/r₀)ⁿ) 和高斯光束 I(r) = exp(-2(r/w)²)，
        理论透过率需要通过数值积分计算：
        T = ∫∫ I(r) × T(r)² r dr dθ / ∫∫ I(r) r dr dθ
          = ∫₀^∞ exp(-2(r/w)²) × exp(-2(r/r₀)ⁿ) × r dr / ∫₀^∞ exp(-2(r/w)²) × r dr
        
        参数:
            aperture_radius: 光阑特征半径 r₀ (m)
            beam_radius: 光束半径 w (m)
            order: 超高斯阶数 n
        
        返回:
            理论透过率
        """
        from scipy import integrate
        
        # 被积函数（分子）：I(r) × T(r)² × r
        # T(r)² = exp(-2(r/r₀)ⁿ)（振幅透过率的平方 = 强度透过率）
        def integrand_numerator(r):
            gaussian_intensity = np.exp(-2.0 * (r / beam_radius)**2)
            aperture_intensity = np.exp(-2.0 * (r / aperture_radius)**order)
            return gaussian_intensity * aperture_intensity * r
        
        # 被积函数（分母）：I(r) × r
        def integrand_denominator(r):
            gaussian_intensity = np.exp(-2.0 * (r / beam_radius)**2)
            return gaussian_intensity * r
        
        # 积分上限：取足够大的值（10 倍光束半径）
        r_max = 10.0 * beam_radius
        
        # 数值积分
        numerator, _ = integrate.quad(integrand_numerator, 0, r_max)
        denominator, _ = integrate.quad(integrand_denominator, 0, r_max)
        
        if denominator > 0:
            return numerator / denominator
        else:
            return 0.0
    
    def _theoretical_transmission_eighth_order(
        self,
        aperture_radius: float,
        beam_radius: float,
    ) -> float:
        """计算 8 阶光阑的理论透过率
        
        8 阶光阑的透过率分布较为复杂，需要通过数值积分计算。
        这里使用与 _apply_eighth_order 相同的透过率函数进行积分。
        
        参数:
            aperture_radius: 光阑 HWHM 半径 (m)
            beam_radius: 光束半径 (m)
        
        返回:
            理论透过率
        """
        from scipy import integrate
        
        # 8 阶光阑的透过率函数参数
        # 注意：这里简化处理，使用近似的透过率函数
        # 实际的 8 阶光阑透过率依赖于 PROPER 的具体实现
        
        # 使用简化的 8 阶透过率模型：
        # 在 HWHM 处透过率为 0.5，中心为 max_transmission，边缘趋近 min_transmission
        # 近似为：T(r) ≈ (max - min) × (1 - (r/r_hwhm)^8 / (1 + (r/r_hwhm)^8)) + min
        
        def aperture_transmission(r):
            """8 阶光阑的近似强度透过率"""
            x = r / aperture_radius
            # 使用 sigmoid 类型的 8 阶函数
            t_amp = (self.max_transmission - self.min_transmission) / (1.0 + x**8) + self.min_transmission
            return t_amp**2  # 振幅透过率的平方 = 强度透过率
        
        # 被积函数（分子）：I(r) × T(r) × r
        def integrand_numerator(r):
            gaussian_intensity = np.exp(-2.0 * (r / beam_radius)**2)
            return gaussian_intensity * aperture_transmission(r) * r
        
        # 被积函数（分母）：I(r) × r
        def integrand_denominator(r):
            gaussian_intensity = np.exp(-2.0 * (r / beam_radius)**2)
            return gaussian_intensity * r
        
        # 积分上限
        r_max = 10.0 * beam_radius
        
        # 数值积分
        numerator, _ = integrate.quad(integrand_numerator, 0, r_max)
        denominator, _ = integrate.quad(integrand_denominator, 0, r_max)
        
        if denominator > 0:
            return numerator / denominator
        else:
            return 0.0
    
    def _theoretical_transmission_hard_edge(
        self,
        aperture_radius: float,
        beam_radius: float,
    ) -> float:
        """计算硬边光阑的理论透过率
        
        公式: T = 1 - exp(-2 × (a/w)²)
        其中 a 为光阑半径，w 为光束半径。
        
        参数:
            aperture_radius: 光阑半径 (m)
            beam_radius: 光束半径 (m)
        
        返回:
            理论透过率
        
        Requirements: 4.10
        """
        ratio = aperture_radius / beam_radius
        return 1.0 - np.exp(-2.0 * ratio**2)
    
    def _theoretical_transmission_gaussian(
        self,
        sigma: float,
        beam_radius: float,
    ) -> float:
        """计算高斯光阑的理论透过率
        
        对于高斯光阑振幅透过率 T(r) = exp(-0.5 × (r/σ)²) 和高斯光束 I(r) = exp(-2 × (r/w)²)，
        理论透过率为：
        
        T = ∫∫ I(r) × T(r)² r dr dθ / ∫∫ I(r) r dr dθ
        
        其中 T(r)² = exp(-(r/σ)²) 是强度透过率。
        
        推导：
        设 a = 2/w² + 1/σ²，b = 2/w²
        T = b/a = (2/w²) / (2/w² + 1/σ²) = 2σ² / (2σ² + w²)
        
        参数:
            sigma: 高斯光阑的 σ 参数 (m)
            beam_radius: 光束半径 w (m)
        
        返回:
            理论透过率
        """
        w = beam_radius
        s = sigma
        # T = 2σ² / (2σ² + w²)
        return (2.0 * s**2) / (2.0 * s**2 + w**2)
    
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"CircularAperture("
            f"type={self.aperture_type.value}, "
            f"radius={self.radius}, "
            f"normalized={self.normalized}, "
            f"center=({self.center_x}, {self.center_y}))"
        )
