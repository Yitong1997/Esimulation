"""
相位修正模块

本模块实现相位修正功能，用于从光线追迹的 OPD 中提取残差相位。

主要功能：
- 在光线位置插值参考相位
- 计算残差相位（实际相位 - 参考相位）
- 修正光线相位
- 检查残差相位范围

作者：混合光学仿真项目
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional
from scipy.interpolate import RegularGridInterpolator
import warnings


class PhaseCorrector:
    """相位修正器
    
    计算并应用相位修正，从光线追迹的 OPD 中提取残差相位。
    
    参数:
        reference_phase: 参考相位网格，形状 (N, N)，单位弧度
        x_coords: x 坐标数组，单位 mm
        y_coords: y 坐标数组，单位 mm
    
    示例:
        >>> import numpy as np
        >>> # 创建参考相位网格
        >>> n = 64
        >>> x = np.linspace(-10, 10, n)
        >>> y = np.linspace(-10, 10, n)
        >>> X, Y = np.meshgrid(x, y)
        >>> reference_phase = 0.1 * (X**2 + Y**2)
        >>> 
        >>> # 创建相位修正器
        >>> corrector = PhaseCorrector(reference_phase, x, y)
        >>> 
        >>> # 在光线位置插值参考相位
        >>> ray_x = np.array([0.0, 1.0, 2.0])
        >>> ray_y = np.array([0.0, 1.0, 2.0])
        >>> ref_phase_at_rays = corrector.interpolate_reference_phase(ray_x, ray_y)
    
    **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
    """
    
    def __init__(
        self,
        reference_phase: NDArray[np.floating],
        x_coords: NDArray[np.floating],
        y_coords: NDArray[np.floating],
    ):
        """初始化相位修正器
        
        参数:
            reference_phase: 参考相位网格，形状 (N, N)，单位弧度
            x_coords: x 坐标数组，单位 mm
            y_coords: y 坐标数组，单位 mm
        """
        self.reference_phase = reference_phase
        self.x_coords = x_coords
        self.y_coords = y_coords
        
        # 创建插值器（使用双线性插值）
        # 注意：RegularGridInterpolator 期望 (y, x) 顺序的网格
        self._interpolator = RegularGridInterpolator(
            (y_coords, x_coords),
            reference_phase,
            method='linear',
            bounds_error=False,
            fill_value=np.nan,
        )
    
    def interpolate_reference_phase(
        self,
        ray_x: NDArray[np.floating],
        ray_y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """在光线位置插值参考相位
        
        使用双线性插值获取每条光线位置的参考相位值。
        
        参数:
            ray_x: 光线 x 坐标数组，单位 mm
            ray_y: 光线 y 坐标数组，单位 mm
        
        返回:
            参考相位数组（弧度），与输入光线数组形状相同
        
        **Validates: Requirements 6.1**
        """
        # 构造插值点 (y, x) 顺序
        points = np.column_stack([ray_y.ravel(), ray_x.ravel()])
        
        # 执行插值
        result = self._interpolator(points)
        
        # 恢复原始形状
        return result.reshape(ray_x.shape).astype(np.float64)
    
    def compute_residual_phase(
        self,
        ray_phase: NDArray[np.floating],
        reference_phase_at_rays: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """计算残差相位
        
        残差相位 = 实际相位 - 参考相位
        结果包裹到 [-π, π] 范围。
        
        参数:
            ray_phase: 光线实际相位数组（弧度）
            reference_phase_at_rays: 光线位置的参考相位数组（弧度）
        
        返回:
            残差相位数组（弧度），包裹到 [-π, π] 范围
        
        **Validates: Requirements 6.2, 6.4**
        """
        # 计算残差
        residual = ray_phase - reference_phase_at_rays
        
        # 包裹到 [-π, π] 范围
        residual_wrapped = self._wrap_phase(residual)
        
        return residual_wrapped.astype(np.float64)
    
    @staticmethod
    def _wrap_phase(phase: NDArray[np.floating]) -> NDArray[np.floating]:
        """将相位包裹到 [-π, π] 范围
        
        参数:
            phase: 相位数组（弧度）
        
        返回:
            包裹后的相位数组（弧度）
        """
        return np.angle(np.exp(1j * phase))
    
    def correct_ray_phase(
        self,
        ray_opd_waves: NDArray[np.floating],
        residual_phase: NDArray[np.floating],
        wavelength: float,
    ) -> NDArray[np.floating]:
        """修正光线相位
        
        从光线 OPD 中减去残差相位对应的 OPD。
        
        参数:
            ray_opd_waves: 光线 OPD 数组（波长数）
            residual_phase: 残差相位数组（弧度）
            wavelength: 波长，单位 μm（用于单位转换，但这里不需要）
        
        返回:
            修正后的 OPD 数组（波长数）
        
        **Validates: Requirements 6.3**
        """
        # 将残差相位转换为波长数
        # 相位（弧度）= 2π × OPD（波长数）
        # OPD（波长数）= 相位（弧度）/ (2π)
        residual_opd_waves = residual_phase / (2 * np.pi)
        
        # 从原始 OPD 中减去残差
        corrected_opd = ray_opd_waves - residual_opd_waves
        
        return corrected_opd.astype(np.float64)
    
    def check_residual_range(
        self,
        residual_phase: NDArray[np.floating],
    ) -> Tuple[bool, List[str]]:
        """检查残差相位范围
        
        如果残差超出 [-π, π]，返回警告信息。
        注意：由于 compute_residual_phase 已经进行了相位包裹，
        这里主要检查包裹前的原始残差是否过大。
        
        参数:
            residual_phase: 残差相位数组（弧度）
        
        返回:
            (is_valid, warnings): 是否有效和警告信息列表
        
        **Validates: Requirements 6.5**
        """
        warnings_list: List[str] = []
        
        # 检查是否有 NaN 值
        nan_count = np.sum(np.isnan(residual_phase))
        if nan_count > 0:
            warnings_list.append(
                f"残差相位中有 {nan_count} 个 NaN 值，可能是光线超出参考相位网格范围"
            )
        
        # 检查有效值的范围
        valid_residual = residual_phase[~np.isnan(residual_phase)]
        if len(valid_residual) > 0:
            max_abs_residual = np.max(np.abs(valid_residual))
            
            # 如果最大绝对值接近 π，说明可能有相位跳变
            if max_abs_residual > 0.9 * np.pi:
                warnings_list.append(
                    f"残差相位最大绝对值 {max_abs_residual:.4f} rad 接近 π，"
                    "可能存在相位跳变或参考相位不准确"
                )
        
        is_valid = len(warnings_list) == 0
        
        return is_valid, warnings_list
    
    def correct_rays(
        self,
        ray_x: NDArray[np.floating],
        ray_y: NDArray[np.floating],
        ray_opd_waves: NDArray[np.floating],
        wavelength: float,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], List[str]]:
        """完整的光线相位修正流程
        
        执行完整的相位修正流程：
        1. 在光线位置插值参考相位
        2. 计算残差相位
        3. 修正光线 OPD
        4. 检查残差范围
        
        参数:
            ray_x: 光线 x 坐标数组，单位 mm
            ray_y: 光线 y 坐标数组，单位 mm
            ray_opd_waves: 光线 OPD 数组（波长数）
            wavelength: 波长，单位 μm
        
        返回:
            (corrected_opd, residual_phase, warnings):
            - corrected_opd: 修正后的 OPD（波长数）
            - residual_phase: 残差相位（弧度）
            - warnings: 警告信息列表
        """
        # 1. 插值参考相位
        ref_phase_at_rays = self.interpolate_reference_phase(ray_x, ray_y)
        
        # 2. 将光线 OPD 转换为相位
        ray_phase = 2 * np.pi * ray_opd_waves
        
        # 3. 计算残差相位
        residual_phase = self.compute_residual_phase(ray_phase, ref_phase_at_rays)
        
        # 4. 修正光线 OPD
        corrected_opd = self.correct_ray_phase(ray_opd_waves, residual_phase, wavelength)
        
        # 5. 检查残差范围
        _, warnings_list = self.check_residual_range(residual_phase)
        
        return corrected_opd, residual_phase, warnings_list
