"""
验证器模块

本模块实现混合光学追迹的验证器，用于检查相位连续性和能量守恒。

主要验证器：
- PhaseContinuityValidator: 相位连续性验证器
- EnergyConservationValidator: 能量守恒验证器

**Feature: hybrid-raytracing-validation**
**Validates: Requirements 2.5, 5.1, 5.2, 7.2**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class ValidationResult:
    """验证结果
    
    属性:
        is_valid: 验证是否通过
        message: 结果消息
        details: 详细信息字典
        warnings: 警告列表
    """
    is_valid: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class PhaseContinuityValidator:
    """相位连续性验证器
    
    验证相位网格相邻像素的相位差是否小于 π。
    
    **Feature: hybrid-raytracing-validation**
    **Validates: Requirements 2.5, 7.2**
    """
    
    def __init__(self, threshold_rad: float = np.pi) -> None:
        """初始化验证器
        
        参数:
            threshold_rad: 相位差阈值（弧度），默认 π
        """
        self._threshold_rad = threshold_rad
    
    @property
    def threshold_rad(self) -> float:
        """相位差阈值（弧度）"""
        return self._threshold_rad
    
    def validate_phase_grid(
        self,
        phase_grid: NDArray[np.floating],
        valid_mask: Optional[NDArray[np.bool_]] = None,
        grid_name: str = "phase_grid",
    ) -> ValidationResult:
        """验证相位网格的连续性
        
        检查相邻像素的相位差是否小于阈值。
        
        参数:
            phase_grid: 相位网格（弧度）
            valid_mask: 有效区域掩模（可选）
            grid_name: 网格名称（用于报告）
        
        返回:
            ValidationResult 对象
        
        **Validates: Requirements 2.5, 7.2**
        """
        # 计算 x 方向相邻像素的相位差
        phase_diff_x = np.abs(np.diff(phase_grid, axis=1))
        # 计算 y 方向相邻像素的相位差
        phase_diff_y = np.abs(np.diff(phase_grid, axis=0))
        
        # 如果有有效区域掩模，只检查有效区域
        if valid_mask is not None:
            valid_x = valid_mask[:, :-1] & valid_mask[:, 1:]
            valid_y = valid_mask[:-1, :] & valid_mask[1:, :]
            max_diff_x = np.max(phase_diff_x[valid_x]) if np.any(valid_x) else 0.0
            max_diff_y = np.max(phase_diff_y[valid_y]) if np.any(valid_y) else 0.0
        else:
            max_diff_x = np.max(phase_diff_x) if phase_diff_x.size > 0 else 0.0
            max_diff_y = np.max(phase_diff_y) if phase_diff_y.size > 0 else 0.0
        
        max_phase_diff = max(max_diff_x, max_diff_y)
        is_valid = max_phase_diff < self._threshold_rad
        
        # 找到不连续位置
        discontinuity_locations: List[Tuple[int, int]] = []
        if not is_valid:
            if valid_mask is not None:
                locs_x = np.argwhere((phase_diff_x > self._threshold_rad) & valid_x)
                locs_y = np.argwhere((phase_diff_y > self._threshold_rad) & valid_y)
            else:
                locs_x = np.argwhere(phase_diff_x > self._threshold_rad)
                locs_y = np.argwhere(phase_diff_y > self._threshold_rad)
            discontinuity_locations = [tuple(loc) for loc in locs_x[:10]]
            discontinuity_locations.extend([tuple(loc) for loc in locs_y[:10]])
        
        return ValidationResult(
            is_valid=is_valid,
            message=f"{grid_name}: 最大相位差 = {max_phase_diff:.4f} rad "
                    f"({'通过' if is_valid else '失败'})",
            details={
                'max_phase_diff_rad': float(max_phase_diff),
                'max_phase_diff_waves': float(max_phase_diff / (2 * np.pi)),
                'threshold_rad': self._threshold_rad,
                'discontinuity_locations': discontinuity_locations,
            },
        )
    
    def validate_wavefront_error_range(
        self,
        wavefront_error: NDArray[np.floating],
        valid_mask: Optional[NDArray[np.bool_]] = None,
        expected_max_rad: float = 1.0,
    ) -> ValidationResult:
        """验证波前误差的取值范围
        
        波前误差（相对于 Pilot Beam 的相位偏差）应该很小，
        代表像差而非整体曲率。
        
        参数:
            wavefront_error: 波前误差（弧度）
            valid_mask: 有效区域掩模（可选）
            expected_max_rad: 期望的最大值（弧度）
        
        返回:
            ValidationResult 对象
        """
        if valid_mask is not None:
            valid_error = wavefront_error[valid_mask]
        else:
            valid_error = wavefront_error.ravel()
        
        if valid_error.size == 0:
            return ValidationResult(
                is_valid=True,
                message="波前误差: 无有效数据",
                details={},
            )
        
        max_error = float(np.max(np.abs(valid_error)))
        rms_error = float(np.sqrt(np.mean(valid_error ** 2)))
        
        is_valid = max_error < expected_max_rad
        
        return ValidationResult(
            is_valid=is_valid,
            message=f"波前误差: RMS = {rms_error:.4f} rad, "
                    f"Max = {max_error:.4f} rad "
                    f"({'通过' if is_valid else '失败'})",
            details={
                'rms_error_rad': rms_error,
                'max_error_rad': max_error,
                'rms_error_waves': rms_error / (2 * np.pi),
                'max_error_waves': max_error / (2 * np.pi),
                'expected_max_rad': expected_max_rad,
            },
        )


class EnergyConservationValidator:
    """能量守恒验证器
    
    验证雅可比矩阵计算振幅时的能量损失。
    
    **Feature: hybrid-raytracing-validation**
    **Validates: Requirements 5.1, 5.2**
    """
    
    def __init__(self, tolerance: float = 0.01) -> None:
        """初始化验证器
        
        参数:
            tolerance: 能量损失容差（相对值），默认 1%
        """
        self._tolerance = tolerance
    
    @property
    def tolerance(self) -> float:
        """能量损失容差"""
        return self._tolerance
    
    def compute_total_energy(
        self,
        amplitude: NDArray[np.floating],
        pixel_area: float = 1.0,
    ) -> float:
        """计算总能量
        
        总能量 = Σ |A|² × pixel_area
        
        参数:
            amplitude: 振幅数组
            pixel_area: 像素面积
        
        返回:
            总能量（任意单位）
        """
        intensity = np.abs(amplitude) ** 2
        return float(np.sum(intensity) * pixel_area)
    
    def validate_energy_conservation(
        self,
        amplitude_before: NDArray[np.floating],
        amplitude_after: NDArray[np.floating],
        pixel_area_before: float = 1.0,
        pixel_area_after: float = 1.0,
    ) -> ValidationResult:
        """验证能量守恒
        
        检查传播前后总能量是否相等（相对误差 < tolerance）。
        
        参数:
            amplitude_before: 传播前振幅
            amplitude_after: 传播后振幅
            pixel_area_before: 传播前像素面积
            pixel_area_after: 传播后像素面积
        
        返回:
            ValidationResult 对象
        
        **Validates: Requirements 5.1, 5.2**
        """
        energy_before = self.compute_total_energy(amplitude_before, pixel_area_before)
        energy_after = self.compute_total_energy(amplitude_after, pixel_area_after)
        
        if energy_before > 0:
            relative_change = (energy_after - energy_before) / energy_before
            energy_loss = -relative_change if relative_change < 0 else 0.0
            energy_gain = relative_change if relative_change > 0 else 0.0
            is_valid = abs(relative_change) < self._tolerance
        else:
            relative_change = 0.0
            energy_loss = 0.0
            energy_gain = 0.0
            is_valid = energy_after == 0
        
        return ValidationResult(
            is_valid=is_valid,
            message=f"能量守恒: 变化 = {relative_change*100:.2f}% "
                    f"({'通过' if is_valid else '失败'})",
            details={
                'energy_before': energy_before,
                'energy_after': energy_after,
                'relative_change': relative_change,
                'energy_loss': energy_loss,
                'energy_gain': energy_gain,
                'tolerance': self._tolerance,
            },
        )
    
    def validate_jacobian_amplitude(
        self,
        jacobian_det: NDArray[np.floating],
        amplitude: NDArray[np.floating],
        valid_mask: NDArray[np.bool_],
    ) -> ValidationResult:
        """验证雅可比矩阵振幅计算
        
        检查振幅是否符合公式 A = 1/sqrt(|J|)。
        
        参数:
            jacobian_det: 雅可比行列式
            amplitude: 计算的振幅
            valid_mask: 有效光线掩模
        
        返回:
            ValidationResult 对象
        """
        # 只检查有效区域
        valid_jacobian = jacobian_det[valid_mask]
        valid_amplitude = amplitude[valid_mask]
        
        if valid_jacobian.size == 0:
            return ValidationResult(
                is_valid=True,
                message="雅可比振幅: 无有效数据",
                details={},
            )
        
        # 期望振幅 = 1/sqrt(|J|)
        expected_amplitude = 1.0 / np.sqrt(np.maximum(valid_jacobian, 1e-10))
        
        # 归一化后比较
        mean_expected = np.mean(expected_amplitude)
        mean_actual = np.mean(valid_amplitude)
        
        if mean_expected > 0:
            expected_normalized = expected_amplitude / mean_expected
        else:
            expected_normalized = expected_amplitude
        
        if mean_actual > 0:
            actual_normalized = valid_amplitude / mean_actual
        else:
            actual_normalized = valid_amplitude
        
        # 计算相对误差
        relative_error = np.abs(actual_normalized - expected_normalized)
        max_error = float(np.max(relative_error))
        rms_error = float(np.sqrt(np.mean(relative_error ** 2)))
        
        is_valid = max_error < 0.01  # 1% 容差
        
        return ValidationResult(
            is_valid=is_valid,
            message=f"雅可比振幅: RMS误差 = {rms_error:.4f}, "
                    f"Max误差 = {max_error:.4f} "
                    f"({'通过' if is_valid else '失败'})",
            details={
                'rms_error': rms_error,
                'max_error': max_error,
            },
        )
