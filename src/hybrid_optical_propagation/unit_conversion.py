-----------------------------------------------------"""
单位转换工具函数

本模块提供混合光学传播系统中使用的单位转换函数。

主要转换：
- 长度单位：mm ↔ m
- 相位单位：OPD（波长数）↔ 相位（弧度）
- 波长单位：μm ↔ mm ↔ m

**Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7**
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Union

# 类型别名
ArrayLike = Union[float, NDArray[np.floating]]


# =============================================================================
# 长度单位转换
# =============================================================================

def mm_to_m(value_mm: ArrayLike) -> ArrayLike:
    """毫米转换为米
    
    参数:
        value_mm: 毫米值
    
    返回:
        米值
    
    **Validates: Requirements 12.3**
    """
    return value_mm * 1e-3


def m_to_mm(value_m: ArrayLike) -> ArrayLike:
    """米转换为毫米
    
    参数:
        value_m: 米值
    
    返回:
        毫米值
    
    **Validates: Requirements 12.3**
    """
    return value_m * 1e3


def um_to_mm(value_um: ArrayLike) -> ArrayLike:
    """微米转换为毫米
    
    参数:
        value_um: 微米值
    
    返回:
        毫米值
    
    **Validates: Requirements 12.4**
    """
    return value_um * 1e-3


def mm_to_um(value_mm: ArrayLike) -> ArrayLike:
    """毫米转换为微米
    
    参数:
        value_mm: 毫米值
    
    返回:
        微米值
    
    **Validates: Requirements 12.4**
    """
    return value_mm * 1e3


def um_to_m(value_um: ArrayLike) -> ArrayLike:
    """微米转换为米
    
    参数:
        value_um: 微米值
    
    返回:
        米值
    
    **Validates: Requirements 12.5**
    """
    return value_um * 1e-6


def m_to_um(value_m: ArrayLike) -> ArrayLike:
    """米转换为微米
    
    参数:
        value_m: 米值
    
    返回:
        微米值
    
    **Validates: Requirements 12.5**
    """
    return value_m * 1e6


# =============================================================================
# 相位单位转换
# =============================================================================

def opd_waves_to_phase_rad(opd_waves: ArrayLike) -> ArrayLike:
    """OPD（波长数）转换为相位（弧度）
    
    公式: phase = 2π × opd_waves
    
    参数:
        opd_waves: OPD 值（波长数）
    
    返回:
        相位值（弧度）
    
    **Validates: Requirements 12.6**
    """
    return 2 * np.pi * opd_waves


def phase_rad_to_opd_waves(phase_rad: ArrayLike) -> ArrayLike:
    """相位（弧度）转换为 OPD（波长数）
    
    公式: opd_waves = phase / (2π)
    
    参数:
        phase_rad: 相位值（弧度）
    
    返回:
        OPD 值（波长数）
    
    **Validates: Requirements 12.6**
    """
    return phase_rad / (2 * np.pi)


def opd_mm_to_phase_rad(opd_mm: ArrayLike, wavelength_mm: float) -> ArrayLike:
    """OPD（毫米）转换为相位（弧度）
    
    公式: phase = 2π × opd_mm / wavelength_mm
    
    参数:
        opd_mm: OPD 值（毫米）
        wavelength_mm: 波长（毫米）
    
    返回:
        相位值（弧度）
    
    **Validates: Requirements 12.7**
    """
    return 2 * np.pi * opd_mm / wavelength_mm


def phase_rad_to_opd_mm(phase_rad: ArrayLike, wavelength_mm: float) -> ArrayLike:
    """相位（弧度）转换为 OPD（毫米）
    
    公式: opd_mm = phase × wavelength_mm / (2π)
    
    参数:
        phase_rad: 相位值（弧度）
        wavelength_mm: 波长（毫米）
    
    返回:
        OPD 值（毫米）
    
    **Validates: Requirements 12.7**
    """
    return phase_rad * wavelength_mm / (2 * np.pi)


# =============================================================================
# 波数计算
# =============================================================================

def wavenumber_mm(wavelength_mm: float) -> float:
    """计算波数（单位 1/mm）
    
    公式: k = 2π / λ
    
    参数:
        wavelength_mm: 波长（毫米）
    
    返回:
        波数（1/mm）
    """
    return 2 * np.pi / wavelength_mm


def wavenumber_m(wavelength_m: float) -> float:
    """计算波数（单位 1/m）
    
    公式: k = 2π / λ
    
    参数:
        wavelength_m: 波长（米）
    
    返回:
        波数（1/m）
    """
    return 2 * np.pi / wavelength_m


# =============================================================================
# 便捷转换函数
# =============================================================================

def wavelength_um_to_mm(wavelength_um: float) -> float:
    """波长从微米转换为毫米
    
    参数:
        wavelength_um: 波长（微米）
    
    返回:
        波长（毫米）
    """
    return wavelength_um * 1e-3


def wavelength_um_to_m(wavelength_um: float) -> float:
    """波长从微米转换为米
    
    参数:
        wavelength_um: 波长（微米）
    
    返回:
        波长（米）
    """
    return wavelength_um * 1e-6


class UnitConverter:
    """单位转换器类
    
    提供基于波长的单位转换方法。
    
    属性:
        wavelength_um: 波长（微米）
        wavelength_mm: 波长（毫米）
        wavelength_m: 波长（米）
        k_mm: 波数（1/mm）
        k_m: 波数（1/m）
    """
    
    def __init__(self, wavelength_um: float):
        """初始化单位转换器
        
        参数:
            wavelength_um: 波长（微米）
        """
        self._wavelength_um = wavelength_um
        self._wavelength_mm = wavelength_um * 1e-3
        self._wavelength_m = wavelength_um * 1e-6
        self._k_mm = 2 * np.pi / self._wavelength_mm
        self._k_m = 2 * np.pi / self._wavelength_m
    
    @property
    def wavelength_um(self) -> float:
        """波长（微米）"""
        return self._wavelength_um
    
    @property
    def wavelength_mm(self) -> float:
        """波长（毫米）"""
        return self._wavelength_mm
    
    @property
    def wavelength_m(self) -> float:
        """波长（米）"""
        return self._wavelength_m
    
    @property
    def k_mm(self) -> float:
        """波数（1/mm）"""
        return self._k_mm
    
    @property
    def k_m(self) -> float:
        """波数（1/m）"""
        return self._k_m
    
    def opd_waves_to_phase(self, opd_waves: ArrayLike) -> ArrayLike:
        """OPD（波长数）转换为相位（弧度）"""
        return opd_waves_to_phase_rad(opd_waves)
    
    def phase_to_opd_waves(self, phase_rad: ArrayLike) -> ArrayLike:
        """相位（弧度）转换为 OPD（波长数）"""
        return phase_rad_to_opd_waves(phase_rad)
    
    def opd_mm_to_phase(self, opd_mm: ArrayLike) -> ArrayLike:
        """OPD（毫米）转换为相位（弧度）"""
        return opd_mm_to_phase_rad(opd_mm, self._wavelength_mm)
    
    def phase_to_opd_mm(self, phase_rad: ArrayLike) -> ArrayLike:
        """相位（弧度）转换为 OPD（毫米）"""
        return phase_rad_to_opd_mm(phase_rad, self._wavelength_mm)

