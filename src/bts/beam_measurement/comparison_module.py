# -*- coding: utf-8 -*-
"""
测量结果与理论对比模块

本模块提供测量结果与理论预期的对比分析功能，包括：
1. 计算理论光束直径
2. 对比测量值与理论值
3. 计算菲涅尔数
4. 估算衍射效应

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import numpy as np
from typing import Optional

from .data_models import ComparisonResult
from .exceptions import InvalidInputError


class ComparisonModule:
    """测量结果与理论对比模块
    
    提供测量结果与理论预期的对比分析功能。
    
    主要功能：
    - 计算理论光束直径 w(z) = w₀ × √(1 + (z/z_R)²)
    - 对比测量光束直径与理论值
    - 计算菲涅尔数 N_F = a² / (λ × z)
    - 基于菲涅尔数估算衍射效应
    
    Attributes:
        wavelength: 波长 (m)
        w0: 束腰半径 (m)
        z_rayleigh: 瑞利距离 (m)，计算为 π × w₀² / λ
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """
    
    def __init__(self, wavelength: float, w0: float):
        """初始化对比模块
        
        参数:
            wavelength: 波长 (m)
            w0: 束腰半径 (m)
        
        Raises:
            InvalidInputError: 当波长或束腰半径为零或负时
        """
        if wavelength <= 0:
            raise InvalidInputError(f"波长必须为正数，当前值: {wavelength}")
        if w0 <= 0:
            raise InvalidInputError(f"束腰半径必须为正数，当前值: {w0}")
        
        self.wavelength = wavelength
        self.w0 = w0
        # 计算瑞利距离: z_R = π × w₀² / λ
        self.z_rayleigh = np.pi * w0**2 / wavelength
    
    def theoretical_beam_diameter(self, z: float) -> float:
        """计算理论光束直径
        
        使用高斯光束传播公式计算理论光束直径（D4sigma）。
        
        公式:
            w(z) = w₀ × √(1 + (z/z_R)²)
            D4sigma = 2 × w(z)
        
        其中:
            w₀: 束腰半径
            z_R: 瑞利距离 = π × w₀² / λ
            z: 相对于束腰的传输距离
        
        参数:
            z: 相对于束腰的传输距离 (m)
        
        返回:
            理论光束直径 D4sigma (m)
        
        Requirements: 8.1
        """
        # 计算光斑半径 w(z) = w₀ × √(1 + (z/z_R)²)
        w_z = self.w0 * np.sqrt(1 + (z / self.z_rayleigh)**2)
        # D4sigma = 2 × w(z)
        return 2 * w_z
    
    def compare_beam_diameters(
        self,
        z_positions: np.ndarray,
        measured_diameters: np.ndarray,
    ) -> ComparisonResult:
        """对比测量光束直径与理论值
        
        计算每个位置的理论光束直径，并与测量值进行对比分析。
        
        参数:
            z_positions: 传输位置数组 (m)，相对于束腰的距离
            measured_diameters: 测量的光束直径数组 (m)
        
        返回:
            ComparisonResult 对象，包含：
            - measured_values: 测量值数组
            - theoretical_values: 理论值数组
            - relative_errors: 相对误差数组
            - rms_error: RMS 误差
            - max_error: 最大误差
            - fresnel_number: None（此方法不计算菲涅尔数）
        
        Raises:
            InvalidInputError: 当输入数组为空或长度不匹配时
        
        Requirements: 8.2
        """
        # 输入验证
        z_positions = np.asarray(z_positions)
        measured_diameters = np.asarray(measured_diameters)
        
        if z_positions.size == 0:
            raise InvalidInputError("传输位置数组不能为空")
        if measured_diameters.size == 0:
            raise InvalidInputError("测量直径数组不能为空")
        if z_positions.shape != measured_diameters.shape:
            raise InvalidInputError(
                f"数组长度不匹配: z_positions={z_positions.shape}, "
                f"measured_diameters={measured_diameters.shape}"
            )
        
        # 计算每个位置的理论光束直径
        theoretical_values = np.array([
            self.theoretical_beam_diameter(z) for z in z_positions
        ])
        
        # 计算相对误差
        # 避免除以零：当理论值为零时，使用绝对误差
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_errors = np.abs(
                (measured_diameters - theoretical_values) / theoretical_values
            )
            # 处理理论值为零的情况
            relative_errors = np.where(
                theoretical_values == 0,
                np.abs(measured_diameters),
                relative_errors
            )
        
        # 计算 RMS 误差
        rms_error = np.sqrt(np.mean(relative_errors**2))
        
        # 计算最大误差
        max_error = np.max(relative_errors)
        
        return ComparisonResult(
            measured_values=measured_diameters,
            theoretical_values=theoretical_values,
            relative_errors=relative_errors,
            rms_error=rms_error,
            max_error=max_error,
            fresnel_number=None,
        )
    
    def calculate_fresnel_number(
        self,
        aperture_radius: float,
        propagation_distance: float,
    ) -> float:
        """计算菲涅尔数
        
        菲涅尔数用于判断衍射效应的类型和强度。
        
        公式:
            N_F = a² / (λ × z)
        
        其中:
            a: 光阑半径
            λ: 波长
            z: 传播距离
        
        参数:
            aperture_radius: 光阑半径 (m)
            propagation_distance: 传播距离 (m)
        
        返回:
            菲涅尔数 N_F（无量纲）
        
        Raises:
            InvalidInputError: 当光阑半径或传播距离为零或负时
        
        Requirements: 8.3
        """
        if aperture_radius <= 0:
            raise InvalidInputError(
                f"光阑半径必须为正数，当前值: {aperture_radius}"
            )
        if propagation_distance <= 0:
            raise InvalidInputError(
                f"传播距离必须为正数，当前值: {propagation_distance}"
            )
        
        # N_F = a² / (λ × z)
        fresnel_number = aperture_radius**2 / (self.wavelength * propagation_distance)
        
        return fresnel_number
    
    def estimate_diffraction_effect(
        self,
        fresnel_number: float,
    ) -> str:
        """基于菲涅尔数估算衍射效应
        
        根据菲涅尔数的大小判断衍射效应的类型：
        - N_F > 10: 几何光学区域，衍射效应可忽略
        - 1 < N_F ≤ 10: 菲涅尔衍射区域
        - N_F ≤ 1: 夫琅禾费衍射区域
        
        参数:
            fresnel_number: 菲涅尔数
        
        返回:
            衍射效应描述字符串
        
        Raises:
            InvalidInputError: 当菲涅尔数为负时
        
        Requirements: 8.4
        """
        if fresnel_number < 0:
            raise InvalidInputError(
                f"菲涅尔数不能为负，当前值: {fresnel_number}"
            )
        
        if fresnel_number > 10:
            return (
                "几何光学区域 (N_F > 10): "
                "衍射效应可忽略，光束传播可用几何光学近似描述。"
            )
        elif fresnel_number > 1:
            return (
                "菲涅尔衍射区域 (1 < N_F ≤ 10): "
                "存在中等程度的衍射效应，需要考虑近场衍射。"
            )
        else:
            return (
                "夫琅禾费衍射区域 (N_F ≤ 1): "
                "衍射效应显著，光束传播需要用远场衍射理论描述。"
            )
