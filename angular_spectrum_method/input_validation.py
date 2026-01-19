# -*- coding: utf-8 -*-
"""
输入验证模块

本模块提供输入参数验证功能，确保传入角谱法函数的参数有效。

主要函数：
- validate_field: 验证输入光场
- validate_positive: 验证正数参数
- validate_rotation_matrix: 验证旋转矩阵

异常类：
- ValidationError: 输入验证错误

Validates: Requirements 8.7
"""

import numpy as np
from numpy.typing import NDArray


class ValidationError(Exception):
    """
    输入验证错误
    
    当输入参数不满足要求时抛出此异常。
    """
    pass


def validate_field(u: np.ndarray, name: str = "u") -> None:
    """
    验证输入光场
    
    检查输入光场是否满足以下条件：
    1. 是 2D 数组
    2. 是复数类型
    3. 不包含 NaN 或 Inf
    
    参数：
        u: 输入光场数组
        name: 参数名称（用于错误消息）
    
    异常：
        ValidationError: 如果输入不满足条件
    
    示例：
        >>> import numpy as np
        >>> from angular_spectrum_method.validation import validate_field
        >>> 
        >>> # 有效输入
        >>> u = np.ones((32, 32), dtype=complex)
        >>> validate_field(u)  # 不抛出异常
        >>> 
        >>> # 无效输入（非 2D）
        >>> u_1d = np.ones(32, dtype=complex)
        >>> validate_field(u_1d)  # 抛出 ValidationError
    
    Validates: Requirements 8.7
    """
    if not isinstance(u, np.ndarray):
        raise ValidationError(f"{name} 必须是 NumPy 数组，实际类型: {type(u)}")
    
    if u.ndim != 2:
        raise ValidationError(f"{name} 必须是 2D 数组，实际维度: {u.ndim}")
    
    if not np.issubdtype(u.dtype, np.complexfloating):
        raise ValidationError(f"{name} 必须是复数类型，实际类型: {u.dtype}")
    
    if not np.all(np.isfinite(u)):
        raise ValidationError(f"{name} 包含 NaN 或 Inf 值")


def validate_positive(value: float, name: str) -> None:
    """
    验证正数参数
    
    检查参数是否为正数（> 0）。
    
    参数：
        value: 要验证的值
        name: 参数名称（用于错误消息）
    
    异常：
        ValidationError: 如果值不是正数
    
    示例：
        >>> from angular_spectrum_method.validation import validate_positive
        >>> 
        >>> validate_positive(633e-9, "wavelength")  # 不抛出异常
        >>> validate_positive(-1.0, "wavelength")    # 抛出 ValidationError
    
    Validates: Requirements 8.7
    """
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValidationError(f"{name} 必须是数值类型，实际类型: {type(value)}")
    
    if not np.isfinite(value):
        raise ValidationError(f"{name} 必须是有限数值，实际值: {value}")
    
    if value <= 0:
        raise ValidationError(f"{name} 必须为正数，实际值: {value}")


def validate_non_negative(value: float, name: str) -> None:
    """
    验证非负数参数
    
    检查参数是否为非负数（>= 0）。
    
    参数：
        value: 要验证的值
        name: 参数名称（用于错误消息）
    
    异常：
        ValidationError: 如果值为负数
    
    Validates: Requirements 8.7
    """
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValidationError(f"{name} 必须是数值类型，实际类型: {type(value)}")
    
    if not np.isfinite(value):
        raise ValidationError(f"{name} 必须是有限数值，实际值: {value}")
    
    if value < 0:
        raise ValidationError(f"{name} 必须为非负数，实际值: {value}")


def validate_rotation_matrix(T: np.ndarray, name: str = "T") -> None:
    """
    验证旋转矩阵
    
    检查旋转矩阵是否满足以下条件：
    1. 是 3×3 矩阵
    2. 是正交矩阵（T @ T.T ≈ I）
    3. 行列式为 1（det(T) ≈ 1）
    
    参数：
        T: 旋转矩阵
        name: 参数名称（用于错误消息）
    
    异常：
        ValidationError: 如果矩阵不是有效的旋转矩阵
    
    示例：
        >>> import numpy as np
        >>> from scipy.spatial.transform import Rotation
        >>> from angular_spectrum_method.validation import validate_rotation_matrix
        >>> 
        >>> # 有效旋转矩阵
        >>> T = Rotation.from_euler('x', 5, degrees=True).as_matrix()
        >>> validate_rotation_matrix(T)  # 不抛出异常
        >>> 
        >>> # 无效矩阵（非正交）
        >>> T_invalid = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        >>> validate_rotation_matrix(T_invalid)  # 抛出 ValidationError
    
    Validates: Requirements 8.7
    """
    if not isinstance(T, np.ndarray):
        raise ValidationError(f"{name} 必须是 NumPy 数组，实际类型: {type(T)}")
    
    if T.shape != (3, 3):
        raise ValidationError(f"{name} 必须是 3×3 矩阵，实际形状: {T.shape}")
    
    if not np.all(np.isfinite(T)):
        raise ValidationError(f"{name} 包含 NaN 或 Inf 值")
    
    # 检查正交性: T @ T.T ≈ I
    identity = T @ T.T
    if not np.allclose(identity, np.eye(3), atol=1e-10):
        raise ValidationError(f"{name} 必须是正交矩阵（T @ T.T ≈ I）")
    
    # 检查行列式: det(T) ≈ 1
    det = np.linalg.det(T)
    if not np.isclose(det, 1.0, atol=1e-10):
        raise ValidationError(f"{name} 的行列式必须为 1，实际值: {det}")


def validate_scale_factor(R: float, name: str = "R") -> None:
    """
    验证缩放因子
    
    检查缩放因子是否为非零有限数。
    
    参数：
        R: 缩放因子
        name: 参数名称（用于错误消息）
    
    异常：
        ValidationError: 如果缩放因子无效
    
    Validates: Requirements 8.7
    """
    if not isinstance(R, (int, float, np.integer, np.floating)):
        raise ValidationError(f"{name} 必须是数值类型，实际类型: {type(R)}")
    
    if not np.isfinite(R):
        raise ValidationError(f"{name} 必须是有限数值，实际值: {R}")
    
    if R == 0:
        raise ValidationError(f"{name} 不能为零")
