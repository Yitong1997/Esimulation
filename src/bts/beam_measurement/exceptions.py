# -*- coding: utf-8 -*-
"""
光束测量异常类

本模块定义光束测量相关的异常类型。

Requirements: 2.5, 3.4
"""


class BeamMeasurementError(Exception):
    """光束测量基础异常
    
    所有光束测量相关异常的基类。
    """
    pass


class InvalidInputError(BeamMeasurementError):
    """无效输入错误
    
    当输入数据不满足要求时抛出，例如：
    - 空数组输入
    - 采样间隔为零或负
    - 无效光阑类型
    - 光阑半径为零或负
    """
    pass


class ConvergenceError(BeamMeasurementError):
    """收敛失败错误
    
    当迭代算法未能在最大次数内收敛时抛出，例如：
    - ISO D4sigma 迭代不收敛
    """
    pass


class InsufficientDataError(BeamMeasurementError):
    """数据不足错误
    
    当提供的数据点数量不足以进行计算时抛出，例如：
    - M² 测量点数少于 5 个
    """
    pass
