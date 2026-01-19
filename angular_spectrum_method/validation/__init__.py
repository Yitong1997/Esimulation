# -*- coding: utf-8 -*-
"""
验证模块

本模块提供与 Julia 实现的对比验证功能，用于确保 Python 实现的数值一致性。

主要功能：
- Julia 对比验证
- 相对误差计算
- 能量守恒验证

Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7
"""

from .julia_comparison import JuliaComparison, ComparisonResult

__all__ = ['JuliaComparison', 'ComparisonResult']
