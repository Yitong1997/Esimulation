# -*- coding: utf-8 -*-
"""
工具函数模块

本模块提供角谱法计算所需的工具函数，包括：
- select_region: 选择或扩展数组区域（返回副本）
- select_region_view: 返回数组区域的视图（零填充时创建新数组）

这些函数类似于 Julia NDTools 包的功能，用于：
1. 零填充扩展（expand=True 时将数组扩展到 4 倍大小）
2. 裁剪到原始尺寸（传播后裁剪回原始大小）
3. 居中处理（确保数组中心对齐）

Validates: Requirements 9.1, 9.2
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Union


def _ft_center(size: int) -> int:
    """
    计算 FFT 中心索引（右中心）
    
    对于偶数尺寸 n，中心在 n//2
    对于奇数尺寸 n，中心在 n//2
    
    这与 Julia 的 ft_center_diff(size) + 1 一致
    
    参数：
        size: 数组在某一维度的尺寸
    
    返回：
        FFT 中心索引（0-based）
    """
    return size // 2


def _compute_slice_ranges(
    src_size: Tuple[int, ...],
    dst_size: Tuple[int, ...],
    src_center: Optional[Tuple[int, ...]] = None,
    dst_center: Optional[Tuple[int, ...]] = None
) -> Tuple[Tuple[slice, ...], Tuple[slice, ...], Tuple[int, int, int, int]]:
    """
    计算源数组和目标数组之间的切片范围
    
    参数：
        src_size: 源数组尺寸
        dst_size: 目标数组尺寸
        src_center: 源数组中心（默认为 FFT 中心）
        dst_center: 目标数组中心（默认为 FFT 中心）
    
    返回：
        (src_slices, dst_slices, overlap_info)
        - src_slices: 源数组的切片元组
        - dst_slices: 目标数组的切片元组
        - overlap_info: 重叠区域信息 (src_start_y, src_end_y, src_start_x, src_end_x)
    """
    ndim = len(src_size)
    
    # 默认中心为 FFT 中心
    if src_center is None:
        src_center = tuple(_ft_center(s) for s in src_size)
    if dst_center is None:
        dst_center = tuple(_ft_center(s) for s in dst_size)
    
    src_slices = []
    dst_slices = []
    
    for dim in range(ndim):
        # 计算偏移量：目标中心 - 源中心
        offset = dst_center[dim] - src_center[dim]
        
        # 源数组在目标数组中的起始和结束位置
        dst_start = offset
        dst_end = offset + src_size[dim]
        
        # 源数组的起始和结束位置
        src_start = 0
        src_end = src_size[dim]
        
        # 裁剪到目标数组边界
        if dst_start < 0:
            src_start = -dst_start
            dst_start = 0
        if dst_end > dst_size[dim]:
            src_end = src_size[dim] - (dst_end - dst_size[dim])
            dst_end = dst_size[dim]
        
        # 确保范围有效
        if src_start >= src_end or dst_start >= dst_end:
            # 无重叠区域
            src_slices.append(slice(0, 0))
            dst_slices.append(slice(0, 0))
        else:
            src_slices.append(slice(src_start, src_end))
            dst_slices.append(slice(dst_start, dst_end))
    
    return tuple(src_slices), tuple(dst_slices), None


def select_region(
    arr: NDArray,
    new_size: Tuple[int, ...],
    *,
    center: bool = True,
    pad_value: Union[int, float, complex] = 0
) -> NDArray:
    """
    选择或扩展数组区域（类似 Julia NDTools.select_region）
    
    当 new_size > arr.shape 时，进行零填充扩展，原数组居中放置。
    当 new_size < arr.shape 时，进行裁剪，从中心提取区域。
    
    参数：
        arr: 输入数组（支持任意维度）
        new_size: 目标尺寸，元组形式
        center: 是否居中（默认 True）。如果为 False，则左上角对齐。
        pad_value: 填充值（默认 0）
    
    返回：
        调整后的数组（新分配的副本）
    
    示例：
        >>> arr = np.ones((3, 3))
        >>> select_region(arr, (5, 5))  # 扩展并居中
        array([[0., 0., 0., 0., 0.],
               [0., 1., 1., 1., 0.],
               [0., 1., 1., 1., 0.],
               [0., 1., 1., 1., 0.],
               [0., 0., 0., 0., 0.]])
        
        >>> arr = np.arange(16).reshape(4, 4)
        >>> select_region(arr, (2, 2))  # 裁剪中心区域
        array([[ 5,  6],
               [ 9, 10]])
    
    注意：
        - 此函数总是返回新分配的数组（副本）
        - 对于需要视图的场景，请使用 select_region_view
        - 支持 NumPy 行优先（C 顺序）布局
    
    Validates: Requirements 9.1, 9.2
    """
    # 确保 new_size 是元组
    if isinstance(new_size, int):
        new_size = (new_size,) * arr.ndim
    
    # 如果尺寸相同，直接返回副本
    if arr.shape == tuple(new_size):
        return arr.copy()
    
    # 创建目标数组
    result = np.full(new_size, pad_value, dtype=arr.dtype)
    
    if center:
        # 居中模式：使用 FFT 中心对齐
        src_center = tuple(_ft_center(s) for s in arr.shape)
        dst_center = tuple(_ft_center(s) for s in new_size)
    else:
        # 左上角对齐模式
        src_center = (0,) * arr.ndim
        dst_center = (0,) * arr.ndim
    
    # 计算切片范围
    src_slices, dst_slices, _ = _compute_slice_ranges(
        arr.shape, new_size, src_center, dst_center
    )
    
    # 复制数据
    result[dst_slices] = arr[src_slices]
    
    return result


def select_region_view(
    arr: NDArray,
    new_size: Tuple[int, ...],
    *,
    center: bool = True,
    pad_value: Union[int, float, complex] = 0
) -> NDArray:
    """
    返回数组区域的视图（零填充时创建新数组）
    
    当 new_size <= arr.shape（所有维度）时，返回原数组的视图。
    当 new_size > arr.shape（任意维度）时，创建新数组并填充。
    
    参数：
        arr: 输入数组（支持任意维度）
        new_size: 目标尺寸，元组形式
        center: 是否居中（默认 True）
        pad_value: 填充值（默认 0），仅在需要扩展时使用
    
    返回：
        调整后的数组。如果是纯裁剪操作，返回视图；否则返回新数组。
    
    示例：
        >>> arr = np.arange(16).reshape(4, 4)
        >>> view = select_region_view(arr, (2, 2))  # 返回视图
        >>> view
        array([[ 5,  6],
               [ 9, 10]])
        >>> np.shares_memory(arr, view)  # 共享内存
        True
        
        >>> expanded = select_region_view(arr, (6, 6))  # 返回新数组
        >>> np.shares_memory(arr, expanded)  # 不共享内存
        False
    
    注意：
        - 纯裁剪操作返回视图，修改视图会影响原数组
        - 扩展操作返回新数组，与原数组独立
        - 支持 NumPy 行优先（C 顺序）布局
    
    Validates: Requirements 9.1, 9.2
    """
    # 确保 new_size 是元组
    if isinstance(new_size, int):
        new_size = (new_size,) * arr.ndim
    
    # 如果尺寸相同，返回视图
    if arr.shape == tuple(new_size):
        return arr.view()
    
    # 检查是否可以返回纯视图（所有维度都是裁剪或相等）
    can_return_view = all(ns <= s for ns, s in zip(new_size, arr.shape))
    
    if can_return_view and center:
        # 纯裁剪模式：返回视图
        slices = []
        for dim in range(arr.ndim):
            src_size = arr.shape[dim]
            dst_size = new_size[dim]
            
            # 计算裁剪的起始和结束位置
            # 使用 FFT 中心对齐
            src_center = _ft_center(src_size)
            dst_center = _ft_center(dst_size)
            
            # 从源数组中心提取 dst_size 大小的区域
            start = src_center - dst_center
            end = start + dst_size
            
            slices.append(slice(start, end))
        
        return arr[tuple(slices)]
    
    elif can_return_view and not center:
        # 左上角对齐的裁剪
        slices = tuple(slice(0, ns) for ns in new_size)
        return arr[slices]
    
    else:
        # 需要扩展：创建新数组
        return select_region(arr, new_size, center=center, pad_value=pad_value)
