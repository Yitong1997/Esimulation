# -*- coding: utf-8 -*-
"""
ISO 11146 标准 D4sigma 计算器

本模块实现 ISO 11146 标准的 D4sigma 测量方法，包括：
1. 背景噪声估计与去除
2. 迭代 ROI 方法确定有效测量区域
3. 使用 3 倍 D4sigma 作为 ROI 边界进行迭代

ISO 11146 标准方法相比理想方法更接近实际光束测量仪的测量结果，
因为它考虑了背景噪声和有限测量区域的影响。

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
"""

from typing import Union, Optional, Tuple
import numpy as np

from .data_models import ISOD4sigmaResult
from .exceptions import InvalidInputError


class ISOD4sigmaCalculator:
    """ISO 11146 标准 D4sigma 计算器
    
    实现 ISO 11146 标准的 D4sigma 测量方法，包括：
    1. 背景噪声估计与去除
    2. 迭代 ROI 方法确定有效测量区域
    3. 使用 3 倍 D4sigma 作为 ROI 边界进行迭代
    
    ISO 11146 标准方法的核心思想：
    - 首先估计并去除背景噪声，避免噪声对二阶矩计算的影响
    - 使用迭代 ROI 方法，逐步收敛到最佳测量区域
    - ROI 边界设为 3 倍 D4sigma，确保包含 99.7% 以上的光束能量
    
    Example:
        >>> calculator = ISOD4sigmaCalculator(
        ...     max_iterations=10,
        ...     convergence_threshold=0.01,
        ...     roi_factor=3.0,
        ... )
        >>> result = calculator.calculate(complex_amplitude, sampling=1e-6)
        >>> print(f"D4sigma: {result.d_mean * 1e3:.3f} mm")
        >>> print(f"迭代次数: {result.iterations}")
        >>> print(f"是否收敛: {result.converged}")
    
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
    """
    
    def __init__(
        self,
        max_iterations: int = 10,
        convergence_threshold: float = 0.01,
        roi_factor: float = 3.0,
    ):
        """初始化 ISO D4sigma 计算器
        
        参数:
            max_iterations: 最大迭代次数，默认 10
                - 如果迭代未在此次数内收敛，将返回警告信息
            convergence_threshold: 收敛阈值（相对变化），默认 0.01 (1%)
                - 当 D4sigma 相对变化小于此阈值时认为收敛
            roi_factor: ROI 边界因子，默认 3.0
                - ROI 半径 = roi_factor × D4sigma / 2
                - 3.0 对应 3 倍 D4sigma，包含 99.7% 以上的高斯光束能量
        
        异常:
            InvalidInputError: 参数无效时抛出
        
        Requirements: 2.6
        """
        # 验证参数
        if max_iterations < 1:
            raise InvalidInputError(
                f"最大迭代次数必须至少为 1，当前值: {max_iterations}"
            )
        
        if convergence_threshold <= 0:
            raise InvalidInputError(
                f"收敛阈值必须为正数，当前值: {convergence_threshold}"
            )
        
        if roi_factor <= 0:
            raise InvalidInputError(
                f"ROI 边界因子必须为正数，当前值: {roi_factor}"
            )
        
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.roi_factor = roi_factor

    def calculate(
        self,
        data: Union[np.ndarray, "proper.WaveFront"],
        sampling: Optional[float] = None,
    ) -> ISOD4sigmaResult:
        """使用 ISO 标准方法计算 D4sigma
        
        迭代过程：
        1. 估计并去除背景噪声
        2. 计算初始 D4sigma（使用全图）
        3. 应用 3×D4sigma ROI 掩模
        4. 重新计算 D4sigma
        5. 重复直到收敛或达到最大迭代次数
        
        参数:
            data: 复振幅数组或 PROPER 波前对象
                - 如果是 numpy 数组，必须是 2D 复数数组
                - 如果是 PROPER 波前对象，将自动提取振幅和采样间隔
            sampling: 采样间隔 (m)
                - 如果 data 是 numpy 数组，此参数必须提供
                - 如果 data 是 PROPER 波前对象，此参数可选（自动获取）
        
        返回:
            ISOD4sigmaResult 对象，包含：
            - dx: X 方向直径 (m)
            - dy: Y 方向直径 (m)
            - d_mean: 平均直径 (m)
            - centroid_x: 质心 X 坐标 (m)
            - centroid_y: 质心 Y 坐标 (m)
            - total_power: 总功率（归一化）
            - iterations: 迭代次数
            - converged: 是否收敛
            - roi_radius: 最终 ROI 半径 (m)
            - background_level: 背景噪声水平
            - warning: 警告信息（如迭代未收敛时）
        
        异常:
            InvalidInputError: 输入数据无效时抛出
        
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        # 提取复振幅数组和采样间隔
        amplitude, dx = self._extract_data(data, sampling)
        
        # 验证输入
        self._validate_input(amplitude, dx)
        
        # 计算强度分布
        intensity = np.abs(amplitude) ** 2
        
        # 步骤 1: 估计并去除背景噪声
        background_level = self._estimate_background(intensity)
        intensity_corrected = np.maximum(intensity - background_level, 0)
        
        # 检查去噪后是否还有有效信号
        total_power = np.sum(intensity_corrected)
        if total_power == 0:
            raise InvalidInputError(
                "去除背景噪声后，信号功率为零。"
                "可能是背景噪声估计过高或输入数据无有效信号。"
            )
        
        # 创建坐标网格
        ny, nx = intensity_corrected.shape
        x = (np.arange(nx) - nx / 2) * dx
        y = (np.arange(ny) - ny / 2) * dx
        X, Y = np.meshgrid(x, y)
        
        # 步骤 2: 计算初始 D4sigma（使用全图）
        centroid_x, centroid_y = self._calculate_centroid(
            intensity_corrected, X, Y
        )
        d4sigma_x, d4sigma_y = self._calculate_d4sigma(
            intensity_corrected, X, Y, centroid_x, centroid_y
        )
        d4sigma_mean = (d4sigma_x + d4sigma_y) / 2
        
        # 迭代变量
        converged = False
        warning = None
        iteration = 0
        
        # 步骤 3-5: 迭代 ROI 方法
        for iteration in range(1, self.max_iterations + 1):
            # 计算 ROI 半径（3 倍 D4sigma 的一半，即 1.5 倍 D4sigma）
            roi_radius = self.roi_factor * d4sigma_mean / 2
            
            # 应用 ROI 掩模
            intensity_roi = self._apply_roi(
                intensity_corrected,
                (centroid_x, centroid_y),
                roi_radius,
                dx,
                X, Y,
            )
            
            # 重新计算质心和 D4sigma
            new_centroid_x, new_centroid_y = self._calculate_centroid(
                intensity_roi, X, Y
            )
            new_d4sigma_x, new_d4sigma_y = self._calculate_d4sigma(
                intensity_roi, X, Y, new_centroid_x, new_centroid_y
            )
            new_d4sigma_mean = (new_d4sigma_x + new_d4sigma_y) / 2
            
            # 检查收敛
            if d4sigma_mean > 0:
                relative_change = abs(new_d4sigma_mean - d4sigma_mean) / d4sigma_mean
            else:
                relative_change = float('inf')
            
            # 更新值
            centroid_x, centroid_y = new_centroid_x, new_centroid_y
            d4sigma_x, d4sigma_y = new_d4sigma_x, new_d4sigma_y
            d4sigma_mean = new_d4sigma_mean
            
            # 判断是否收敛
            if relative_change < self.convergence_threshold:
                converged = True
                break
        
        # 如果未收敛，设置警告信息
        if not converged:
            warning = (
                f"ISO D4sigma 迭代未在 {self.max_iterations} 次内收敛。"
                f"最后一次相对变化: {relative_change:.4f}，"
                f"收敛阈值: {self.convergence_threshold:.4f}。"
                "返回当前最佳估计值。"
            )
        
        # 计算最终 ROI 半径
        final_roi_radius = self.roi_factor * d4sigma_mean / 2
        
        # 计算最终的总功率（在 ROI 内）
        final_intensity_roi = self._apply_roi(
            intensity_corrected,
            (centroid_x, centroid_y),
            final_roi_radius,
            dx,
            X, Y,
        )
        final_total_power = np.sum(final_intensity_roi)
        
        return ISOD4sigmaResult(
            dx=d4sigma_x,
            dy=d4sigma_y,
            d_mean=d4sigma_mean,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            total_power=final_total_power,
            iterations=iteration,
            converged=converged,
            roi_radius=final_roi_radius,
            background_level=background_level,
            warning=warning,
        )

    def _estimate_background(self, intensity: np.ndarray) -> float:
        """估计背景噪声水平
        
        使用图像边缘区域估计背景噪声。
        边缘区域定义为图像边缘 10% 的像素区域。
        
        ISO 11146 标准建议使用图像边缘区域来估计背景噪声，
        因为光束通常位于图像中心，边缘区域主要包含背景噪声。
        
        参数:
            intensity: 强度分布数组
        
        返回:
            背景噪声水平（强度值）
        
        Requirements: 2.1
        """
        ny, nx = intensity.shape
        
        # 计算边缘区域的宽度（10% 的图像尺寸，至少 1 像素）
        edge_width_x = max(1, int(nx * 0.1))
        edge_width_y = max(1, int(ny * 0.1))
        
        # 提取边缘区域的像素
        # 上边缘
        top_edge = intensity[:edge_width_y, :]
        # 下边缘
        bottom_edge = intensity[-edge_width_y:, :]
        # 左边缘（排除已包含在上下边缘的角落）
        left_edge = intensity[edge_width_y:-edge_width_y, :edge_width_x]
        # 右边缘（排除已包含在上下边缘的角落）
        right_edge = intensity[edge_width_y:-edge_width_y, -edge_width_x:]
        
        # 合并所有边缘像素
        edge_pixels = np.concatenate([
            top_edge.flatten(),
            bottom_edge.flatten(),
            left_edge.flatten(),
            right_edge.flatten(),
        ])
        
        # 使用边缘像素的平均值作为背景噪声估计
        # 也可以使用中位数，对异常值更鲁棒
        background_level = np.mean(edge_pixels)
        
        return background_level
    
    def _apply_roi(
        self,
        intensity: np.ndarray,
        centroid: Tuple[float, float],
        roi_radius: float,
        sampling: float,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> np.ndarray:
        """应用 ROI 掩模
        
        将 ROI 外的区域设为零。ROI 是以质心为中心的圆形区域。
        
        参数:
            intensity: 强度分布数组
            centroid: 质心坐标 (centroid_x, centroid_y) (m)
            roi_radius: ROI 半径 (m)
            sampling: 采样间隔 (m)
            X: X 坐标网格
            Y: Y 坐标网格
        
        返回:
            应用 ROI 掩模后的强度数组
        
        Requirements: 2.2, 2.3
        """
        centroid_x, centroid_y = centroid
        
        # 计算每个像素到质心的距离
        R = np.sqrt((X - centroid_x) ** 2 + (Y - centroid_y) ** 2)
        
        # 创建 ROI 掩模（圆形区域内为 1，外为 0）
        roi_mask = (R <= roi_radius).astype(float)
        
        # 应用掩模
        intensity_roi = intensity * roi_mask
        
        return intensity_roi
    
    def _calculate_centroid(
        self,
        intensity: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> Tuple[float, float]:
        """计算强度分布的质心
        
        质心公式：
        x̄ = ∫∫ I(x,y) × x dxdy / ∫∫ I(x,y) dxdy
        ȳ = ∫∫ I(x,y) × y dxdy / ∫∫ I(x,y) dxdy
        
        参数:
            intensity: 强度分布数组
            X: X 坐标网格
            Y: Y 坐标网格
        
        返回:
            (centroid_x, centroid_y) 质心坐标 (m)
        """
        total_power = np.sum(intensity)
        
        if total_power == 0:
            # 如果总功率为零，返回网格中心
            return 0.0, 0.0
        
        centroid_x = np.sum(intensity * X) / total_power
        centroid_y = np.sum(intensity * Y) / total_power
        
        return centroid_x, centroid_y
    
    def _calculate_d4sigma(
        self,
        intensity: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        centroid_x: float,
        centroid_y: float,
    ) -> Tuple[float, float]:
        """计算 D4sigma 光束直径
        
        D4sigma 公式：
        D4σ = 4 × √(∫∫ I(x,y) × (x-x̄)² dxdy / ∫∫ I(x,y) dxdy)
        
        参数:
            intensity: 强度分布数组
            X: X 坐标网格
            Y: Y 坐标网格
            centroid_x: 质心 X 坐标 (m)
            centroid_y: 质心 Y 坐标 (m)
        
        返回:
            (d4sigma_x, d4sigma_y) X 和 Y 方向的 D4sigma (m)
        """
        total_power = np.sum(intensity)
        
        if total_power == 0:
            return 0.0, 0.0
        
        # 计算二阶矩
        sigma_x_squared = np.sum(intensity * (X - centroid_x) ** 2) / total_power
        sigma_y_squared = np.sum(intensity * (Y - centroid_y) ** 2) / total_power
        
        # 计算 D4sigma
        sigma_x = np.sqrt(sigma_x_squared)
        sigma_y = np.sqrt(sigma_y_squared)
        
        d4sigma_x = 4 * sigma_x
        d4sigma_y = 4 * sigma_y
        
        return d4sigma_x, d4sigma_y

    def _extract_data(
        self,
        data: Union[np.ndarray, "proper.WaveFront"],
        sampling: Optional[float],
    ) -> Tuple[np.ndarray, float]:
        """从输入数据中提取复振幅数组和采样间隔
        
        参数:
            data: 复振幅数组或 PROPER 波前对象
            sampling: 采样间隔 (m)，可选
        
        返回:
            (amplitude, dx) 元组
            - amplitude: 复振幅数组
            - dx: 采样间隔 (m)
        
        异常:
            InvalidInputError: 无法提取数据时抛出
        """
        # 检查是否为 PROPER 波前对象
        # PROPER 波前对象具有 wfarr 属性
        if hasattr(data, 'wfarr'):
            # 这是 PROPER 波前对象
            try:
                import proper
                amplitude = proper.prop_get_amplitude(data)
                # 获取采样间隔
                if sampling is not None:
                    dx = sampling
                else:
                    dx = proper.prop_get_sampling(data)
                return amplitude, dx
            except ImportError:
                raise InvalidInputError(
                    "检测到 PROPER 波前对象，但无法导入 proper 库"
                )
            except Exception as e:
                raise InvalidInputError(
                    f"从 PROPER 波前对象提取数据失败: {e}"
                )
        
        # 假设是 numpy 数组
        if not isinstance(data, np.ndarray):
            raise InvalidInputError(
                f"不支持的输入类型: {type(data).__name__}。"
                "支持的类型: numpy.ndarray 或 PROPER WaveFront 对象"
            )
        
        # numpy 数组必须提供 sampling 参数
        if sampling is None:
            raise InvalidInputError(
                "使用 numpy 数组输入时，必须提供 sampling 参数"
            )
        
        return data, sampling
    
    def _validate_input(
        self,
        amplitude: np.ndarray,
        sampling: float,
    ) -> None:
        """验证输入数据的有效性
        
        参数:
            amplitude: 复振幅数组
            sampling: 采样间隔 (m)
        
        异常:
            InvalidInputError: 输入数据无效时抛出
        """
        # 检查数组是否为空
        if amplitude.size == 0:
            raise InvalidInputError("输入数组为空")
        
        # 检查数组维度
        if amplitude.ndim != 2:
            raise InvalidInputError(
                f"输入数组必须是 2D 数组，当前维度: {amplitude.ndim}"
            )
        
        # 检查采样间隔
        if sampling <= 0:
            raise InvalidInputError(
                f"采样间隔必须为正数，当前值: {sampling}"
            )
        
        # 检查是否包含 NaN 或 Inf
        if np.any(np.isnan(amplitude)):
            raise InvalidInputError("输入数组包含 NaN 值")
        
        if np.any(np.isinf(amplitude)):
            raise InvalidInputError("输入数组包含 Inf 值")
