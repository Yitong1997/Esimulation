# -*- coding: utf-8 -*-
"""
光束传播分析器

本模块实现 BeamPropagationAnalyzer 类，用于分析光束参数随传输距离的变化。

主要功能：
1. 在各传输位置测量光束直径
2. 计算远场发散角
3. 提供可视化绘图功能

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np

from .data_models import PropagationDataPoint, PropagationAnalysisResult
from .d4sigma_calculator import D4sigmaCalculator
from .iso_d4sigma_calculator import ISOD4sigmaCalculator
from .exceptions import InvalidInputError

if TYPE_CHECKING:
    import proper
    from .circular_aperture import CircularAperture


class BeamPropagationAnalyzer:
    """光束传播分析器
    
    分析光束参数随传输距离的变化，包括：
    - 在各传输位置测量光束直径
    - 计算远场发散角
    - 提供可视化绘图功能
    
    支持两种测量方法：
    - "ideal": 使用 PROPER 的 prop_get_beamradius（适用于理想高斯光束）
    - "iso": ISO 11146 标准方法（适用于经过光阑或有像差的光束）
    
    注意：
    - 对于理想高斯光束，PROPER 使用参考球面跟踪光束参数，wfarr 中存储的
      是相对于参考球面的偏差（振幅均匀）。因此不能直接使用 D4sigma 方法。
    - 当应用光阑后，光束不再是理想高斯光束，此时应使用 "iso" 方法。
    
    Example:
        >>> analyzer = BeamPropagationAnalyzer(
        ...     wavelength=633e-9,  # 633 nm
        ...     w0=1e-3,            # 1 mm 束腰
        ...     grid_size=256,
        ...     measurement_method="ideal",
        ... )
        >>> z_positions = [0, 0.1, 0.2, 0.5, 1.0, 2.0]  # 传输位置 (m)
        >>> result = analyzer.analyze(z_positions)
        >>> print(f"远场发散角: {result.divergence_mean * 1e3:.3f} mrad")
        >>> analyzer.plot(result, show_theory=True)
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
    """

    def __init__(
        self,
        wavelength: float,
        w0: float,
        grid_size: int = 256,
        measurement_method: str = "ideal",
    ):
        """初始化传播分析器
        
        参数:
            wavelength: 波长 (m)
            w0: 束腰半径 (m)
            grid_size: 网格大小，默认 256
            measurement_method: 测量方法，默认 "ideal"
                - "ideal": 使用 PROPER 的 prop_get_beamradius（理想高斯光束）
                - "iso": ISO 11146 标准方法（经过光阑或有像差的光束）
        
        异常:
            InvalidInputError: 参数无效时抛出
        
        Requirements: 5.2
        """
        # 验证参数
        if wavelength <= 0:
            raise InvalidInputError(
                f"波长必须为正数，收到: {wavelength}"
            )
        
        if w0 <= 0:
            raise InvalidInputError(
                f"束腰半径必须为正数，收到: {w0}"
            )
        
        if grid_size < 16:
            raise InvalidInputError(
                f"网格大小必须至少为 16，收到: {grid_size}"
            )
        
        if measurement_method not in ("ideal", "iso"):
            raise InvalidInputError(
                f"测量方法必须是 'ideal' 或 'iso'，收到: {measurement_method}"
            )
        
        self.wavelength = wavelength
        self.w0 = w0
        self.grid_size = grid_size
        self.measurement_method = measurement_method
        
        # 计算瑞利距离
        self.z_rayleigh = np.pi * w0**2 / wavelength
        
        # 初始化 ISO 测量计算器（仅在需要时使用）
        self._iso_calculator = None

    def analyze(
        self,
        z_positions: List[float],
        wfo: Optional["proper.WaveFront"] = None,
        aperture: Optional["CircularAperture"] = None,
    ) -> PropagationAnalysisResult:
        """分析光束参数随传输距离的变化
        
        在每个指定的传输位置测量光束直径，并计算远场发散角。
        
        工作流程：
        1. 如果没有提供 wfo，使用 PROPER 创建理想高斯光束
        2. 如果提供了 aperture，在传播前应用光阑（此时自动切换到 ISO 方法）
        3. 对每个 z 位置：
           a. 重新创建初始波前（确保每次传播独立）
           b. 如果有光阑，应用光阑
           c. 传播到目标位置
           d. 测量光束直径
        4. 计算远场发散角
        
        参数:
            z_positions: 传输位置列表 (m)
                - 必须是非空列表
                - 位置可以是任意顺序，但建议按升序排列
            wfo: 初始 PROPER 波前对象（可选）
                - 如果为 None，则创建理想高斯光束
                - 如果提供，将使用该波前作为初始状态
            aperture: 要应用的光阑（可选）
                - 如果提供，将在传播前应用光阑
                - 应用光阑后会自动使用 ISO 方法测量
        
        返回:
            PropagationAnalysisResult 对象，包含：
            - data_points: 各位置的测量数据点列表
            - divergence_x: X 方向远场发散角 (rad)
            - divergence_y: Y 方向远场发散角 (rad)
            - divergence_mean: 平均远场发散角 (rad)
            - wavelength: 波长 (m)
            - w0: 初始束腰 (m)
        
        异常:
            InvalidInputError: 输入参数无效时抛出
        
        Requirements: 5.1, 5.2, 5.3, 5.4
        """
        import proper
        
        # 验证 z_positions
        if not z_positions:
            raise InvalidInputError("传输位置列表不能为空")
        
        if not all(isinstance(z, (int, float)) for z in z_positions):
            raise InvalidInputError("传输位置必须是数值类型")
        
        # 确定是否使用自定义波前
        use_custom_wfo = wfo is not None
        
        # 确定实际使用的测量方法
        # 如果应用了光阑，强制使用 ISO 方法
        actual_method = self.measurement_method
        if aperture is not None and actual_method == "ideal":
            actual_method = "iso"
        
        # 测量各位置的光束直径
        data_points: List[PropagationDataPoint] = []
        
        for z in z_positions:
            # 为每个位置创建新的波前（确保独立性）
            if use_custom_wfo:
                # 使用自定义波前时，需要复制
                current_wfo = self._copy_wavefront(wfo)
            else:
                # 创建新的理想高斯光束
                current_wfo = self._create_initial_wavefront()
            
            # 如果提供了光阑，应用光阑
            if aperture is not None:
                aperture.apply(current_wfo)
            
            # 传播到目标位置
            if z != 0:
                proper.prop_propagate(current_wfo, z)
            
            # 测量光束直径
            dx, dy = self._measure_beam_diameter(current_wfo, actual_method)
            d_mean = (dx + dy) / 2
            
            # 创建数据点
            data_point = PropagationDataPoint(
                z=z,
                dx=dx,
                dy=dy,
                d_mean=d_mean,
                method=actual_method,
            )
            data_points.append(data_point)
        
        # 计算远场发散角
        divergence_x, divergence_y = self._calculate_far_field_divergence(
            data_points
        )
        divergence_mean = (divergence_x + divergence_y) / 2
        
        return PropagationAnalysisResult(
            data_points=data_points,
            divergence_x=divergence_x,
            divergence_y=divergence_y,
            divergence_mean=divergence_mean,
            wavelength=self.wavelength,
            w0=self.w0,
        )

    def _measure_beam_diameter(
        self,
        wfo: "proper.WaveFront",
        method: str,
    ) -> Tuple[float, float]:
        """测量光束直径
        
        参数:
            wfo: PROPER 波前对象
            method: 测量方法 ("ideal" 或 "iso")
        
        返回:
            (dx, dy) X 和 Y 方向的光束直径 (m)
        """
        import proper
        
        if method == "ideal":
            # 使用 PROPER 的 prop_get_beamradius
            # 这个函数返回 PROPER 内部跟踪的光束半径
            beam_radius = proper.prop_get_beamradius(wfo)
            # D4sigma = 2 × w（光束直径 = 2 × 光束半径）
            d4sigma = 2 * beam_radius
            return d4sigma, d4sigma
        else:
            # 使用 ISO D4sigma 方法
            if self._iso_calculator is None:
                self._iso_calculator = ISOD4sigmaCalculator()
            result = self._iso_calculator.calculate(wfo)
            return result.dx, result.dy
    
    def _copy_wavefront(self, wfo: "proper.WaveFront") -> "proper.WaveFront":
        """复制 PROPER 波前对象
        
        参数:
            wfo: 原始 PROPER 波前对象
        
        返回:
            复制的 PROPER 波前对象
        """
        import proper
        import copy
        
        # 深拷贝波前对象
        new_wfo = copy.deepcopy(wfo)
        return new_wfo
    
    def _create_initial_wavefront(self) -> "proper.WaveFront":
        """创建理想高斯光束的初始波前
        
        使用 PROPER 库创建一个理想高斯光束。
        
        根据 BTS 规范：
        - beam_diameter = 2 × w0
        - beam_diam_fraction = 0.5
        - 网格物理尺寸 = 4 × w0
        
        返回:
            PROPER 波前对象
        """
        import proper
        
        # 根据 BTS 规范设置参数
        beam_diameter = 2 * self.w0  # beam_diameter = 2 × 束腰半径
        beam_diam_fraction = 0.5     # 固定为 0.5
        
        # 创建波前对象
        wfo = proper.prop_begin(
            beam_diameter,
            self.wavelength,
            self.grid_size,
            beam_diam_fraction,
        )
        
        # 定义高斯光束
        # prop_define_entrance 会设置高斯光束的初始振幅分布
        proper.prop_define_entrance(wfo)
        
        return wfo

    def _calculate_far_field_divergence(
        self,
        data_points: List[PropagationDataPoint],
    ) -> Tuple[float, float]:
        """计算远场发散角
        
        使用远场数据点的线性拟合计算发散角。
        远场定义为 z >> z_R（瑞利距离）的区域。
        
        在远场，光束半径近似线性增长：
        w(z) ≈ w0 × z / z_R = θ × z
        
        其中 θ = w0 / z_R = λ / (π × w0) 是远场发散角（半角）。
        
        D4sigma = 2 × w，所以：
        D4sigma(z) ≈ 2 × θ × z
        
        通过线性拟合 D4sigma vs z，斜率的一半即为发散角。
        
        参数:
            data_points: 测量数据点列表
        
        返回:
            (divergence_x, divergence_y) 远场发散角 (rad)
            - 如果没有足够的远场数据点，返回理论发散角
        
        Requirements: 5.4
        """
        # 筛选远场数据点（z > 2 × z_R）
        far_field_threshold = 2 * self.z_rayleigh
        far_field_points = [
            dp for dp in data_points if dp.z > far_field_threshold
        ]
        
        # 如果远场数据点不足，使用理论发散角
        if len(far_field_points) < 2:
            # 理论发散角（半角）：θ = λ / (π × w0)
            theoretical_divergence = self.wavelength / (np.pi * self.w0)
            return theoretical_divergence, theoretical_divergence
        
        # 提取远场数据
        z_far = np.array([dp.z for dp in far_field_points])
        dx_far = np.array([dp.dx for dp in far_field_points])
        dy_far = np.array([dp.dy for dp in far_field_points])
        
        # 线性拟合：D4sigma = 2 × θ × z + b
        # 使用 numpy 的 polyfit 进行线性拟合
        # 斜率 = 2 × θ，所以 θ = 斜率 / 2
        
        # X 方向
        coeffs_x = np.polyfit(z_far, dx_far, 1)
        slope_x = coeffs_x[0]
        divergence_x = slope_x / 2  # D4sigma = 2w，所以斜率是 2θ
        
        # Y 方向
        coeffs_y = np.polyfit(z_far, dy_far, 1)
        slope_y = coeffs_y[0]
        divergence_y = slope_y / 2
        
        # 确保发散角为正值
        divergence_x = abs(divergence_x)
        divergence_y = abs(divergence_y)
        
        return divergence_x, divergence_y

    def plot(
        self,
        result: PropagationAnalysisResult,
        show_theory: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """绘制光束直径变化曲线
        
        绘制测量的光束直径随传输距离的变化曲线。
        可选择显示理论曲线进行对比。
        
        参数:
            result: 分析结果（PropagationAnalysisResult 对象）
            show_theory: 是否显示理论曲线，默认 True
                - 理论曲线：w(z) = w₀ × √(1 + (z/z_R)²)
                - D4sigma = 2 × w(z)
            save_path: 保存路径（可选）
                - 如果提供，将图像保存到指定路径
                - 支持的格式：png, pdf, svg 等
        
        Requirements: 5.5
        """
        import matplotlib.pyplot as plt
        
        # 提取测量数据
        z_measured = np.array([dp.z for dp in result.data_points])
        dx_measured = np.array([dp.dx for dp in result.data_points])
        dy_measured = np.array([dp.dy for dp in result.data_points])
        d_mean_measured = np.array([dp.d_mean for dp in result.data_points])
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制测量数据
        ax.scatter(
            z_measured * 1e3,  # 转换为 mm
            dx_measured * 1e3,  # 转换为 mm
            marker='o',
            color='blue',
            label='测量值 (X 方向)',
            alpha=0.7,
        )
        ax.scatter(
            z_measured * 1e3,
            dy_measured * 1e3,
            marker='s',
            color='red',
            label='测量值 (Y 方向)',
            alpha=0.7,
        )
        ax.scatter(
            z_measured * 1e3,
            d_mean_measured * 1e3,
            marker='^',
            color='green',
            label='测量值 (平均)',
            alpha=0.7,
        )
        
        # 绘制理论曲线
        if show_theory:
            # 生成理论曲线的 z 值
            z_min = min(z_measured) if len(z_measured) > 0 else 0
            z_max = max(z_measured) if len(z_measured) > 0 else 1
            z_theory = np.linspace(z_min, z_max, 200)
            
            # 计算理论光束直径
            # w(z) = w₀ × √(1 + (z/z_R)²)
            # D4sigma = 2 × w(z)
            w_theory = result.w0 * np.sqrt(1 + (z_theory / self.z_rayleigh)**2)
            d4sigma_theory = 2 * w_theory
            
            ax.plot(
                z_theory * 1e3,
                d4sigma_theory * 1e3,
                'k--',
                linewidth=2,
                label='理论曲线',
            )
        
        # 标注瑞利距离
        ax.axvline(
            x=self.z_rayleigh * 1e3,
            color='gray',
            linestyle=':',
            alpha=0.5,
            label=f'瑞利距离 z_R = {self.z_rayleigh * 1e3:.2f} mm',
        )
        
        # 设置图形属性
        ax.set_xlabel('传输距离 z (mm)', fontsize=12)
        ax.set_ylabel('光束直径 D4σ (mm)', fontsize=12)
        ax.set_title(
            f'光束直径随传输距离变化\n'
            f'λ = {result.wavelength * 1e9:.1f} nm, '
            f'w₀ = {result.w0 * 1e3:.3f} mm, '
            f'z_R = {self.z_rayleigh * 1e3:.2f} mm',
            fontsize=14,
        )
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加发散角信息
        info_text = (
            f'远场发散角:\n'
            f'  θ_x = {result.divergence_x * 1e3:.3f} mrad\n'
            f'  θ_y = {result.divergence_y * 1e3:.3f} mrad\n'
            f'  θ_mean = {result.divergence_mean * 1e3:.3f} mrad\n'
            f'理论值: {self.wavelength / (np.pi * self.w0) * 1e3:.3f} mrad'
        )
        ax.text(
            0.02, 0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )
        
        plt.tight_layout()
        
        # 保存或显示图形
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)

    def theoretical_beam_diameter(self, z: float) -> float:
        """计算理论光束直径
        
        使用高斯光束传播公式计算理论光束直径。
        
        公式：
        w(z) = w₀ × √(1 + (z/z_R)²)
        D4sigma = 2 × w(z)
        
        参数:
            z: 传输距离 (m)
        
        返回:
            理论光束直径 D4sigma (m)
        
        Requirements: 8.1
        """
        w_z = self.w0 * np.sqrt(1 + (z / self.z_rayleigh)**2)
        return 2 * w_z
    
    def theoretical_divergence(self) -> float:
        """计算理论远场发散角
        
        理论发散角（半角）公式：
        θ = λ / (π × w₀)
        
        返回:
            理论远场发散角 (rad)
        """
        return self.wavelength / (np.pi * self.w0)
    
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"BeamPropagationAnalyzer("
            f"wavelength={self.wavelength:.3e} m, "
            f"w0={self.w0:.3e} m, "
            f"grid_size={self.grid_size}, "
            f"method={self.measurement_method})"
        )
