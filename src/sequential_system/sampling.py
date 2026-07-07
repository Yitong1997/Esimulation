"""
采样面和仿真结果定义模块

本模块定义采样面（SamplingPlane）和仿真结果（SamplingResult, SimulationResults）类。

采样面用于在光路中指定位置记录波前数据。
仿真结果包含波前复振幅、光束半径、波前质量指标等信息。

验证需求:
- Requirements 4.1, 4.2: 采样面定义
- Requirements 4.3, 4.4, 4.5, 4.6: 采样面数据
- Requirements 5.6, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7: 仿真结果

作者：混合光学仿真项目
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union, Iterator, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .exceptions import SamplingError

if TYPE_CHECKING:
    from .source import GaussianBeamSource
    from .coordinate_tracking import OpticalAxisState
    from gaussian_beam_simulation.optical_elements import OpticalElement


@dataclass
class SamplingPlane:
    """采样面定义
    
    用于在光路中指定位置记录波前数据。
    
    参数:
        distance: 从光源沿光路的累积距离（mm）
            - 必须为非负值
        name: 采样面名称（可选）
            - 用于在结果中标识采样面
    
    属性:
        result: 仿真后填充的结果（SamplingResult 对象）
    
    示例:
        >>> # 在光程距离 150mm 处添加采样面
        >>> plane = SamplingPlane(distance=150.0, name="focus")
        >>> 
        >>> # 仿真后访问结果
        >>> if plane.result is not None:
        ...     print(f"光束半径: {plane.result.beam_radius:.3f} mm")
    
    验证需求:
        - Requirements 4.1: 接受位置参数（光程距离）
        - Requirements 4.2: 接受可选名称参数
    """
    
    distance: float  # mm，从光源沿光路的累积距离
    name: Optional[str] = None
    
    # 仿真后填充的结果
    result: Optional['SamplingResult'] = field(default=None, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """初始化后验证参数"""
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """验证参数"""
        # 验证 distance
        if not isinstance(self.distance, (int, float)):
            raise SamplingError(
                f"采样面距离 (distance) 必须为数值类型，"
                f"实际类型为 {type(self.distance).__name__}。"
            )
        
        if not np.isfinite(self.distance):
            raise SamplingError(
                f"采样面距离 (distance) 必须为有限值，"
                f"实际为 {self.distance} mm（无穷大或 NaN 不允许）。"
            )
        
        if self.distance < 0:
            raise SamplingError(
                f"采样面距离 (distance) 必须为非负值，"
                f"实际为 {self.distance} mm。"
            )
    
    def __repr__(self) -> str:
        name_str = f"'{self.name}'" if self.name else "unnamed"
        has_result = "有结果" if self.result is not None else "无结果"
        return f"SamplingPlane({name_str}, distance={self.distance}mm, {has_result})"


@dataclass
class SamplingResult:
    """单个采样面的仿真结果
    
    包含波前复振幅、光束参数和波前质量指标。
    
    属性:
        distance: 光程距离（mm）
        z_position: z 坐标位置（mm）
        wavefront: 复振幅分布（2D 复数数组）
        sampling: 采样间隔（mm/pixel）
        beam_radius: 光束半径（mm）
        name: 采样面名称（可选）
        axis_state: 采样面处的光轴状态（可选）
    
    计算属性:
        amplitude: 振幅分布
        phase: 相位分布（rad）
        wavefront_rms: 波前 RMS（waves）
        wavefront_pv: 波前 PV（waves）
        grid_size: 网格大小
        physical_size: 物理尺寸（mm）
    
    方法:
        compute_m2(): 从振幅分布计算 M² 因子
    
    验证需求:
        - Requirements 4.4: 存储复振幅分布
        - Requirements 4.5: 存储采样间隔
        - Requirements 4.6: 计算并存储光束半径
        - Requirements 8.1-8.7: 仿真结果属性
    """
    
    distance: float  # mm，光程距离
    z_position: float  # mm，z 坐标位置
    wavefront: NDArray  # 复振幅分布
    sampling: float  # mm/pixel，采样间隔
    beam_radius: float  # mm，光束半径
    name: Optional[str] = None
    _wavelength: float = 0.633  # μm，用于波前质量计算
    axis_state: Optional['OpticalAxisState'] = None  # 采样面处的光轴状态
    _reference_radius: float = float('inf')  # mm，PROPER 内部参考球面曲率半径（用于调试）
    
    @property
    def amplitude(self) -> NDArray:
        """振幅分布
        
        Returns:
            振幅数组（实数）
        """
        return np.abs(self.wavefront)
    
    @property
    def phase(self) -> NDArray:
        """相位分布（rad）
        
        Returns:
            相位数组（弧度）
        """
        return np.angle(self.wavefront)
    
    @property
    def grid_size(self) -> int:
        """网格大小"""
        return self.wavefront.shape[0]
    
    @property
    def physical_size(self) -> float:
        """物理尺寸（mm）"""
        return self.sampling * self.grid_size
    
    @property
    def wavefront_rms(self) -> float:
        """波前 RMS（waves）
        
        计算光瞳内波前的均方根值。
        使用振幅作为权重，只计算有效区域。
        
        Returns:
            波前 RMS（波长数）
        """
        amp = self.amplitude
        phase = self.phase
        
        # 使用振幅阈值确定有效区域
        threshold = 0.01 * np.max(amp)
        mask = amp > threshold
        
        if not np.any(mask):
            return 0.0
        
        # 提取有效区域的相位
        valid_phase = phase[mask]
        
        # 去除 piston（平均值）
        mean_phase = np.mean(valid_phase)
        phase_centered = valid_phase - mean_phase
        
        # 计算 RMS（弧度）
        rms_rad = np.sqrt(np.mean(phase_centered**2))
        
        # 转换为波长数
        rms_waves = rms_rad / (2 * np.pi)
        
        return float(rms_waves)
    
    @property
    def wavefront_pv(self) -> float:
        """波前 PV（waves）
        
        计算光瞳内波前的峰谷值。
        
        Returns:
            波前 PV（波长数）
        """
        amp = self.amplitude
        phase = self.phase
        
        # 使用振幅阈值确定有效区域
        threshold = 0.01 * np.max(amp)
        mask = amp > threshold
        
        if not np.any(mask):
            return 0.0
        
        # 提取有效区域的相位
        valid_phase = phase[mask]
        
        # 计算 PV（弧度）
        pv_rad = np.max(valid_phase) - np.min(valid_phase)
        
        # 转换为波长数
        pv_waves = pv_rad / (2 * np.pi)
        
        return float(pv_waves)
    
    @property
    def wavefront_curvature_radius(self) -> float:
        """波前曲率半径（mm）
        
        从相位分布拟合球面波前，计算曲率半径。
        使用最小二乘法拟合相位中的二次项。
        
        相位模型：φ(r) = φ0 + a * r²
        其中 a = -k / (2 * R)，所以 R = -k / (2 * a)
        
        Returns:
            波前曲率半径（mm）
            - 正值：发散波前（曲率中心在光束后方）
            - 负值：会聚波前（曲率中心在光束前方）
            - np.inf：平面波前
        
        注意:
            此方法从真实的绝对相位分布拟合曲率半径，
            不依赖于 PROPER 内部的参考面定义。
        """
        amp = self.amplitude
        phase = self.phase
        
        # 使用振幅阈值确定有效区域
        threshold = 0.01 * np.max(amp)
        mask = amp > threshold
        
        if not np.any(mask):
            return np.inf
        
        # 创建坐标网格（mm）
        n = self.grid_size
        half_size = self.physical_size / 2.0
        coords = np.linspace(-half_size, half_size, n)
        X, Y = np.meshgrid(coords, coords)
        R_sq = X**2 + Y**2  # r²，单位 mm²
        
        # 提取有效区域的数据
        r_sq_valid = R_sq[mask]
        phase_valid = phase[mask]
        
        # 去除 piston（常数项）
        phase_mean = np.mean(phase_valid)
        phase_centered = phase_valid - phase_mean
        
        # 最小二乘拟合：phase = a * r²
        # 使用加权拟合，权重为振幅的平方（强度）
        weights = amp[mask]**2
        
        # 加权最小二乘：a = Σ(w * r² * phase) / Σ(w * r⁴)
        numerator = np.sum(weights * r_sq_valid * phase_centered)
        denominator = np.sum(weights * r_sq_valid**2)
        
        if abs(denominator) < 1e-20:
            return np.inf
        
        a = numerator / denominator
        
        # 从拟合系数计算曲率半径
        # φ = -k * r² / (2 * R)，所以 a = -k / (2 * R)
        # R = -k / (2 * a)
        if abs(a) < 1e-15:
            return np.inf
        
        # 波数 k = 2π / λ，波长单位 mm
        wavelength_mm = self._wavelength * 1e-3
        k = 2 * np.pi / wavelength_mm  # 单位 1/mm
        
        R = -k / (2 * a)  # 单位 mm
        
        return float(R)
    
    def compute_m2(self) -> float:
        """从振幅分布计算 M² 因子
        
        使用二阶矩方法计算光束质量因子。
        
        Returns:
            M² 因子（>= 1.0）
        
        注意:
            这是一个简化的计算方法，假设光束接近高斯分布。
            对于非高斯光束，结果可能不准确。
        """
        amp = self.amplitude
        intensity = amp**2
        
        # 创建坐标网格
        n = self.grid_size
        half_size = self.physical_size / 2.0
        coords = np.linspace(-half_size, half_size, n)
        X, Y = np.meshgrid(coords, coords)
        
        # 计算总强度
        total_intensity = np.sum(intensity)
        if total_intensity < 1e-15:
            return 1.0
        
        # 计算质心
        x_centroid = np.sum(X * intensity) / total_intensity
        y_centroid = np.sum(Y * intensity) / total_intensity
        
        # 计算二阶矩
        x_var = np.sum((X - x_centroid)**2 * intensity) / total_intensity
        y_var = np.sum((Y - y_centroid)**2 * intensity) / total_intensity
        
        # 计算光束半径（二阶矩定义）
        w_measured = np.sqrt(2 * (x_var + y_var))
        
        # 估算 M²（需要知道理想高斯光束的参数）
        # 这里返回一个基于测量光束半径与期望光束半径比值的估计
        # 简化处理：假设 M² ≈ 1.0（理想高斯光束）
        # 实际应用中需要更复杂的计算
        
        # 使用光束半径比值估算
        if self.beam_radius > 0:
            ratio = w_measured / self.beam_radius
            m2 = max(1.0, ratio**2)
        else:
            m2 = 1.0
        
        return float(m2)
    
    def __repr__(self) -> str:
        name_str = f"'{self.name}'" if self.name else "unnamed"
        return (
            f"SamplingResult({name_str}, distance={self.distance:.2f}mm, "
            f"w={self.beam_radius:.3f}mm, grid={self.grid_size})"
        )


@dataclass
class SimulationResults:
    """仿真结果集合
    
    包含所有采样面的仿真结果，支持通过名称或索引访问。
    
    属性:
        sampling_results: 采样结果字典，键为采样面名称或索引字符串
        source: 光源参数
        surfaces: 光学面列表
    
    方法:
        __getitem__: 通过名称或索引获取采样结果
        __iter__: 迭代所有采样结果
        __len__: 返回采样结果数量
    
    示例:
        >>> results = system.run()
        >>> 
        >>> # 通过名称访问
        >>> focus_result = results["focus"]
        >>> 
        >>> # 通过索引访问
        >>> first_result = results[0]
        >>> 
        >>> # 迭代所有结果
        >>> for result in results:
        ...     print(f"{result.name}: w={result.beam_radius:.3f}mm")
    
    验证需求:
        - Requirements 5.6: 返回所有采样面结果
        - Requirements 8.1: 包含复振幅分布
    """
    
    sampling_results: Dict[str, SamplingResult]
    source: 'GaussianBeamSource'
    surfaces: List['OpticalElement']
    
    def __getitem__(self, key: Union[str, int]) -> SamplingResult:
        """通过名称或索引获取采样结果
        
        参数:
            key: 采样面名称（str）或索引（int）
        
        Returns:
            SamplingResult 对象
        
        Raises:
            KeyError: 当名称不存在时
            IndexError: 当索引超出范围时
        """
        if isinstance(key, int):
            values = list(self.sampling_results.values())
            if key < 0 or key >= len(values):
                raise IndexError(
                    f"索引 {key} 超出范围，有效范围为 0 到 {len(values) - 1}。"
                )
            return values[key]
        
        if key not in self.sampling_results:
            available = list(self.sampling_results.keys())
            raise KeyError(
                f"采样面 '{key}' 不存在。可用的采样面: {available}"
            )
        return self.sampling_results[key]
    
    def __iter__(self) -> Iterator[SamplingResult]:
        """迭代所有采样结果"""
        return iter(self.sampling_results.values())
    
    def __len__(self) -> int:
        """返回采样结果数量"""
        return len(self.sampling_results)
    
    def __contains__(self, key: str) -> bool:
        """检查采样面名称是否存在
        
        参数:
            key: 采样面名称
        
        Returns:
            如果名称存在返回 True，否则返回 False
        """
        return key in self.sampling_results
    
    def __repr__(self) -> str:
        n_results = len(self.sampling_results)
        n_surfaces = len(self.surfaces)
        return (
            f"SimulationResults({n_results} 个采样面, {n_surfaces} 个光学面)"
        )
