"""
高斯光束光源定义模块

本模块定义 GaussianBeamSource 数据类，用于封装高斯光束光源的参数。
该类作为序列光学系统的输入光源定义，并提供转换为现有 GaussianBeam 对象的方法。

设计说明：
- 使用 @dataclass 装饰器简化类定义
- 复用 GaussianBeam 类的参数验证逻辑
- 在 __post_init__ 中进行参数验证，验证失败时抛出 SourceConfigurationError

参数单位约定：
- 波长 (wavelength): μm
- 束腰半径 (w0): mm
- 束腰位置 (z0): mm
- M² 因子 (m2): 无量纲

作者：混合光学仿真项目
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

from .exceptions import SourceConfigurationError

# 延迟导入以避免循环依赖
if TYPE_CHECKING:
    from gaussian_beam_simulation.gaussian_beam import GaussianBeam


@dataclass
class GaussianBeamSource:
    """高斯光束光源定义
    
    封装高斯光束光源的参数，提供简洁的接口用于定义序列光学系统的输入光源。
    
    参数:
        wavelength: 波长（μm）
            - 必须为正有限值
            - 典型值：0.38 ~ 0.78 μm（可见光），0.633 μm（He-Ne 激光）
        w0: 束腰半径（mm）
            - 必须为正有限值
            - 定义高斯光束在束腰处的 1/e² 强度半径
        z0: 束腰位置（mm），相对于光源位置，默认 0.0
            - 负值表示束腰在光源之前（发散光束）
            - 正值表示束腰在光源之后（会聚光束）
            - 零表示束腰在光源位置（平面波前）
        m2: M² 因子，默认 1.0
            - 必须 >= 1.0
            - M² = 1.0 表示理想高斯光束
            - M² > 1.0 表示实际光束（光束质量较差）
    
    属性:
        zR: 瑞利距离（mm），计算公式：zR = π * w0² / (M² * λ)
        divergence: 远场发散角（rad），计算公式：θ = M² * λ / (π * w0)
    
    方法:
        to_gaussian_beam(): 转换为 GaussianBeam 对象
    
    示例:
        >>> # 创建发散高斯光束（束腰在光源前 50mm）
        >>> source = GaussianBeamSource(
        ...     wavelength=0.633,  # μm，He-Ne 激光
        ...     w0=1.0,            # mm，束腰半径
        ...     z0=-50.0,          # mm，束腰在光源前 50mm
        ... )
        >>> print(f"瑞利距离: {source.zR:.2f} mm")
        
        >>> # 创建平面波前高斯光束（束腰在光源位置）
        >>> source = GaussianBeamSource(
        ...     wavelength=0.532,  # μm，绿光激光
        ...     w0=2.0,            # mm
        ...     z0=0.0,            # mm，束腰在光源位置
        ... )
        
        >>> # 创建带 M² 因子的高斯光束
        >>> source = GaussianBeamSource(
        ...     wavelength=0.633,
        ...     w0=1.0,
        ...     z0=-50.0,
        ...     m2=1.3,  # M² = 1.3，实际光束
        ... )
        
        >>> # 转换为 GaussianBeam 对象用于仿真
        >>> beam = source.to_gaussian_beam()
    
    Raises:
        SourceConfigurationError: 当参数无效时抛出
            - wavelength 不是正有限值
            - w0 不是正有限值
            - m2 < 1.0
    
    验证需求:
        - Requirements 1.1: 接受波长参数（μm）
        - Requirements 1.2: 验证波长为正有限值
        - Requirements 1.3: 接受束腰半径 w0（mm）
        - Requirements 1.4: 验证 w0 为正有限值
        - Requirements 1.5: 接受束腰位置 z0（mm）
        - Requirements 1.6: 接受 M² 因子，默认 1.0
        - Requirements 1.7: 验证 M² >= 1.0
    """
    
    wavelength: float  # μm
    w0: float          # mm
    z0: float = 0.0    # mm
    m2: float = 1.0
    
    def __post_init__(self) -> None:
        """初始化后验证参数
        
        验证规则：
        - wavelength: 必须为正有限值（Requirements 1.1, 1.2）
        - w0: 必须为正有限值（Requirements 1.3, 1.4）
        - z0: 必须为有限实数（Requirements 1.5）
        - m2: 必须 >= 1.0（Requirements 1.6, 1.7）
        
        Raises:
            SourceConfigurationError: 当参数无效时抛出
        """
        # 验证波长（必须为正有限值）
        self._validate_wavelength()
        
        # 验证束腰半径（必须为正有限值）
        self._validate_w0()
        
        # 验证束腰位置（必须为有限实数）
        self._validate_z0()
        
        # 验证 M² 因子（必须 >= 1.0）
        self._validate_m2()
    
    def _validate_wavelength(self) -> None:
        """验证波长参数
        
        Raises:
            SourceConfigurationError: 当波长无效时抛出
        """
        # 检查类型
        if not isinstance(self.wavelength, (int, float)):
            raise SourceConfigurationError(
                f"光源参数 'wavelength' 类型无效：期望数值类型，"
                f"实际类型为 {type(self.wavelength).__name__}。"
            )
        
        # 检查是否为有限值
        if not np.isfinite(self.wavelength):
            raise SourceConfigurationError(
                f"光源参数 'wavelength' 无效：期望有限值，"
                f"实际为 {self.wavelength} μm（无穷大或 NaN 不允许）。"
                f"请确保波长为正有限值。"
            )
        
        # 检查是否为正值
        if self.wavelength <= 0:
            raise SourceConfigurationError(
                f"光源参数 'wavelength' 无效：期望正值，"
                f"实际为 {self.wavelength} μm。"
                f"请确保波长为正有限值。"
            )
    
    def _validate_w0(self) -> None:
        """验证束腰半径参数
        
        Raises:
            SourceConfigurationError: 当束腰半径无效时抛出
        """
        # 检查类型
        if not isinstance(self.w0, (int, float)):
            raise SourceConfigurationError(
                f"光源参数 'w0' 类型无效：期望数值类型，"
                f"实际类型为 {type(self.w0).__name__}。"
            )
        
        # 检查是否为有限值
        if not np.isfinite(self.w0):
            raise SourceConfigurationError(
                f"光源参数 'w0'（束腰半径）无效：期望有限值，"
                f"实际为 {self.w0} mm（无穷大或 NaN 不允许）。"
                f"请确保束腰半径为正有限值。"
            )
        
        # 检查是否为正值
        if self.w0 <= 0:
            raise SourceConfigurationError(
                f"光源参数 'w0'（束腰半径）无效：期望正值，"
                f"实际为 {self.w0} mm。"
                f"请确保束腰半径为正有限值。"
            )
    
    def _validate_z0(self) -> None:
        """验证束腰位置参数
        
        Raises:
            SourceConfigurationError: 当束腰位置无效时抛出
        """
        # 检查类型
        if not isinstance(self.z0, (int, float)):
            raise SourceConfigurationError(
                f"光源参数 'z0' 类型无效：期望数值类型，"
                f"实际类型为 {type(self.z0).__name__}。"
            )
        
        # 检查是否为有限值
        if not np.isfinite(self.z0):
            raise SourceConfigurationError(
                f"光源参数 'z0'（束腰位置）无效：期望有限值，"
                f"实际为 {self.z0} mm（无穷大或 NaN 不允许）。"
                f"请确保束腰位置为有限值。"
            )
    
    def _validate_m2(self) -> None:
        """验证 M² 因子参数
        
        Raises:
            SourceConfigurationError: 当 M² 因子无效时抛出
        """
        # 检查类型
        if not isinstance(self.m2, (int, float)):
            raise SourceConfigurationError(
                f"光源参数 'm2' 类型无效：期望数值类型，"
                f"实际类型为 {type(self.m2).__name__}。"
            )
        
        # 检查是否为有限值
        if not np.isfinite(self.m2):
            raise SourceConfigurationError(
                f"光源参数 'm2'（M² 因子）无效：期望有限值，"
                f"实际为 {self.m2}（无穷大或 NaN 不允许）。"
                f"请确保 M² 因子为有限值且 >= 1.0。"
            )
        
        # 检查是否 >= 1.0
        if self.m2 < 1.0:
            raise SourceConfigurationError(
                f"光源参数 'm2'（M² 因子）无效：期望 >= 1.0，"
                f"实际为 {self.m2}。"
                f"物理上 M² 因子不能小于 1.0（理想高斯光束 M² = 1.0）。"
            )
    
    @property
    def wavelength_mm(self) -> float:
        """波长（mm）
        
        将波长从 μm 转换为 mm，用于内部计算。
        """
        return self.wavelength * 1e-3
    
    @property
    def zR(self) -> float:
        """瑞利距离（mm）
        
        计算公式：zR = π * w0² / (M² * λ)
        
        瑞利距离是高斯光束的重要参数，定义为光束半径增大到束腰半径 √2 倍的距离。
        在瑞利距离内，光束可近似为平行光；超过瑞利距离后，光束开始明显发散。
        
        Returns:
            瑞利距离（mm）
        """
        return np.pi * self.w0**2 / (self.m2 * self.wavelength_mm)
    
    @property
    def divergence(self) -> float:
        """远场发散角（rad）
        
        计算公式：θ = M² * λ / (π * w0)
        
        远场发散角描述高斯光束在远场（z >> zR）的发散特性。
        
        Returns:
            远场发散角（rad）
        """
        return self.m2 * self.wavelength_mm / (np.pi * self.w0)
    
    def w(self, z: float) -> float:
        """计算位置 z 处的光束半径（mm）
        
        计算公式：w(z) = w0 * sqrt(1 + ((z - z0) / zR)²)
        
        参数:
            z: 位置（mm），相对于光源位置
        
        Returns:
            光束半径（mm）
        """
        dz = z - self.z0
        return self.w0 * np.sqrt(1 + (dz / self.zR)**2)
    
    def R(self, z: float) -> float:
        """计算位置 z 处的波前曲率半径（mm）
        
        计算公式：R(z) = (z - z0) * (1 + (zR / (z - z0))²)
        
        在束腰处 R = ∞（平面波前）
        
        参数:
            z: 位置（mm），相对于光源位置
        
        Returns:
            波前曲率半径（mm），束腰处返回 np.inf
        """
        dz = z - self.z0
        if abs(dz) < 1e-10:
            return np.inf
        return dz * (1 + (self.zR / dz)**2)
    
    def to_gaussian_beam(self) -> "GaussianBeam":
        """转换为 GaussianBeam 对象
        
        创建一个 GaussianBeam 对象，用于后续的波前生成和仿真。
        复用现有 GaussianBeam 类的功能。
        
        Returns:
            GaussianBeam 对象，初始面位置 z_init = 0.0
        
        示例:
            >>> source = GaussianBeamSource(
            ...     wavelength=0.633,
            ...     w0=1.0,
            ...     z0=-50.0,
            ... )
            >>> beam = source.to_gaussian_beam()
            >>> print(beam)
            GaussianBeam(λ=0.633μm, w0=1.0mm, z0=-50.0mm, M²=1.0, zR=...)
        """
        # 延迟导入以避免循环依赖
        from gaussian_beam_simulation.gaussian_beam import GaussianBeam
        
        return GaussianBeam(
            wavelength=self.wavelength,
            w0=self.w0,
            z0=self.z0,
            m2=self.m2,
            z_init=0.0,
        )
    
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"GaussianBeamSource(λ={self.wavelength}μm, w0={self.w0}mm, "
            f"z0={self.z0}mm, M²={self.m2}, zR={self.zR:.2f}mm)"
        )

