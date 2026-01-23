"""
GaussianSource 类模块

提供高斯光源定义功能。
"""

from typing import Optional
import numpy as np


class GaussianSource:
    """高斯光源定义
    
    定义入射高斯光束的参数。
    
    属性:
        wavelength_um: 波长 (μm)
        w0_mm: 束腰半径 (mm)
        grid_size: 网格大小
        physical_size_mm: 物理尺寸 (mm)
        z0_mm: 束腰位置 (mm)
        beam_diam_fraction: PROPER beam_diam_fraction 参数
    
    示例:
        >>> source = GaussianSource(wavelength_um=0.633, w0_mm=5.0)
        >>> print(f"瑞利距离: {source.z_rayleigh_mm:.2f} mm")
        >>> source.print_info()
    """
    
    def __init__(
        self,
        wavelength_um: float,
        w0_mm: float,
        grid_size: int = 256,
        physical_size_mm: Optional[float] = None,
        z0_mm: float = 0.0,
        beam_diam_fraction: Optional[float] = None,
    ) -> None:
        """创建高斯光源
        
        参数:
            wavelength_um: 波长 (μm)，必须为正数
            w0_mm: 束腰半径 (mm)，必须为正数
            grid_size: 网格大小，默认 256，必须为正整数
            physical_size_mm: 物理尺寸 (mm)，默认 8 倍束腰
            z0_mm: 束腰位置 (mm)，默认 0
            beam_diam_fraction: PROPER beam_diam_fraction 参数（可选）
        
        异常:
            ValueError: 参数值无效
        
        示例:
            >>> # 基本用法
            >>> source = GaussianSource(wavelength_um=0.633, w0_mm=5.0)
            
            >>> # 指定所有参数
            >>> source = GaussianSource(
            ...     wavelength_um=1.064,
            ...     w0_mm=10.0,
            ...     grid_size=512,
            ...     physical_size_mm=100.0,
            ...     z0_mm=0.0,
            ...     beam_diam_fraction=0.5,
            ... )
        """
        # 参数验证
        if wavelength_um <= 0:
            raise ValueError(f"无效参数: wavelength_um = {wavelength_um}，必须为正数")
        if w0_mm <= 0:
            raise ValueError(f"无效参数: w0_mm = {w0_mm}，必须为正数")
        if grid_size <= 0:
            raise ValueError(f"无效参数: grid_size = {grid_size}，必须为正整数")
        if not isinstance(grid_size, int):
            raise ValueError(f"无效参数: grid_size = {grid_size}，必须为整数")
        if physical_size_mm is not None and physical_size_mm <= 0:
            raise ValueError(f"无效参数: physical_size_mm = {physical_size_mm}，必须为正数")
        if beam_diam_fraction is not None and beam_diam_fraction <= 0:
            raise ValueError(f"无效参数: beam_diam_fraction = {beam_diam_fraction}，必须为正数")
        
        # 存储参数
        self._wavelength_um = wavelength_um
        self._w0_mm = w0_mm
        self._grid_size = grid_size
        self._z0_mm = z0_mm
        self._beam_diam_fraction = beam_diam_fraction
        
        # 计算默认物理尺寸（8 倍束腰）
        if physical_size_mm is None:
            self._physical_size_mm = 8.0 * w0_mm
        else:
            self._physical_size_mm = physical_size_mm
    
    @property
    def wavelength_um(self) -> float:
        """波长 (μm)"""
        return self._wavelength_um
    
    @property
    def w0_mm(self) -> float:
        """束腰半径 (mm)"""
        return self._w0_mm
    
    @property
    def grid_size(self) -> int:
        """网格大小"""
        return self._grid_size
    
    @property
    def physical_size_mm(self) -> float:
        """物理尺寸 (mm)"""
        return self._physical_size_mm
    
    @property
    def z0_mm(self) -> float:
        """束腰位置 (mm)"""
        return self._z0_mm
    
    @property
    def beam_diam_fraction(self) -> Optional[float]:
        """PROPER beam_diam_fraction 参数"""
        return self._beam_diam_fraction
    
    @property
    def z_rayleigh_mm(self) -> float:
        """瑞利距离 (mm)
        
        计算公式: z_R = π × w0² / λ
        
        其中:
            - w0: 束腰半径 (mm)
            - λ: 波长 (mm)
        
        返回:
            瑞利距离 (mm)
        """
        # 将波长从 μm 转换为 mm
        wavelength_mm = self._wavelength_um * 1e-3
        # 计算瑞利距离: z_R = π × w0² / λ
        z_rayleigh = np.pi * self._w0_mm**2 / wavelength_mm
        return float(z_rayleigh)
    
    @property
    def wavelength_mm(self) -> float:
        """波长 (mm)，便于内部计算使用"""
        return self._wavelength_um * 1e-3
    
    def print_info(self) -> None:
        """打印光源参数
        
        输出格式化的光源参数信息，包括：
        - 波长
        - 束腰半径
        - 瑞利距离
        - 网格大小
        - 物理尺寸
        - 束腰位置
        - beam_diam_fraction（如果设置）
        
        示例:
            >>> source = GaussianSource(wavelength_um=0.633, w0_mm=5.0)
            >>> source.print_info()
            ╔══════════════════════════════════════════════════════════════╗
            ║                      高斯光源参数                              ║
            ╠══════════════════════════════════════════════════════════════╣
            ║  波长:           0.633 μm                                     ║
            ║  束腰半径:       5.000 mm                                     ║
            ║  瑞利距离:       124140.05 mm                                 ║
            ║  网格大小:       256 × 256                                    ║
            ║  物理尺寸:       40.000 mm × 40.000 mm                        ║
            ║  束腰位置:       0.000 mm                                     ║
            ╚══════════════════════════════════════════════════════════════╝
        """
        # 计算瑞利距离
        z_r = self.z_rayleigh_mm
        
        # 打印格式化信息
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║                      高斯光源参数                              ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print(f"║  波长:           {self._wavelength_um:.3f} μm{' ' * 37}║")
        print(f"║  束腰半径:       {self._w0_mm:.3f} mm{' ' * 37}║")
        print(f"║  瑞利距离:       {z_r:.2f} mm{' ' * (39 - len(f'{z_r:.2f}'))}║")
        print(f"║  网格大小:       {self._grid_size} × {self._grid_size}{' ' * (40 - len(f'{self._grid_size} × {self._grid_size}'))}║")
        print(f"║  物理尺寸:       {self._physical_size_mm:.3f} mm × {self._physical_size_mm:.3f} mm{' ' * (26 - len(f'{self._physical_size_mm:.3f}'))}║")
        print(f"║  束腰位置:       {self._z0_mm:.3f} mm{' ' * 37}║")
        
        if self._beam_diam_fraction is not None:
            print(f"║  beam_diam_fraction: {self._beam_diam_fraction:.3f}{' ' * 35}║")
        
        print("╚══════════════════════════════════════════════════════════════╝")
    
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"GaussianSource("
            f"wavelength_um={self._wavelength_um}, "
            f"w0_mm={self._w0_mm}, "
            f"grid_size={self._grid_size}, "
            f"physical_size_mm={self._physical_size_mm}, "
            f"z0_mm={self._z0_mm}"
            f")"
        )
