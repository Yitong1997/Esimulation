# -*- coding: utf-8 -*-
"""
Julia 对比验证模块

本模块使用 juliacall 调用原始 Julia AngularSpectrumMethod 实现，
对比 Python 和 Julia 输出的数值一致性。

主要功能：
- 初始化 Julia 环境
- 调用 Julia ASM 函数
- 计算相对误差
- 验证能量守恒

依赖：
- juliacall: Python 调用 Julia 的桥接库
- Julia AngularSpectrumMethod 包

Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass

# 类型别名
ComplexArray = NDArray[np.complexfloating]


@dataclass
class ComparisonResult:
    """
    对比验证结果
    
    属性：
        method_name: 方法名称
        max_rel_error: 最大相对误差
        mean_rel_error: 平均相对误差
        max_abs_error: 最大绝对误差
        input_energy: 输入能量
        py_output_energy: Python 输出能量
        jl_output_energy: Julia 输出能量
        energy_ratio_py: Python 能量比（输出/输入）
        energy_ratio_jl: Julia 能量比（输出/输入）
        passed: 是否通过验证
        error_threshold: 误差阈值
    """
    method_name: str
    max_rel_error: float
    mean_rel_error: float
    max_abs_error: float
    input_energy: float
    py_output_energy: float
    jl_output_energy: float
    energy_ratio_py: float
    energy_ratio_jl: float
    passed: bool
    error_threshold: float = 1e-6


class JuliaComparison:
    """
    Julia 对比验证类
    
    提供与 Julia AngularSpectrumMethod 包的对比验证功能。
    
    示例：
        >>> from angular_spectrum_method.validation import JuliaComparison
        >>> 
        >>> # 初始化
        >>> jc = JuliaComparison()
        >>> 
        >>> # 对比 ASM
        >>> u = np.ones((32, 32), dtype=complex)
        >>> result = jc.compare_asm(u, 633e-9, 1e-6, 1e-6, 0.01)
        >>> print(f"最大相对误差: {result.max_rel_error}")
    
    Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7
    """
    
    def __init__(self):
        """
        初始化 Julia 环境
        
        异常：
            ImportError: 如果 juliacall 未安装
            RuntimeError: 如果 Julia 环境初始化失败
        """
        self._jl = None
        self._initialized = False
        self._init_julia()
    
    def _init_julia(self) -> None:
        """初始化 Julia 环境并加载 AngularSpectrumMethod"""
        try:
            from juliacall import Main as jl
            self._jl = jl
            
            # 加载 AngularSpectrumMethod
            jl.seval('using AngularSpectrumMethod')
            
            # 定义辅助函数（将 PyArray 转换为 Julia Array）
            jl.seval('''
            # ASM
            function py_asm(u_py, wavelength, dx, dy, z; expand=true)
                u = collect(u_py)
                return AngularSpectrumMethod.ASM(u, wavelength, dx, dy, z; expand=expand)
            end
            
            # BandLimitedASM
            function py_band_limited_asm(u_py, wavelength, dx, dy, z; expand=true)
                u = collect(u_py)
                return AngularSpectrumMethod.BandLimitedASM(u, wavelength, dx, dy, z; expand=expand)
            end
            
            # ScalableASM
            function py_scalable_asm(u_py, wavelength, dx, dy, z; expand=true)
                u = collect(u_py)
                return AngularSpectrumMethod.ScalableASM(u, wavelength, dx, dy, z; expand=expand)
            end
            
            # ScaledASM
            function py_scaled_asm(u_py, wavelength, dx, dy, z, R; expand=true)
                u = collect(u_py)
                return AngularSpectrumMethod.ScaledASM(u, wavelength, dx, dy, z, R; expand=expand)
            end
            
            # ShiftedASM
            function py_shifted_asm(u_py, wavelength, dx, dy, z, x0, y0; expand=true)
                u = collect(u_py)
                return AngularSpectrumMethod.ShiftedASM(u, wavelength, dx, dy, z, x0, y0; expand=expand)
            end
            
            # TiltedASM
            function py_tilted_asm(u_py, wavelength, dx, dy, T_py; expand=true, weight=false)
                u = collect(u_py)
                T = collect(T_py)
                return AngularSpectrumMethod.TiltedASM(u, wavelength, dx, dy, T; expand=expand, weight=weight)
            end
            ''')
            
            self._initialized = True
            
        except ImportError:
            raise ImportError(
                "juliacall 未安装。请使用 'pip install juliacall' 安装。"
            )
        except Exception as e:
            raise RuntimeError(f"Julia 环境初始化失败: {e}")
    
    @property
    def is_initialized(self) -> bool:
        """检查 Julia 环境是否已初始化"""
        return self._initialized
    
    def _compute_errors(
        self,
        py_result: ComplexArray,
        jl_result: ComplexArray
    ) -> Tuple[float, float, float]:
        """
        计算相对误差和绝对误差
        
        参数：
            py_result: Python 结果
            jl_result: Julia 结果
        
        返回：
            (max_rel_error, mean_rel_error, max_abs_error)
        """
        eps = 1e-15
        abs_error = np.abs(py_result - jl_result)
        rel_error = abs_error / (np.abs(jl_result) + eps)
        
        return (
            float(np.max(rel_error)),
            float(np.mean(rel_error)),
            float(np.max(abs_error))
        )
    
    def _compute_energy(self, u: ComplexArray, dx: float, dy: float) -> float:
        """计算光场能量"""
        return float(np.sum(np.abs(u)**2) * dx * dy)
    
    def compare_asm(
        self,
        u: ComplexArray,
        wavelength: float,
        dx: float,
        dy: float,
        z: float,
        expand: bool = True,
        error_threshold: float = 1e-6
    ) -> ComparisonResult:
        """
        对比 ASM 实现
        
        参数：
            u: 输入光场
            wavelength: 波长
            dx, dy: 采样间隔
            z: 传播距离
            expand: 是否零填充扩展
            error_threshold: 误差阈值
        
        返回：
            ComparisonResult 对象
        
        Validates: Requirements 10.1, 10.3
        """
        from angular_spectrum_method import asm
        
        # Python 结果
        py_result = asm(u, wavelength, dx, dy, z, expand=expand)
        
        # Julia 结果
        jl_result = np.array(self._jl.py_asm(u, wavelength, dx, dy, z, expand=expand))
        
        # 计算误差
        max_rel, mean_rel, max_abs = self._compute_errors(py_result, jl_result)
        
        # 计算能量
        input_energy = self._compute_energy(u, dx, dy)
        py_energy = self._compute_energy(py_result, dx, dy)
        jl_energy = self._compute_energy(jl_result, dx, dy)
        
        return ComparisonResult(
            method_name="ASM",
            max_rel_error=max_rel,
            mean_rel_error=mean_rel,
            max_abs_error=max_abs,
            input_energy=input_energy,
            py_output_energy=py_energy,
            jl_output_energy=jl_energy,
            energy_ratio_py=py_energy / input_energy if input_energy > 0 else 0,
            energy_ratio_jl=jl_energy / input_energy if input_energy > 0 else 0,
            passed=max_rel < error_threshold,
            error_threshold=error_threshold
        )
    
    def compare_band_limited_asm(
        self,
        u: ComplexArray,
        wavelength: float,
        dx: float,
        dy: float,
        z: float,
        expand: bool = True,
        error_threshold: float = 1e-6
    ) -> ComparisonResult:
        """
        对比 BandLimitedASM 实现
        
        Validates: Requirements 10.1, 10.3
        """
        from angular_spectrum_method import band_limited_asm
        
        py_result = band_limited_asm(u, wavelength, dx, dy, z, expand=expand)
        jl_result = np.array(self._jl.py_band_limited_asm(u, wavelength, dx, dy, z, expand=expand))
        
        max_rel, mean_rel, max_abs = self._compute_errors(py_result, jl_result)
        
        input_energy = self._compute_energy(u, dx, dy)
        py_energy = self._compute_energy(py_result, dx, dy)
        jl_energy = self._compute_energy(jl_result, dx, dy)
        
        return ComparisonResult(
            method_name="BandLimitedASM",
            max_rel_error=max_rel,
            mean_rel_error=mean_rel,
            max_abs_error=max_abs,
            input_energy=input_energy,
            py_output_energy=py_energy,
            jl_output_energy=jl_energy,
            energy_ratio_py=py_energy / input_energy if input_energy > 0 else 0,
            energy_ratio_jl=jl_energy / input_energy if input_energy > 0 else 0,
            passed=max_rel < error_threshold,
            error_threshold=error_threshold
        )
    
    def compare_scalable_asm(
        self,
        u: ComplexArray,
        wavelength: float,
        dx: float,
        dy: float,
        z: float,
        expand: bool = True,
        error_threshold: float = 1e-6
    ) -> ComparisonResult:
        """
        对比 ScalableASM 实现
        
        Validates: Requirements 10.1, 10.3
        """
        from angular_spectrum_method import scalable_asm
        
        py_result = scalable_asm(u, wavelength, dx, dy, z, expand=expand)
        jl_result = np.array(self._jl.py_scalable_asm(u, wavelength, dx, dy, z, expand=expand))
        
        max_rel, mean_rel, max_abs = self._compute_errors(py_result, jl_result)
        
        input_energy = self._compute_energy(u, dx, dy)
        py_energy = self._compute_energy(py_result, dx, dy)
        jl_energy = self._compute_energy(jl_result, dx, dy)
        
        return ComparisonResult(
            method_name="ScalableASM",
            max_rel_error=max_rel,
            mean_rel_error=mean_rel,
            max_abs_error=max_abs,
            input_energy=input_energy,
            py_output_energy=py_energy,
            jl_output_energy=jl_energy,
            energy_ratio_py=py_energy / input_energy if input_energy > 0 else 0,
            energy_ratio_jl=jl_energy / input_energy if input_energy > 0 else 0,
            passed=max_rel < error_threshold,
            error_threshold=error_threshold
        )
    
    def compare_scaled_asm(
        self,
        u: ComplexArray,
        wavelength: float,
        dx: float,
        dy: float,
        z: float,
        R: float,
        expand: bool = True,
        error_threshold: float = 1e-6
    ) -> ComparisonResult:
        """
        对比 ScaledASM 实现
        
        Validates: Requirements 10.1, 10.3
        """
        from angular_spectrum_method import scaled_asm
        
        py_result = scaled_asm(u, wavelength, dx, dy, z, R, expand=expand)
        jl_result = np.array(self._jl.py_scaled_asm(u, wavelength, dx, dy, z, R, expand=expand))
        
        max_rel, mean_rel, max_abs = self._compute_errors(py_result, jl_result)
        
        input_energy = self._compute_energy(u, dx, dy)
        py_energy = self._compute_energy(py_result, dx, dy)
        jl_energy = self._compute_energy(jl_result, dx, dy)
        
        return ComparisonResult(
            method_name="ScaledASM",
            max_rel_error=max_rel,
            mean_rel_error=mean_rel,
            max_abs_error=max_abs,
            input_energy=input_energy,
            py_output_energy=py_energy,
            jl_output_energy=jl_energy,
            energy_ratio_py=py_energy / input_energy if input_energy > 0 else 0,
            energy_ratio_jl=jl_energy / input_energy if input_energy > 0 else 0,
            passed=max_rel < error_threshold,
            error_threshold=error_threshold
        )
    
    def compare_shifted_asm(
        self,
        u: ComplexArray,
        wavelength: float,
        dx: float,
        dy: float,
        z: float,
        x0: float,
        y0: float,
        expand: bool = True,
        error_threshold: float = 1e-6
    ) -> ComparisonResult:
        """
        对比 ShiftedASM 实现
        
        Validates: Requirements 10.1, 10.3
        """
        from angular_spectrum_method import shifted_asm
        
        py_result = shifted_asm(u, wavelength, dx, dy, z, x0, y0, expand=expand)
        jl_result = np.array(self._jl.py_shifted_asm(u, wavelength, dx, dy, z, x0, y0, expand=expand))
        
        max_rel, mean_rel, max_abs = self._compute_errors(py_result, jl_result)
        
        input_energy = self._compute_energy(u, dx, dy)
        py_energy = self._compute_energy(py_result, dx, dy)
        jl_energy = self._compute_energy(jl_result, dx, dy)
        
        return ComparisonResult(
            method_name="ShiftedASM",
            max_rel_error=max_rel,
            mean_rel_error=mean_rel,
            max_abs_error=max_abs,
            input_energy=input_energy,
            py_output_energy=py_energy,
            jl_output_energy=jl_energy,
            energy_ratio_py=py_energy / input_energy if input_energy > 0 else 0,
            energy_ratio_jl=jl_energy / input_energy if input_energy > 0 else 0,
            passed=max_rel < error_threshold,
            error_threshold=error_threshold
        )
    
    def compare_tilted_asm(
        self,
        u: ComplexArray,
        wavelength: float,
        dx: float,
        dy: float,
        T: NDArray[np.floating],
        expand: bool = True,
        weight: bool = False,
        error_threshold: float = 1e-6
    ) -> ComparisonResult:
        """
        对比 TiltedASM 实现
        
        Validates: Requirements 10.1, 10.3
        """
        from angular_spectrum_method import tilted_asm
        
        py_result = tilted_asm(u, wavelength, dx, dy, T, expand=expand, weight=weight)
        jl_result = np.array(self._jl.py_tilted_asm(u, wavelength, dx, dy, T, expand=expand, weight=weight))
        
        max_rel, mean_rel, max_abs = self._compute_errors(py_result, jl_result)
        
        input_energy = self._compute_energy(u, dx, dy)
        py_energy = self._compute_energy(py_result, dx, dy)
        jl_energy = self._compute_energy(jl_result, dx, dy)
        
        return ComparisonResult(
            method_name="TiltedASM",
            max_rel_error=max_rel,
            mean_rel_error=mean_rel,
            max_abs_error=max_abs,
            input_energy=input_energy,
            py_output_energy=py_energy,
            jl_output_energy=jl_energy,
            energy_ratio_py=py_energy / input_energy if input_energy > 0 else 0,
            energy_ratio_jl=jl_energy / input_energy if input_energy > 0 else 0,
            passed=max_rel < error_threshold,
            error_threshold=error_threshold
        )
    
    def run_all_comparisons(
        self,
        u: ComplexArray,
        wavelength: float,
        dx: float,
        dy: float,
        z: float,
        T: Optional[NDArray[np.floating]] = None,
        R: float = 0.5,
        x0: float = 0.0,
        y0: float = 0.0,
        error_threshold: float = 1e-6
    ) -> Dict[str, ComparisonResult]:
        """
        运行所有方法的对比验证
        
        参数：
            u: 输入光场
            wavelength: 波长
            dx, dy: 采样间隔
            z: 传播距离
            T: 旋转矩阵（TiltedASM 用）
            R: 缩放因子（ScaledASM 用）
            x0, y0: 偏移量（ShiftedASM 用）
            error_threshold: 误差阈值
        
        返回：
            方法名到 ComparisonResult 的字典
        
        Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7
        """
        results = {}
        
        # ASM
        results['ASM'] = self.compare_asm(u, wavelength, dx, dy, z, error_threshold=error_threshold)
        
        # BandLimitedASM
        results['BandLimitedASM'] = self.compare_band_limited_asm(
            u, wavelength, dx, dy, z, error_threshold=error_threshold
        )
        
        # ScalableASM
        try:
            results['ScalableASM'] = self.compare_scalable_asm(
                u, wavelength, dx, dy, z, error_threshold=error_threshold
            )
        except Exception as e:
            print(f"ScalableASM 对比失败: {e}")
        
        # ScaledASM
        results['ScaledASM'] = self.compare_scaled_asm(
            u, wavelength, dx, dy, z, R, error_threshold=error_threshold
        )
        
        # ShiftedASM
        results['ShiftedASM'] = self.compare_shifted_asm(
            u, wavelength, dx, dy, z, x0, y0, error_threshold=error_threshold
        )
        
        # TiltedASM
        if T is not None:
            results['TiltedASM'] = self.compare_tilted_asm(
                u, wavelength, dx, dy, T, error_threshold=error_threshold
            )
        
        return results
    
    def print_report(self, results: Dict[str, ComparisonResult]) -> None:
        """
        打印对比验证报告
        
        参数：
            results: run_all_comparisons 返回的结果字典
        """
        print("=" * 70)
        print("Julia 对比验证报告")
        print("=" * 70)
        
        all_passed = True
        for name, result in results.items():
            status = "✓ 通过" if result.passed else "✗ 失败"
            all_passed = all_passed and result.passed
            
            print(f"\n{name}:")
            print(f"  状态: {status}")
            print(f"  最大相对误差: {result.max_rel_error:.2e} (阈值: {result.error_threshold:.2e})")
            print(f"  平均相对误差: {result.mean_rel_error:.2e}")
            print(f"  最大绝对误差: {result.max_abs_error:.2e}")
            print(f"  能量比 (Python): {result.energy_ratio_py:.6f}")
            print(f"  能量比 (Julia):  {result.energy_ratio_jl:.6f}")
        
        print("\n" + "=" * 70)
        if all_passed:
            print("总结: 所有方法验证通过 ✓")
        else:
            print("总结: 部分方法验证失败 ✗")
        print("=" * 70)
