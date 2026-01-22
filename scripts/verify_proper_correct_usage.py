"""
验证 PROPER 的正确使用方式

核心理解：
1. PROPER 的 wfarr 存储的是相对于理想高斯光束的偏差
2. 对于理想高斯光束，wfarr 应该保持为 1（振幅=1，相位=0）
3. 实际的高斯光束形状由 PROPER 内部参数追踪

正确的使用方式：
1. 写入 PROPER：
   - 计算仿真复振幅相对于 pilot beam 的残差
   - 将残差写入 wfarr
   - 设置 wfo 的所有属性以匹配 pilot beam 参数

2. 读取 PROPER：
   - 直接读取 wfarr（这是残差）
   - 加上 pilot beam 的相位得到绝对相位
   - 如果相位超过 [-π, π]，需要解包裹
"""

import numpy as np
import matplotlib.pyplot as plt
import proper


def gaussian_beam_params(w0, wavelength, z):
    """计算高斯光束在位置 z 处的参数（严格公式）"""
    z_R = np.pi * w0**2 / wavelength
    
    if abs(z) < 1e-12:
        return w0, np.inf, 0.0, z_R
    
    w = w0 * np.sqrt(1 + (z / z_R)**2)
    R = z * (1 + (z_R / z)**2)
    gouy = np.arctan(z / z_R)
    
    return w, R, gouy, z_R


def create_theoretical_gaussian(grid_size, sampling_m, w0, wavelength, z):
    """创建理论高斯光束复振幅（相对于主光线）"""
    w, R, gouy, z_R = gaussian_beam_params(w0, wavelength, z)
    
    n = grid_size
    x = (np.arange(n) - n // 2) * sampling_m
    X, Y = np.meshgrid(x, x)
    r_sq = X**2 + Y**2
    
    k = 2 * np.pi / wavelength
    
    # 高斯光束复振幅
    amplitude = (w0 / w) * np.exp(-r_sq / w**2)
    
    if np.isinf(R):
        phase = np.zeros_like(r_sq)
    else:
        phase = k * r_sq / (2 * R)
    
    complex_amplitude = amplitude * np.exp(1j * phase)
    
    return complex_amplitude, {'w0': w0, 'w': w, 'R': R, 'gouy': gouy, 'z_R': z_R, 'z': z}


def compute_pilot_beam_amplitude(grid_size, sampling_m, w0, wavelength, z):
    """计算 pilot beam 的复振幅（理想高斯光束）"""
    return create_theoretical_gaussian(grid_size, sampling_m, w0, wavelength, z)


def setup_proper_correctly(simulation_amplitude, pilot_params, grid_size, sampling_m, wavelength):
    """正确设置 PROPER wfo 对象
    
    关键：wfarr 存储的是仿真复振幅相对于 pilot beam 的残差
    """
    w0 = pilot_params['w0']
    beam_diameter = 2 * w0
    z = pilot_params['z']
    z_R = pilot_params['z_R']
    
    # 计算 beam_diam_fraction
    grid_width = grid_size * sampling_m
    beam_diam_fraction = beam_diameter / grid_width
    
    # 初始化 wfo
    wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, beam_diam_fraction)
    
    # 设置所有属性
    wfo.z = z
    wfo.z_w0 = 0.0
    wfo.w0 = w0
    wfo.z_Rayleigh = z_R
    wfo.dx = sampling_m
    
    # 确定参考面类型
    rayleigh_factor = proper.rayleigh_factor
    if abs(wfo.z - wfo.z_w0) < rayleigh_factor * wfo.z_Rayleigh:
        wfo.reference_surface = "PLANAR"
        wfo.beam_type_old = "INSIDE_"
    else:
        wfo.reference_surface = "SPHERI"
        wfo.beam_type_old = "OUTSIDE"
    
    # 计算 pilot beam 复振幅
    pilot_amplitude, _ = compute_pilot_beam_amplitude(
        grid_size, sampling_m, w0, wavelength, z
    )
    
    # 计算残差：仿真复振幅 / pilot beam 复振幅
    # 这样 wfarr 存储的就是相对于 pilot beam 的偏差
    residual = simulation_amplitude / pilot_amplitude
    
    # 处理 pilot beam 振幅为零的区域
    mask = np.abs(pilot_amplitude) > 1e-10
    residual[~mask] = 0.0
    
    # 写入 wfarr
    wfo.wfarr = proper.prop_shift_center(residual.astype(np.complex128))
    
    return wfo, pilot_amplitude


def read_simulation_correctly(wfo, pilot_params, grid_size, sampling_m, wavelength):
    """正确从 PROPER 读取仿真复振幅
    
    关键：wfarr 是残差，需要乘以 pilot beam 得到仿真复振幅
    """
    # 读取残差
    residual_amp = proper.prop_get_amplitude(wfo)
    residual_phase = proper.prop_get_phase(wfo)
    residual = residual_amp * np.exp(1j * residual_phase)
    
    # 计算 pilot beam 复振幅
    pilot_amplitude, _ = compute_pilot_beam_amplitude(
        grid_size, sampling_m, pilot_params['w0'], wavelength, pilot_params['z']
    )
    
    # 仿真复振幅 = 残差 × pilot beam
    simulation_amplitude = residual * pilot_amplitude
    
    return simulation_amplitude, residual, pilot_amplitude
