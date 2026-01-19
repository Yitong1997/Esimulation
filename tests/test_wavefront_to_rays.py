"""
波前采样为几何光线模块的测试代码

本测试文件验证：
1. 波前复振幅到相位的转换
2. 相位面的创建和光线追迹
3. 出射光束 OPD 与输入波前相位的一致性

测试方法：
- 创建已知相位分布的波前
- 通过相位面进行光线追迹
- 比较出射光线的 OPD 与原始相位的差值

作者：混合光学仿真项目
"""

import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose

from wavefront_to_rays import WavefrontToRaysSampler, create_phase_surface_optic
from wavefront_to_rays.phase_surface import (
    wavefront_to_phase,
    phase_to_opd_waves,
    opd_waves_to_phase,
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def create_test_wavefront_spherical(
    grid_size: int = 64,
    physical_size: float = 10.0,
    defocus_waves: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """创建球面波前（离焦）用于测试
    
    注意：相位场定义在整个方形区域上，不使用圆形光瞳掩模。
    
    参数:
        grid_size: 网格大小
        physical_size: 物理尺寸（直径），单位：mm
        defocus_waves: 离焦量，单位：波长数
    
    返回:
        (wavefront, x_coords, y_coords) 元组
    """
    half_size = physical_size / 2.0
    x = np.linspace(-half_size, half_size, grid_size)
    y = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # 归一化半径
    R_norm = np.sqrt(X**2 + Y**2) / half_size
    
    # 离焦相位：W20 * (rho^2)，其中 W20 是离焦系数（波长数）
    # 相位 = 2π * W20 * rho^2
    # 注意：整个方形区域都有相位定义，不使用圆形掩模
    phase = 2 * np.pi * defocus_waves * R_norm**2
    
    # 创建复振幅（整个方形区域振幅为 1）
    amplitude = np.ones_like(phase)
    wavefront = amplitude * np.exp(1j * phase)
    
    return wavefront, x, y


def create_test_wavefront_tilt(
    grid_size: int = 64,
    physical_size: float = 10.0,
    tilt_x_waves: float = 0.5,
    tilt_y_waves: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """创建倾斜波前用于测试
    
    注意：相位场定义在整个方形区域上，不使用圆形光瞳掩模。
    
    参数:
        grid_size: 网格大小
        physical_size: 物理尺寸（直径），单位：mm
        tilt_x_waves: X 方向倾斜量，单位：波长数
        tilt_y_waves: Y 方向倾斜量，单位：波长数
    
    返回:
        (wavefront, x_coords, y_coords) 元组
    """
    half_size = physical_size / 2.0
    x = np.linspace(-half_size, half_size, grid_size)
    y = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # 归一化坐标
    X_norm = X / half_size
    Y_norm = Y / half_size
    
    # 倾斜相位：2π * (Wx * x + Wy * y)
    # 注意：整个方形区域都有相位定义，不使用圆形掩模
    phase = 2 * np.pi * (tilt_x_waves * X_norm + tilt_y_waves * Y_norm)
    
    # 创建复振幅（整个方形区域振幅为 1）
    amplitude = np.ones_like(phase)
    wavefront = amplitude * np.exp(1j * phase)
    
    return wavefront, x, y


def create_test_wavefront_zernike(
    grid_size: int = 64,
    physical_size: float = 10.0,
    zernike_coeffs: dict = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """创建 Zernike 像差波前用于测试
    
    注意：相位场定义在整个方形区域上，不使用圆形光瞳掩模。
    
    参数:
        grid_size: 网格大小
        physical_size: 物理尺寸（直径），单位：mm
        zernike_coeffs: Zernike 系数字典，格式 {(n, m): coeff}
                       例如 {(2, 0): 0.5} 表示离焦
    
    返回:
        (wavefront, x_coords, y_coords) 元组
    """
    if zernike_coeffs is None:
        zernike_coeffs = {(2, 0): 0.5}  # 默认离焦
    
    half_size = physical_size / 2.0
    x = np.linspace(-half_size, half_size, grid_size)
    y = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # 归一化极坐标
    rho = np.sqrt(X**2 + Y**2) / half_size
    theta = np.arctan2(Y, X)
    
    # 计算 Zernike 多项式
    # 注意：整个方形区域都有相位定义，不使用圆形掩模
    phase = np.zeros_like(rho)
    
    for (n, m), coeff in zernike_coeffs.items():
        Z = _zernike_polynomial(n, m, rho, theta)
        phase += coeff * Z
    
    # 转换为弧度
    phase = 2 * np.pi * phase
    
    # 创建复振幅（整个方形区域振幅为 1）
    amplitude = np.ones_like(phase)
    wavefront = amplitude * np.exp(1j * phase)
    
    return wavefront, x, y


def _zernike_polynomial(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """计算 Zernike 多项式（简化版本）
    
    仅实现常用的几个 Zernike 项：
    - (0, 0): 活塞
    - (1, 1): X 倾斜
    - (1, -1): Y 倾斜
    - (2, 0): 离焦
    - (2, 2): 像散
    - (3, 1): 彗差
    """
    if (n, m) == (0, 0):
        return np.ones_like(rho)
    elif (n, m) == (1, 1):
        return rho * np.cos(theta)
    elif (n, m) == (1, -1):
        return rho * np.sin(theta)
    elif (n, m) == (2, 0):
        return 2 * rho**2 - 1
    elif (n, m) == (2, 2):
        return rho**2 * np.cos(2 * theta)
    elif (n, m) == (2, -2):
        return rho**2 * np.sin(2 * theta)
    elif (n, m) == (3, 1):
        return (3 * rho**3 - 2 * rho) * np.cos(theta)
    elif (n, m) == (3, -1):
        return (3 * rho**3 - 2 * rho) * np.sin(theta)
    else:
        raise ValueError(f"未实现的 Zernike 项: ({n}, {m})")


class TestWavefrontToRays:
    """波前采样为几何光线的测试类"""
    
    def test_phase_extraction(self):
        """测试相位提取功能"""
        # 创建已知相位的波前
        phase_input = np.array([[0, np.pi/4], [np.pi/2, np.pi]])
        amplitude = np.ones_like(phase_input)
        wavefront = amplitude * np.exp(1j * phase_input)
        
        # 提取相位
        phase_output = wavefront_to_phase(wavefront)
        
        # 验证
        assert_allclose(phase_output, phase_input, rtol=1e-10)
        print("✓ 相位提取测试通过")
    
    def test_phase_opd_conversion(self):
        """测试相位与 OPD 的转换"""
        phase = np.array([0, np.pi, 2*np.pi, -np.pi])
        
        # 相位转 OPD
        opd = phase_to_opd_waves(phase)
        expected_opd = np.array([0, 0.5, 1.0, -0.5])
        assert_allclose(opd, expected_opd, rtol=1e-10)
        
        # OPD 转相位
        phase_back = opd_waves_to_phase(opd)
        assert_allclose(phase_back, phase, rtol=1e-10)
        
        print("✓ 相位-OPD 转换测试通过")
    
    def test_sampler_initialization(self):
        """测试采样器初始化"""
        wavefront, x, y = create_test_wavefront_spherical(
            grid_size=32,
            physical_size=10.0,
            defocus_waves=0.5,
        )
        
        sampler = WavefrontToRaysSampler(
            wavefront_amplitude=wavefront,
            physical_size=10.0,
            wavelength=0.55,
            num_rays=50,
        )
        
        # 验证基本属性
        assert sampler.phase_grid.shape == (32, 32)
        assert sampler.amplitude_grid.shape == (32, 32)
        assert sampler.optic is not None
        assert sampler.output_rays is not None
        
        print("✓ 采样器初始化测试通过")
    
    def test_ray_output(self):
        """测试光线输出"""
        wavefront, x, y = create_test_wavefront_tilt(
            grid_size=32,
            physical_size=10.0,
            tilt_x_waves=0.1,
            tilt_y_waves=0.0,
        )
        
        sampler = WavefrontToRaysSampler(
            wavefront_amplitude=wavefront,
            physical_size=10.0,
            wavelength=0.55,
            num_rays=50,
        )
        
        # 获取光线数据
        rays = sampler.get_output_rays()
        x_pos, y_pos = sampler.get_ray_positions()
        L, M, N = sampler.get_ray_directions()
        opd = sampler.get_ray_opd()
        intensity = sampler.get_ray_intensity()
        
        # 验证数据形状
        assert len(x_pos) > 0
        assert len(L) == len(x_pos)
        assert len(opd) == len(x_pos)
        
        # 验证方向余弦归一化
        direction_mag = np.sqrt(L**2 + M**2 + N**2)
        assert_allclose(direction_mag, 1.0, rtol=1e-6)
        
        print("✓ 光线输出测试通过")


def visualize_wavefront_to_rays_test():
    """可视化测试：验证出射光束 OPD 与输入波前相位的一致性
    
    注意：
    - 相位场定义在整个方形区域上
    - 在计算 OPD 差值时，使用 90% 的圆形光瞳 ROI 进行验证
    """
    
    print("=" * 60)
    print("波前采样为几何光线 - 可视化验证测试")
    print("=" * 60)
    
    # 测试参数
    grid_size = 64
    physical_size = 10.0  # mm
    wavelength = 0.55  # μm
    num_rays = 200
    pupil_roi_ratio = 0.9  # 90% 圆形光瞳 ROI
    
    # 创建测试波前（离焦 + 像散）
    # 注意：相位场定义在整个方形区域上
    defocus_waves = 0.5
    astigmatism_waves = 0.3
    
    wavefront, x_coords, y_coords = create_test_wavefront_zernike(
        grid_size=grid_size,
        physical_size=physical_size,
        zernike_coeffs={
            (2, 0): defocus_waves,   # 离焦
            (2, 2): astigmatism_waves,  # 像散
        },
    )
    
    print(f"\n测试参数:")
    print(f"  网格大小: {grid_size} x {grid_size}")
    print(f"  物理尺寸: {physical_size} mm")
    print(f"  波长: {wavelength} μm")
    print(f"  采样光线数: {num_rays}")
    print(f"  离焦量: {defocus_waves} 波长")
    print(f"  像散量: {astigmatism_waves} 波长")
    print(f"  圆形光瞳 ROI: {pupil_roi_ratio * 100:.0f}%")
    
    # 创建采样器
    sampler = WavefrontToRaysSampler(
        wavefront_amplitude=wavefront,
        physical_size=physical_size,
        wavelength=wavelength,
        num_rays=num_rays,
        distribution='hexapolar',
    )
    
    # 获取结果
    x_pos, y_pos = sampler.get_ray_positions()
    opd_rays = sampler.get_ray_opd()
    intensity = sampler.get_ray_intensity()
    
    # 计算输入相位对应的 OPD（波长数）
    input_opd_grid = sampler.phase_to_opd_waves()
    
    # 在光线位置处插值输入 OPD
    from scipy.interpolate import RectBivariateSpline
    
    interp = RectBivariateSpline(y_coords, x_coords, input_opd_grid)
    input_opd_at_rays = interp.ev(y_pos, x_pos)
    
    # 获取主光线位置的输入 OPD（作为参考）
    input_opd_chief = interp.ev(0.0, 0.0)
    
    # 计算相对于主光线的输入 OPD
    input_opd_relative = input_opd_at_rays - input_opd_chief
    
    # 计算差值
    # opd_rays 是相对于主光线的输出 OPD
    # input_opd_relative 是相对于主光线的输入 OPD
    # 相位面引入的 OPD 是负的（相位增加 = 光程减少）
    # 所以预期：opd_rays = -input_opd_relative
    opd_difference = opd_rays - (-input_opd_relative)
    
    # 计算光线位置的归一化半径
    half_size = physical_size / 2.0
    r_norm = np.sqrt(x_pos**2 + y_pos**2) / half_size
    
    # 创建 90% 圆形光瞳 ROI 掩模
    # 只考虑在 90% 圆形光瞳内且强度 > 0 的有效光线
    roi_mask = (r_norm <= pupil_roi_ratio) & (intensity > 0.5)
    valid_diff = opd_difference[roi_mask]
    
    # 统计（仅在 ROI 内）
    mean_diff = np.mean(valid_diff)
    std_diff = np.std(valid_diff)
    max_diff = np.max(np.abs(valid_diff))
    
    print(f"\n验证结果（{pupil_roi_ratio * 100:.0f}% 圆形光瞳 ROI 内）:")
    print(f"  总光线数: {len(x_pos)}")
    print(f"  ROI 内有效光线数: {np.sum(roi_mask)}")
    print(f"  OPD 差值均值: {mean_diff:.6f} 波长")
    print(f"  OPD 差值标准差: {std_diff:.6f} 波长")
    print(f"  OPD 差值最大值: {max_diff:.6f} 波长")
    
    # 创建可视化图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 输入波前相位（方形区域）
    ax1 = axes[0, 0]
    X, Y = np.meshgrid(x_coords, y_coords)
    im1 = ax1.pcolormesh(X, Y, sampler.phase_grid, cmap='RdBu_r', shading='auto')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_title('输入波前相位 (弧度)\n方形区域')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)
    
    # 2. 输入波前 OPD（方形区域）
    ax2 = axes[0, 1]
    im2 = ax2.pcolormesh(X, Y, input_opd_grid, cmap='RdBu_r', shading='auto')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('输入波前 OPD (波长数)\n方形区域')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2)
    
    # 3. 出射光线 OPD（显示所有光线，ROI 边界用圆圈标出）
    ax3 = axes[0, 2]
    # 先绘制所有光线
    all_valid = intensity > 0.5
    scatter3 = ax3.scatter(x_pos[all_valid], y_pos[all_valid], 
                           c=-opd_rays[all_valid], cmap='RdBu_r', s=10, alpha=0.5)
    # 绘制 ROI 边界圆
    theta_circle = np.linspace(0, 2*np.pi, 100)
    roi_radius = half_size * pupil_roi_ratio
    ax3.plot(roi_radius * np.cos(theta_circle), roi_radius * np.sin(theta_circle), 
             'g--', linewidth=2, label=f'{pupil_roi_ratio*100:.0f}% ROI')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_title('出射光线 OPD (波长数)')
    ax3.set_aspect('equal')
    ax3.set_xlim(-physical_size/2, physical_size/2)
    ax3.set_ylim(-physical_size/2, physical_size/2)
    ax3.legend(loc='upper right')
    plt.colorbar(scatter3, ax=ax3)
    
    # 4. 输入 OPD 在光线位置的插值（ROI 内，相对于主光线）
    ax4 = axes[1, 0]
    scatter4 = ax4.scatter(x_pos[roi_mask], y_pos[roi_mask], 
                           c=input_opd_relative[roi_mask], cmap='RdBu_r', s=10)
    # 绘制 ROI 边界圆
    ax4.plot(roi_radius * np.cos(theta_circle), roi_radius * np.sin(theta_circle), 
             'g--', linewidth=2)
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    ax4.set_title(f'输入 OPD（相对于主光线）\n{pupil_roi_ratio*100:.0f}% ROI 内')
    ax4.set_aspect('equal')
    ax4.set_xlim(-physical_size/2, physical_size/2)
    ax4.set_ylim(-physical_size/2, physical_size/2)
    plt.colorbar(scatter4, ax=ax4)
    
    # 5. OPD 差值（ROI 内）
    ax5 = axes[1, 1]
    scatter5 = ax5.scatter(x_pos[roi_mask], y_pos[roi_mask], 
                           c=valid_diff, cmap='RdBu_r', s=10)
    # 绘制 ROI 边界圆
    ax5.plot(roi_radius * np.cos(theta_circle), roi_radius * np.sin(theta_circle), 
             'g--', linewidth=2)
    ax5.set_xlabel('X (mm)')
    ax5.set_ylabel('Y (mm)')
    ax5.set_title(f'OPD 差值 (波长数) - {pupil_roi_ratio*100:.0f}% ROI\n均值={mean_diff:.4f}, 标准差={std_diff:.4f}')
    ax5.set_aspect('equal')
    ax5.set_xlim(-physical_size/2, physical_size/2)
    ax5.set_ylim(-physical_size/2, physical_size/2)
    plt.colorbar(scatter5, ax=ax5)
    
    # 6. OPD 差值直方图（ROI 内）
    ax6 = axes[1, 2]
    ax6.hist(valid_diff, bins=30, edgecolor='black', alpha=0.7)
    ax6.axvline(mean_diff, color='r', linestyle='--', label=f'均值: {mean_diff:.4f}')
    ax6.set_xlabel('OPD 差值 (波长数)')
    ax6.set_ylabel('频数')
    ax6.set_title(f'OPD 差值分布 ({pupil_roi_ratio*100:.0f}% ROI 内)')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('wavefront_to_rays_validation.png', dpi=150)
    print(f"\n可视化结果已保存到: wavefront_to_rays_validation.png")
    plt.show()
    
    # 判断测试是否通过
    tolerance = 0.01  # 允许 0.01 波长的误差
    if max_diff < tolerance:
        print(f"\n✓ 测试通过！OPD 差值在容差范围内 (< {tolerance} 波长)")
        return True
    else:
        print(f"\n✗ 测试失败！OPD 差值超出容差范围 (> {tolerance} 波长)")
        return False


def run_unit_tests():
    """运行单元测试"""
    print("=" * 60)
    print("运行单元测试")
    print("=" * 60)
    
    test = TestWavefrontToRays()
    test.test_phase_extraction()
    test.test_phase_opd_conversion()
    test.test_sampler_initialization()
    test.test_ray_output()
    
    print("\n所有单元测试通过！")


if __name__ == "__main__":
    # 运行单元测试
    run_unit_tests()
    
    print("\n")
    
    # 运行可视化验证测试
    visualize_wavefront_to_rays_test()
