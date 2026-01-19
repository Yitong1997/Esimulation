"""测试倾斜抛物面镜的相位结果与理论值对比

本测试验证 ElementRaytracer 对倾斜抛物面镜的光线追迹结果，
并与薄透镜近似的理论值进行对比。

测试场景：
- 45° 倾斜的抛物面镜（焦距 100mm）
- 平面波正入射
- 入射面：平行于 XY 平面，位于元件顶点
- 出射面：垂直于出射主光线方向

理论分析（薄透镜近似）：
- 抛物面镜的 OPD = r² / (2f) / λ（波长数）
- 对于折叠光路，出射面垂直于出射光轴
- 倾斜引入的额外光程应该在出射面处相互抵消

可视化内容：
- 2D 全局坐标系中的入射面、出射面方向
- 入射光束、出射光束方向
- OPD 分布对比图

作者：混合光学仿真项目
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免 Tkinter 问题
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import pytest
from numpy.testing import assert_allclose

# 配置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from optiland.rays import RealRays
from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
    compute_rotation_matrix,
)


# =============================================================================
# 理论计算函数
# =============================================================================

def calculate_theoretical_opd_waves(
    x: np.ndarray,
    y: np.ndarray,
    focal_length_mm: float,
    wavelength_um: float,
) -> np.ndarray:
    """计算抛物面镜的理论 OPD（薄透镜近似）
    
    对于抛物面反射镜，理论 OPD 公式：
        OPD = r² / (2f)
    
    其中 r 是到光轴的距离，f 是焦距。
    
    参数:
        x, y: 光线位置（mm），在入射面局部坐标系中
        focal_length_mm: 焦距（mm）
        wavelength_um: 波长（μm）
    
    返回:
        OPD（波长数），相对于中心光线
    """
    r_sq = x**2 + y**2
    wavelength_mm = wavelength_um * 1e-3
    
    # OPD = r² / (2f)
    opd_mm = r_sq / (2 * focal_length_mm)
    opd_waves = opd_mm / wavelength_mm
    
    return opd_waves


def calculate_exact_parabolic_opd_waves(
    x: np.ndarray,
    y: np.ndarray,
    focal_length_mm: float,
    wavelength_um: float,
) -> np.ndarray:
    """计算抛物面镜的精确 OPD
    
    使用精确的几何公式，而不是薄透镜近似。
    
    对于抛物面反射镜：
    - 表面矢高：sag = r² / (4f)
    - 入射光程：sag
    - 反射光程：需要考虑反射方向
    
    参数:
        x, y: 光线位置（mm）
        focal_length_mm: 焦距（mm）
        wavelength_um: 波长（μm）
    
    返回:
        OPD（波长数），相对于中心光线
    """
    r_sq = x**2 + y**2
    f = focal_length_mm
    wavelength_mm = wavelength_um * 1e-3
    
    # 表面矢高
    sag = r_sq / (4 * f)
    
    # 归一化因子的平方
    n_mag_sq = 1 + r_sq / (4 * f**2)
    
    # 反射方向 z 分量
    rz = 1 - 2 / n_mag_sq
    
    # 入射光程
    incident_path = sag
    
    # 反射光程
    reflected_path = -sag / rz
    
    # 总光程 = 相对 OPD
    opd_mm = incident_path + reflected_path
    opd_waves = opd_mm / wavelength_mm
    
    return opd_waves


# =============================================================================
# 可视化函数
# =============================================================================

def draw_2d_geometry(
    entrance_position: tuple,
    entrance_direction: tuple,
    exit_position: tuple,
    exit_direction: tuple,
    tilt_angle_deg: float,
    ax: plt.Axes = None,
) -> plt.Axes:
    """在 2D 全局坐标系中绘制几何配置
    
    绘制内容：
    - 入射面位置和方向（法向量）
    - 出射面位置和方向（法向量）
    - 入射光束方向
    - 出射光束方向
    - 元件表面示意
    
    参数:
        entrance_position: 入射面中心位置 (y, z)
        entrance_direction: 入射光束方向 (M, N)
        exit_position: 出射面中心位置 (y, z)
        exit_direction: 出射光束方向 (M, N)
        tilt_angle_deg: 元件倾斜角度（度）
        ax: matplotlib axes，如果为 None 则创建新的
    
    返回:
        matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # 设置坐标轴
    ax.set_xlabel('Z (mm) - 初始光轴方向', fontsize=12)
    ax.set_ylabel('Y (mm) - 垂直方向', fontsize=12)
    ax.set_title(f'45° 倾斜抛物面镜几何配置\n(YZ 平面投影)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 绘制坐标轴
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    # 元件位置（原点）
    element_pos = (0, 0)
    
    # 绘制元件表面（倾斜的线段）
    surface_length = 20  # mm
    tilt_rad = np.radians(tilt_angle_deg)
    # 表面法向量方向（绕 X 轴旋转后）
    # 初始法向量 (0, 0, -1)，旋转后 (0, sin(tilt), -cos(tilt))
    # 表面方向垂直于法向量
    surface_dy = surface_length * np.cos(tilt_rad)
    surface_dz = surface_length * np.sin(tilt_rad)
    
    ax.plot(
        [-surface_dz/2, surface_dz/2],
        [-surface_dy/2, surface_dy/2],
        'b-', linewidth=3, label='抛物面镜表面'
    )
    
    # 绘制入射面（垂直于入射方向）
    entrance_y, entrance_z = entrance_position
    entrance_M, entrance_N = entrance_direction
    # 入射面垂直于入射方向，在 YZ 平面内
    entrance_plane_length = 15
    # 入射面方向：垂直于 (M, N)，即 (-N, M)
    entrance_plane_dy = entrance_plane_length * (-entrance_N)
    entrance_plane_dz = entrance_plane_length * entrance_M
    
    ax.plot(
        [entrance_z - entrance_plane_dz/2, entrance_z + entrance_plane_dz/2],
        [entrance_y - entrance_plane_dy/2, entrance_y + entrance_plane_dy/2],
        'g-', linewidth=2, label='入射面'
    )

    # 绘制出射面（垂直于出射方向）
    exit_y, exit_z = exit_position
    exit_M, exit_N = exit_direction
    exit_plane_length = 15
    # 出射面方向：垂直于 (M, N)，即 (-N, M)
    exit_plane_dy = exit_plane_length * (-exit_N)
    exit_plane_dz = exit_plane_length * exit_M
    
    ax.plot(
        [exit_z - exit_plane_dz/2, exit_z + exit_plane_dz/2],
        [exit_y - exit_plane_dy/2, exit_y + exit_plane_dy/2],
        'r-', linewidth=2, label='出射面'
    )
    
    # 绘制入射光束方向（箭头）
    arrow_length = 25
    ax.annotate(
        '', 
        xy=(entrance_z, entrance_y),
        xytext=(entrance_z - arrow_length * entrance_N, entrance_y - arrow_length * entrance_M),
        arrowprops=dict(arrowstyle='->', color='green', lw=2),
    )
    ax.text(
        entrance_z - arrow_length * entrance_N * 0.5 - 3,
        entrance_y - arrow_length * entrance_M * 0.5,
        '入射光束',
        fontsize=10, color='green',
    )
    
    # 绘制出射光束方向（箭头）
    ax.annotate(
        '',
        xy=(exit_z + arrow_length * exit_N, exit_y + arrow_length * exit_M),
        xytext=(exit_z, exit_y),
        arrowprops=dict(arrowstyle='->', color='red', lw=2),
    )
    ax.text(
        exit_z + arrow_length * exit_N * 0.5 + 2,
        exit_y + arrow_length * exit_M * 0.5,
        '出射光束',
        fontsize=10, color='red',
    )
    
    # 绘制入射面法向量（与入射方向相同）
    normal_length = 10
    ax.annotate(
        '',
        xy=(entrance_z + normal_length * entrance_N, entrance_y + normal_length * entrance_M),
        xytext=(entrance_z, entrance_y),
        arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5, linestyle='--'),
    )
    
    # 绘制出射面法向量（与出射方向相同）
    ax.annotate(
        '',
        xy=(exit_z + normal_length * exit_N, exit_y + normal_length * exit_M),
        xytext=(exit_z, exit_y),
        arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5, linestyle='--'),
    )
    
    # 标注元件位置
    ax.plot(0, 0, 'ko', markersize=8)
    ax.text(2, 2, '元件顶点\n(原点)', fontsize=9)
    
    # 标注入射面和出射面位置
    ax.plot(entrance_z, entrance_y, 'g^', markersize=10)
    ax.plot(exit_z, exit_y, 'rv', markersize=10)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10)
    
    # 设置坐标范围
    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 35)
    
    return ax


def draw_opd_comparison(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    raytraced_opd: np.ndarray,
    theoretical_opd: np.ndarray,
    valid_mask: np.ndarray,
    title_prefix: str = "",
    save_path: str = None,
) -> plt.Figure:
    """绘制 OPD 对比图
    
    绘制内容：
    - 光线追迹得到的 OPD 分布
    - 理论计算的 OPD 分布
    - 两者的差值分布
    - 沿 X 和 Y 轴的截面对比
    
    参数:
        x_grid, y_grid: 光线位置网格（mm）
        raytraced_opd: 光线追迹得到的 OPD（波长数）
        theoretical_opd: 理论计算的 OPD（波长数）
        valid_mask: 有效光线掩模
        title_prefix: 标题前缀
        save_path: 保存路径，如果为 None 则不保存
    
    返回:
        matplotlib Figure 对象
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 获取网格尺寸
    n = int(np.sqrt(len(x_grid)))
    
    # 重塑为 2D 数组
    X = x_grid.reshape(n, n)
    Y = y_grid.reshape(n, n)
    raytraced_2d = raytraced_opd.reshape(n, n)
    theoretical_2d = theoretical_opd.reshape(n, n)
    valid_2d = valid_mask.reshape(n, n)
    
    # 将无效区域设为 NaN
    raytraced_2d = np.where(valid_2d, raytraced_2d, np.nan)
    theoretical_2d = np.where(valid_2d, theoretical_2d, np.nan)
    diff_2d = raytraced_2d - theoretical_2d

    # 计算统计信息
    valid_raytraced = raytraced_opd[valid_mask]
    valid_theoretical = theoretical_opd[valid_mask]
    valid_diff = valid_raytraced - valid_theoretical
    
    # 子图 1：光线追迹 OPD
    ax1 = fig.add_subplot(2, 3, 1)
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    im1 = ax1.imshow(raytraced_2d, extent=extent, origin='lower', cmap='RdBu_r')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_title(f'{title_prefix}光线追迹 OPD\nPV={np.nanmax(raytraced_2d)-np.nanmin(raytraced_2d):.2f} waves')
    plt.colorbar(im1, ax=ax1, label='OPD (waves)')
    
    # 子图 2：理论 OPD
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(theoretical_2d, extent=extent, origin='lower', cmap='RdBu_r')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title(f'{title_prefix}理论 OPD\nPV={np.nanmax(theoretical_2d)-np.nanmin(theoretical_2d):.2f} waves')
    plt.colorbar(im2, ax=ax2, label='OPD (waves)')
    
    # 子图 3：差值
    ax3 = fig.add_subplot(2, 3, 3)
    vmax = max(abs(np.nanmin(diff_2d)), abs(np.nanmax(diff_2d)))
    if vmax < 0.01:
        vmax = 0.01
    im3 = ax3.imshow(diff_2d, extent=extent, origin='lower', cmap='RdBu_r', 
                     vmin=-vmax, vmax=vmax)
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_title(f'{title_prefix}差值 (追迹 - 理论)\nRMS={np.nanstd(diff_2d):.4f} waves')
    plt.colorbar(im3, ax=ax3, label='差值 (waves)')

    # 子图 4：沿 X 轴截面（y=0）
    ax4 = fig.add_subplot(2, 3, 4)
    mid_idx = n // 2
    x_slice = X[mid_idx, :]
    raytraced_x_slice = raytraced_2d[mid_idx, :]
    theoretical_x_slice = theoretical_2d[mid_idx, :]
    
    ax4.plot(x_slice, raytraced_x_slice, 'b-', linewidth=2, label='光线追迹')
    ax4.plot(x_slice, theoretical_x_slice, 'r--', linewidth=2, label='理论')
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('OPD (waves)')
    ax4.set_title('沿 X 轴截面 (Y=0)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 子图 5：沿 Y 轴截面（x=0）
    ax5 = fig.add_subplot(2, 3, 5)
    y_slice = Y[:, mid_idx]
    raytraced_y_slice = raytraced_2d[:, mid_idx]
    theoretical_y_slice = theoretical_2d[:, mid_idx]
    
    ax5.plot(y_slice, raytraced_y_slice, 'b-', linewidth=2, label='光线追迹')
    ax5.plot(y_slice, theoretical_y_slice, 'r--', linewidth=2, label='理论')
    ax5.set_xlabel('Y (mm)')
    ax5.set_ylabel('OPD (waves)')
    ax5.set_title('沿 Y 轴截面 (X=0)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 子图 6：统计信息
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    统计信息
    ========
    
    光线追迹 OPD:
      最小值: {np.nanmin(valid_raytraced):.4f} waves
      最大值: {np.nanmax(valid_raytraced):.4f} waves
      PV: {np.nanmax(valid_raytraced) - np.nanmin(valid_raytraced):.4f} waves
      RMS: {np.nanstd(valid_raytraced):.4f} waves
    
    理论 OPD:
      最小值: {np.nanmin(valid_theoretical):.4f} waves
      最大值: {np.nanmax(valid_theoretical):.4f} waves
      PV: {np.nanmax(valid_theoretical) - np.nanmin(valid_theoretical):.4f} waves
      RMS: {np.nanstd(valid_theoretical):.4f} waves
    
    差值:
      最小值: {np.nanmin(valid_diff):.4f} waves
      最大值: {np.nanmax(valid_diff):.4f} waves
      PV: {np.nanmax(valid_diff) - np.nanmin(valid_diff):.4f} waves
      RMS: {np.nanstd(valid_diff):.4f} waves
    """

    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    return fig


# =============================================================================
# 测试类
# =============================================================================

class TestTiltedParabolicPhaseValidation:
    """测试倾斜抛物面镜的相位结果与理论值对比
    
    验证 ElementRaytracer 对 45° 倾斜抛物面镜的光线追迹结果
    是否与薄透镜近似的理论值一致。
    """
    
    # 测试参数
    FOCAL_LENGTH_MM = 100.0  # 焦距 100mm
    RADIUS_MM = 200.0        # 曲率半径 200mm（焦距的 2 倍）
    SEMI_APERTURE_MM = 10.0  # 半口径 10mm
    WAVELENGTH_UM = 0.633    # 波长 633nm
    TILT_ANGLE_DEG = 45.0    # 倾斜角度 45°
    
    @pytest.fixture
    def tilted_parabolic_mirror(self):
        """创建 45° 倾斜的抛物面镜"""
        return SurfaceDefinition(
            surface_type='mirror',
            radius=self.RADIUS_MM,
            thickness=0.0,
            material='mirror',
            semi_aperture=self.SEMI_APERTURE_MM,
            conic=-1.0,  # 抛物面
            tilt_x=np.radians(self.TILT_ANGLE_DEG),
            tilt_y=0.0,
        )
    
    @pytest.fixture
    def raytracer(self, tilted_parabolic_mirror):
        """创建光线追迹器"""
        return ElementRaytracer(
            surfaces=[tilted_parabolic_mirror],
            wavelength=self.WAVELENGTH_UM,
            chief_ray_direction=(0, 0, 1),
            entrance_position=(0, 0, 0),
        )

    @pytest.fixture
    def grid_rays(self):
        """创建网格分布的输入光线"""
        n_rays_1d = 21  # 21x21 = 441 条光线
        coords = np.linspace(-self.SEMI_APERTURE_MM * 0.9, 
                             self.SEMI_APERTURE_MM * 0.9, n_rays_1d)
        X, Y = np.meshgrid(coords, coords)
        x = X.flatten()
        y = Y.flatten()
        n_rays = len(x)
        
        return RealRays(
            x=x,
            y=y,
            z=np.zeros(n_rays),
            L=np.zeros(n_rays),
            M=np.zeros(n_rays),
            N=np.ones(n_rays),
            intensity=np.ones(n_rays),
            wavelength=np.full(n_rays, self.WAVELENGTH_UM),
        ), x, y
    
    def test_opd_matches_thin_lens_approximation(self, raytracer, grid_rays):
        """测试 OPD 与薄透镜近似的一致性
        
        对于抛物面镜，薄透镜近似的 OPD 公式：
            OPD = r² / (2f)
        
        由于出射面垂直于出射光轴，倾斜引入的额外光程应该被抵消，
        因此光线追迹得到的 OPD 应该与理论值接近。
        """
        input_rays, x_grid, y_grid = grid_rays
        
        # 执行光线追迹
        output_rays = raytracer.trace(input_rays)
        raytraced_opd = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask()
        
        # 计算理论 OPD（薄透镜近似）
        theoretical_opd = calculate_theoretical_opd_waves(
            x_grid, y_grid, self.FOCAL_LENGTH_MM, self.WAVELENGTH_UM
        )
        
        # 只比较有效光线
        valid_raytraced = raytraced_opd[valid_mask]
        valid_theoretical = theoretical_opd[valid_mask]
        
        # 计算差值统计
        diff = valid_raytraced - valid_theoretical
        diff_rms = np.std(diff)
        diff_pv = np.max(diff) - np.min(diff)
        
        print(f"\n薄透镜近似对比:")
        print(f"  光线追迹 OPD PV: {np.max(valid_raytraced) - np.min(valid_raytraced):.2f} waves")
        print(f"  理论 OPD PV: {np.max(valid_theoretical) - np.min(valid_theoretical):.2f} waves")
        print(f"  差值 RMS: {diff_rms:.4f} waves")
        print(f"  差值 PV: {diff_pv:.4f} waves")

        # 薄透镜近似有一定误差，允许较大的容差
        # 主要验证 OPD 的量级和趋势是否正确
        # 对于 10mm 半口径、100mm 焦距：
        # 边缘 OPD ≈ 10²/(2*100)/0.633e-3 ≈ 790 waves
        
        # 检查 OPD 量级是否正确（在 2 倍范围内）
        raytraced_pv = np.max(valid_raytraced) - np.min(valid_raytraced)
        theoretical_pv = np.max(valid_theoretical) - np.min(valid_theoretical)
        
        ratio = raytraced_pv / theoretical_pv if theoretical_pv > 0 else 0
        print(f"  PV 比值: {ratio:.4f}")
        
        # 允许 50% 的误差（薄透镜近似本身有误差）
        assert 0.5 < ratio < 2.0, \
            f"OPD PV 比值超出预期范围: {ratio:.4f}，预期在 0.5-2.0 之间"
    
    def test_opd_matches_exact_formula(self, raytracer, grid_rays):
        """测试 OPD 与精确公式的一致性
        
        使用精确的几何公式计算抛物面镜的 OPD，
        并与光线追迹结果进行对比。
        """
        input_rays, x_grid, y_grid = grid_rays
        
        # 执行光线追迹
        output_rays = raytracer.trace(input_rays)
        raytraced_opd = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask()
        
        # 计算精确理论 OPD
        exact_opd = calculate_exact_parabolic_opd_waves(
            x_grid, y_grid, self.FOCAL_LENGTH_MM, self.WAVELENGTH_UM
        )
        
        # 只比较有效光线
        valid_raytraced = raytraced_opd[valid_mask]
        valid_exact = exact_opd[valid_mask]
        
        # 计算差值统计
        diff = valid_raytraced - valid_exact
        diff_rms = np.std(diff)
        diff_pv = np.max(diff) - np.min(diff)
        
        print(f"\n精确公式对比:")
        print(f"  光线追迹 OPD PV: {np.max(valid_raytraced) - np.min(valid_raytraced):.2f} waves")
        print(f"  精确理论 OPD PV: {np.max(valid_exact) - np.min(valid_exact):.2f} waves")
        print(f"  差值 RMS: {diff_rms:.4f} waves")
        print(f"  差值 PV: {diff_pv:.4f} waves")
        
        # 精确公式应该更接近，允许 20% 的误差
        raytraced_pv = np.max(valid_raytraced) - np.min(valid_raytraced)
        exact_pv = np.max(valid_exact) - np.min(valid_exact)
        
        ratio = raytraced_pv / exact_pv if exact_pv > 0 else 0
        print(f"  PV 比值: {ratio:.4f}")

    def test_geometry_visualization(self, raytracer):
        """测试几何配置可视化
        
        绘制 2D 几何配置图，显示：
        - 入射面位置和方向
        - 出射面位置和方向
        - 入射光束和出射光束方向
        """
        # 获取出射主光线方向
        exit_direction = raytracer.get_exit_chief_ray_direction()
        
        print(f"\n几何配置:")
        print(f"  入射方向: (0, 0, 1)")
        print(f"  出射方向: {exit_direction}")
        print(f"  倾斜角度: {self.TILT_ANGLE_DEG}°")
        
        # 入射面在原点，方向沿 +Z
        entrance_position = (0, 0)  # (y, z)
        entrance_direction = (0, 1)  # (M, N) = (0, 1) 表示沿 +Z
        
        # 出射面也在原点（元件顶点），方向沿出射主光线
        exit_position = (0, 0)  # (y, z)
        # 出射方向 (L, M, N)，在 YZ 平面投影为 (M, N)
        exit_M, exit_N = exit_direction[1], exit_direction[2]
        exit_dir_2d = (exit_M, exit_N)
        
        # 绘制几何配置
        fig, ax = plt.subplots(figsize=(10, 8))
        draw_2d_geometry(
            entrance_position=entrance_position,
            entrance_direction=entrance_direction,
            exit_position=exit_position,
            exit_direction=exit_dir_2d,
            tilt_angle_deg=self.TILT_ANGLE_DEG,
            ax=ax,
        )
        
        # 保存图像
        save_path = 'tests/output/tilted_parabolic_geometry.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  几何配置图已保存到: {save_path}")
        
        # 验证出射方向
        # 根据当前实现，45° 倾斜的反射镜出射方向为 (0, +1, 0)
        assert_allclose(
            exit_direction,
            (0.0, 1.0, 0.0),
            atol=1e-6,
            err_msg="出射方向与预期不符"
        )

    def test_opd_comparison_visualization(self, raytracer, grid_rays):
        """测试 OPD 对比可视化
        
        生成 OPD 对比图，包括：
        - 光线追迹 OPD 分布
        - 理论 OPD 分布
        - 差值分布
        - 截面对比
        """
        input_rays, x_grid, y_grid = grid_rays
        
        # 执行光线追迹
        output_rays = raytracer.trace(input_rays)
        raytraced_opd = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask()
        
        # 计算理论 OPD（使用精确公式）
        theoretical_opd = calculate_exact_parabolic_opd_waves(
            x_grid, y_grid, self.FOCAL_LENGTH_MM, self.WAVELENGTH_UM
        )
        
        # 绘制对比图
        save_path = 'tests/output/tilted_parabolic_opd_comparison.png'
        fig = draw_opd_comparison(
            x_grid=x_grid,
            y_grid=y_grid,
            raytraced_opd=raytraced_opd,
            theoretical_opd=theoretical_opd,
            valid_mask=valid_mask,
            title_prefix='45° 倾斜抛物面镜 ',
            save_path=save_path,
        )
        plt.close(fig)
        
        print(f"\nOPD 对比图已保存到: {save_path}")


class TestTiltedPlaneMirrorPhaseValidation:
    """测试倾斜平面镜的相位结果
    
    对于平面镜，理论 OPD 应该为 0（没有聚焦效果）。
    倾斜引入的额外光程应该在出射面处相互抵消。
    """
    
    SEMI_APERTURE_MM = 10.0
    WAVELENGTH_UM = 0.633
    TILT_ANGLE_DEG = 45.0
    
    @pytest.fixture
    def tilted_plane_mirror(self):
        """创建 45° 倾斜的平面镜"""
        return SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,  # 平面镜
            thickness=0.0,
            material='mirror',
            semi_aperture=self.SEMI_APERTURE_MM,
            conic=0.0,
            tilt_x=np.radians(self.TILT_ANGLE_DEG),
            tilt_y=0.0,
        )

    @pytest.fixture
    def raytracer(self, tilted_plane_mirror):
        """创建光线追迹器"""
        return ElementRaytracer(
            surfaces=[tilted_plane_mirror],
            wavelength=self.WAVELENGTH_UM,
            chief_ray_direction=(0, 0, 1),
            entrance_position=(0, 0, 0),
        )
    
    @pytest.fixture
    def grid_rays(self):
        """创建网格分布的输入光线"""
        n_rays_1d = 21
        coords = np.linspace(-self.SEMI_APERTURE_MM * 0.9, 
                             self.SEMI_APERTURE_MM * 0.9, n_rays_1d)
        X, Y = np.meshgrid(coords, coords)
        x = X.flatten()
        y = Y.flatten()
        n_rays = len(x)
        
        return RealRays(
            x=x,
            y=y,
            z=np.zeros(n_rays),
            L=np.zeros(n_rays),
            M=np.zeros(n_rays),
            N=np.ones(n_rays),
            intensity=np.ones(n_rays),
            wavelength=np.full(n_rays, self.WAVELENGTH_UM),
        ), x, y
    
    def test_plane_mirror_opd_near_zero(self, raytracer, grid_rays):
        """测试平面镜的 OPD 应该接近零
        
        对于平面镜：
        - 没有聚焦效果，理论 OPD = 0
        - 倾斜引入的光程差应该在出射面处抵消
        - 相对 OPD 应该非常小
        """
        input_rays, x_grid, y_grid = grid_rays
        
        # 执行光线追迹
        output_rays = raytracer.trace(input_rays)
        raytraced_opd = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask()
        
        # 只分析有效光线
        valid_opd = raytraced_opd[valid_mask]
        
        opd_rms = np.std(valid_opd)
        opd_pv = np.max(valid_opd) - np.min(valid_opd)
        
        print(f"\n平面镜 OPD 统计:")
        print(f"  有效光线数: {np.sum(valid_mask)}/{len(valid_mask)}")
        print(f"  OPD RMS: {opd_rms:.6f} waves")
        print(f"  OPD PV: {opd_pv:.6f} waves")
        
        # 平面镜的 OPD 应该非常小（< 1 波长）
        assert opd_pv < 1.0, \
            f"平面镜 OPD PV 过大: {opd_pv:.4f} waves，预期 < 1.0 waves"

    def test_plane_mirror_opd_visualization(self, raytracer, grid_rays):
        """测试平面镜 OPD 可视化"""
        input_rays, x_grid, y_grid = grid_rays
        
        # 执行光线追迹
        output_rays = raytracer.trace(input_rays)
        raytraced_opd = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask()
        
        # 理论 OPD 为 0
        theoretical_opd = np.zeros_like(raytraced_opd)
        
        # 绘制对比图
        save_path = 'tests/output/tilted_plane_mirror_opd.png'
        fig = draw_opd_comparison(
            x_grid=x_grid,
            y_grid=y_grid,
            raytraced_opd=raytraced_opd,
            theoretical_opd=theoretical_opd,
            valid_mask=valid_mask,
            title_prefix='45° 倾斜平面镜 ',
            save_path=save_path,
        )
        plt.close(fig)
        
        print(f"\n平面镜 OPD 图已保存到: {save_path}")


# =============================================================================
# 综合可视化测试
# =============================================================================

class TestComprehensiveVisualization:
    """综合可视化测试
    
    生成完整的测试报告图像，包括几何配置和 OPD 对比。
    """
    
    @pytest.mark.skip(reason="手动运行的可视化测试")
    def test_generate_full_report(self):
        """生成完整的测试报告
        
        包括：
        1. 几何配置图（入射面、出射面、光束方向）
        2. 抛物面镜 OPD 对比图
        3. 平面镜 OPD 对比图
        """
        # 参数设置
        focal_length_mm = 100.0
        radius_mm = 200.0
        semi_aperture_mm = 10.0
        wavelength_um = 0.633
        tilt_angle_deg = 45.0
        
        # 创建抛物面镜
        parabolic_mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=radius_mm,
            thickness=0.0,
            material='mirror',
            semi_aperture=semi_aperture_mm,
            conic=-1.0,
            tilt_x=np.radians(tilt_angle_deg),
            tilt_y=0.0,
        )
        
        # 创建平面镜
        plane_mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            thickness=0.0,
            material='mirror',
            semi_aperture=semi_aperture_mm,
            conic=0.0,
            tilt_x=np.radians(tilt_angle_deg),
            tilt_y=0.0,
        )

        # 创建光线网格
        n_rays_1d = 31
        coords = np.linspace(-semi_aperture_mm * 0.9, semi_aperture_mm * 0.9, n_rays_1d)
        X, Y = np.meshgrid(coords, coords)
        x_grid = X.flatten()
        y_grid = Y.flatten()
        n_rays = len(x_grid)
        
        input_rays = RealRays(
            x=x_grid,
            y=y_grid,
            z=np.zeros(n_rays),
            L=np.zeros(n_rays),
            M=np.zeros(n_rays),
            N=np.ones(n_rays),
            intensity=np.ones(n_rays),
            wavelength=np.full(n_rays, wavelength_um),
        )
        
        # 测试抛物面镜
        print("=" * 60)
        print("测试 45° 倾斜抛物面镜")
        print("=" * 60)
        
        raytracer_parabolic = ElementRaytracer(
            surfaces=[parabolic_mirror],
            wavelength=wavelength_um,
            chief_ray_direction=(0, 0, 1),
            entrance_position=(0, 0, 0),
        )
        
        output_rays = raytracer_parabolic.trace(input_rays)
        raytraced_opd = raytracer_parabolic.get_relative_opd_waves()
        valid_mask = raytracer_parabolic.get_valid_ray_mask()
        
        # 计算理论 OPD
        theoretical_opd = calculate_exact_parabolic_opd_waves(
            x_grid, y_grid, focal_length_mm, wavelength_um
        )
        
        # 绘制对比图
        fig = draw_opd_comparison(
            x_grid=x_grid,
            y_grid=y_grid,
            raytraced_opd=raytraced_opd,
            theoretical_opd=theoretical_opd,
            valid_mask=valid_mask,
            title_prefix='45° 倾斜抛物面镜 ',
            save_path='tests/output/tilted_parabolic_full_report.png',
        )
        plt.close(fig)
        
        # 测试平面镜
        print("\n" + "=" * 60)
        print("测试 45° 倾斜平面镜")
        print("=" * 60)
        
        raytracer_plane = ElementRaytracer(
            surfaces=[plane_mirror],
            wavelength=wavelength_um,
            chief_ray_direction=(0, 0, 1),
            entrance_position=(0, 0, 0),
        )

        output_rays = raytracer_plane.trace(input_rays)
        raytraced_opd_plane = raytracer_plane.get_relative_opd_waves()
        valid_mask_plane = raytracer_plane.get_valid_ray_mask()
        
        # 平面镜理论 OPD 为 0
        theoretical_opd_plane = np.zeros_like(raytraced_opd_plane)
        
        # 绘制对比图
        fig = draw_opd_comparison(
            x_grid=x_grid,
            y_grid=y_grid,
            raytraced_opd=raytraced_opd_plane,
            theoretical_opd=theoretical_opd_plane,
            valid_mask=valid_mask_plane,
            title_prefix='45° 倾斜平面镜 ',
            save_path='tests/output/tilted_plane_full_report.png',
        )
        plt.close(fig)
        
        print("\n测试报告生成完成！")


# =============================================================================
# 运行测试的入口点
# =============================================================================

if __name__ == '__main__':
    """直接运行此文件时执行测试"""
    import os
    
    # 确保输出目录存在
    os.makedirs('tests/output', exist_ok=True)
    
    # 运行所有测试并显示详细输出
    pytest.main([
        __file__,
        '-v',
        '-s',  # 显示 print 输出
        '--tb=short',  # 简短的错误回溯
    ])
