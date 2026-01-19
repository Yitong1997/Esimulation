"""
混合元件传播测试通用 fixtures

本模块定义了 hybrid_propagation 测试所需的通用 fixtures，包括：
- 测试用的复振幅数组（不同大小）
- 测试用的光学元件定义（平面镜、凹面镜、45°折叠镜）
- 测试用的波长和物理尺寸参数
- 测试用的光线数量配置

使用方法：
    在测试函数中直接使用 fixture 名称作为参数即可自动注入。
    
示例：
    def test_something(gaussian_amplitude_64, flat_mirror):
        # gaussian_amplitude_64 和 flat_mirror 会自动注入
        pass
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest
from numpy.typing import NDArray
from typing import Tuple

# 导入项目模块
from wavefront_to_rays.element_raytracer import SurfaceDefinition


# =============================================================================
# 波长和物理尺寸参数 fixtures
# =============================================================================

@pytest.fixture
def wavelength_visible() -> float:
    """可见光波长（μm）
    
    返回:
        波长值，单位 μm（0.633 μm = 633 nm，HeNe 激光）
    """
    return 0.633


@pytest.fixture
def wavelength_infrared() -> float:
    """红外波长（μm）
    
    返回:
        波长值，单位 μm（1.064 μm = 1064 nm，Nd:YAG 激光）
    """
    return 1.064


@pytest.fixture
def physical_size_small() -> float:
    """小尺寸物理直径（mm）
    
    返回:
        物理尺寸，单位 mm
    """
    return 10.0


@pytest.fixture
def physical_size_medium() -> float:
    """中等尺寸物理直径（mm）
    
    返回:
        物理尺寸，单位 mm
    """
    return 25.0


@pytest.fixture
def physical_size_large() -> float:
    """大尺寸物理直径（mm）
    
    返回:
        物理尺寸，单位 mm
    """
    return 50.0


# =============================================================================
# 光线数量配置 fixtures
# =============================================================================

@pytest.fixture
def num_rays_small() -> int:
    """少量光线数量
    
    返回:
        光线数量（用于快速测试）
    """
    return 25


@pytest.fixture
def num_rays_medium() -> int:
    """中等光线数量
    
    返回:
        光线数量（用于标准测试）
    """
    return 100


@pytest.fixture
def num_rays_large() -> int:
    """大量光线数量
    
    返回:
        光线数量（用于精度测试）
    """
    return 400


# =============================================================================
# 复振幅数组 fixtures
# =============================================================================

def _create_gaussian_amplitude(
    grid_size: int,
    physical_size: float = 10.0,
    beam_radius: float = 2.0,
    wavelength: float = 0.633,
    curvature_radius: float = None,
) -> NDArray:
    """创建高斯光束复振幅数组
    
    参数:
        grid_size: 网格大小
        physical_size: 物理尺寸（直径），单位 mm
        beam_radius: 光束半径（1/e² 强度），单位 mm
        wavelength: 波长，单位 μm
        curvature_radius: 波前曲率半径，单位 mm（None 表示平面波）
    
    返回:
        复振幅数组，形状 (grid_size, grid_size)
    """
    # 创建坐标网格
    half_size = physical_size / 2.0
    x = np.linspace(-half_size, half_size, grid_size)
    y = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(x, y)
    R_sq = X**2 + Y**2
    
    # 高斯振幅
    amplitude = np.exp(-R_sq / beam_radius**2)
    
    # 相位（球面波或平面波）
    if curvature_radius is not None and np.isfinite(curvature_radius):
        # 球面波相位：φ = -k * r² / (2R)
        wavelength_mm = wavelength * 1e-3  # μm -> mm
        k = 2 * np.pi / wavelength_mm
        phase = -k * R_sq / (2 * curvature_radius)
    else:
        # 平面波
        phase = np.zeros_like(R_sq)
    
    return amplitude * np.exp(1j * phase)


@pytest.fixture
def gaussian_amplitude_32() -> NDArray:
    """32x32 高斯光束复振幅
    
    返回:
        复振幅数组，形状 (32, 32)
    """
    return _create_gaussian_amplitude(grid_size=32)


@pytest.fixture
def gaussian_amplitude_64() -> NDArray:
    """64x64 高斯光束复振幅
    
    返回:
        复振幅数组，形状 (64, 64)
    """
    return _create_gaussian_amplitude(grid_size=64)


@pytest.fixture
def gaussian_amplitude_128() -> NDArray:
    """128x128 高斯光束复振幅
    
    返回:
        复振幅数组，形状 (128, 128)
    """
    return _create_gaussian_amplitude(grid_size=128)


@pytest.fixture
def gaussian_amplitude_256() -> NDArray:
    """256x256 高斯光束复振幅
    
    返回:
        复振幅数组，形状 (256, 256)
    """
    return _create_gaussian_amplitude(grid_size=256)


@pytest.fixture
def gaussian_amplitude_512() -> NDArray:
    """512x512 高斯光束复振幅
    
    返回:
        复振幅数组，形状 (512, 512)
    """
    return _create_gaussian_amplitude(grid_size=512)


@pytest.fixture
def spherical_wave_amplitude_64() -> NDArray:
    """64x64 球面波复振幅（发散波）
    
    返回:
        复振幅数组，形状 (64, 64)，曲率半径 100mm
    """
    return _create_gaussian_amplitude(
        grid_size=64,
        curvature_radius=100.0,  # 发散波
    )


@pytest.fixture
def converging_wave_amplitude_64() -> NDArray:
    """64x64 会聚波复振幅
    
    返回:
        复振幅数组，形状 (64, 64)，曲率半径 -100mm
    """
    return _create_gaussian_amplitude(
        grid_size=64,
        curvature_radius=-100.0,  # 会聚波
    )


@pytest.fixture
def uniform_amplitude_64() -> NDArray:
    """64x64 均匀振幅复振幅（平面波）
    
    返回:
        复振幅数组，形状 (64, 64)，振幅为 1
    """
    return np.ones((64, 64), dtype=np.complex128)


@pytest.fixture(params=[32, 64, 128, 256])
def gaussian_amplitude_parametrized(request) -> Tuple[NDArray, int]:
    """参数化的高斯光束复振幅
    
    参数化不同网格大小：32, 64, 128, 256
    
    返回:
        (复振幅数组, 网格大小) 元组
    """
    grid_size = request.param
    amplitude = _create_gaussian_amplitude(grid_size=grid_size)
    return amplitude, grid_size


# =============================================================================
# 光学元件定义 fixtures
# =============================================================================

@pytest.fixture
def flat_mirror() -> SurfaceDefinition:
    """平面反射镜
    
    返回:
        平面镜表面定义，半口径 15mm
    """
    return SurfaceDefinition(
        surface_type='mirror',
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        semi_aperture=15.0,
    )


@pytest.fixture
def concave_mirror_f100() -> SurfaceDefinition:
    """凹面反射镜（焦距 100mm）
    
    返回:
        凹面镜表面定义，曲率半径 200mm，半口径 15mm
    """
    return SurfaceDefinition(
        surface_type='mirror',
        radius=200.0,  # 焦距 = R/2 = 100mm
        thickness=0.0,
        material='mirror',
        semi_aperture=15.0,
    )


@pytest.fixture
def concave_mirror_f50() -> SurfaceDefinition:
    """凹面反射镜（焦距 50mm）
    
    返回:
        凹面镜表面定义，曲率半径 100mm，半口径 10mm
    """
    return SurfaceDefinition(
        surface_type='mirror',
        radius=100.0,  # 焦距 = R/2 = 50mm
        thickness=0.0,
        material='mirror',
        semi_aperture=10.0,
    )


@pytest.fixture
def convex_mirror_f100() -> SurfaceDefinition:
    """凸面反射镜（焦距 -100mm）
    
    返回:
        凸面镜表面定义，曲率半径 -200mm，半口径 15mm
    """
    return SurfaceDefinition(
        surface_type='mirror',
        radius=-200.0,  # 焦距 = R/2 = -100mm
        thickness=0.0,
        material='mirror',
        semi_aperture=15.0,
    )


@pytest.fixture
def fold_mirror_45deg() -> SurfaceDefinition:
    """45° 折叠平面镜
    
    返回:
        45° 倾斜的平面镜表面定义，半口径 20mm
    """
    return SurfaceDefinition(
        surface_type='mirror',
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        tilt_x=np.pi / 4,  # 绕 X 轴旋转 45°
        tilt_y=0.0,
    )


@pytest.fixture
def fold_mirror_30deg() -> SurfaceDefinition:
    """30° 折叠平面镜
    
    返回:
        30° 倾斜的平面镜表面定义，半口径 20mm
    """
    return SurfaceDefinition(
        surface_type='mirror',
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        tilt_x=np.pi / 6,  # 绕 X 轴旋转 30°
        tilt_y=0.0,
    )


@pytest.fixture
def tilted_concave_mirror() -> SurfaceDefinition:
    """倾斜凹面镜（15° 倾斜，焦距 100mm）
    
    返回:
        15° 倾斜的凹面镜表面定义
    """
    return SurfaceDefinition(
        surface_type='mirror',
        radius=200.0,
        thickness=0.0,
        material='mirror',
        semi_aperture=15.0,
        tilt_x=np.pi / 12,  # 绕 X 轴旋转 15°
        tilt_y=0.0,
    )


@pytest.fixture
def parabolic_mirror_f100() -> SurfaceDefinition:
    """抛物面反射镜（焦距 100mm）
    
    返回:
        抛物面镜表面定义，顶点曲率半径 200mm，半口径 15mm
    """
    return SurfaceDefinition(
        surface_type='mirror',
        radius=200.0,  # 顶点曲率半径
        thickness=0.0,
        material='mirror',
        semi_aperture=15.0,
        conic=-1.0,  # 抛物面
    )


@pytest.fixture(params=['flat', 'concave', 'convex', 'fold_45'])
def mirror_parametrized(request) -> Tuple[SurfaceDefinition, str]:
    """参数化的反射镜
    
    参数化不同类型：平面镜、凹面镜、凸面镜、45°折叠镜
    
    返回:
        (表面定义, 类型名称) 元组
    """
    mirror_type = request.param
    
    if mirror_type == 'flat':
        mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            semi_aperture=15.0,
        )
    elif mirror_type == 'concave':
        mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=200.0,
            semi_aperture=15.0,
        )
    elif mirror_type == 'convex':
        mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=-200.0,
            semi_aperture=15.0,
        )
    elif mirror_type == 'fold_45':
        mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            semi_aperture=20.0,
            tilt_x=np.pi / 4,
        )
    else:
        raise ValueError(f"未知的镜面类型: {mirror_type}")
    
    return mirror, mirror_type


# =============================================================================
# 坐标和方向 fixtures
# =============================================================================

@pytest.fixture
def normal_incidence_direction() -> Tuple[float, float, float]:
    """正入射方向（沿 +Z 轴）
    
    返回:
        方向余弦 (L, M, N) = (0, 0, 1)
    """
    return (0.0, 0.0, 1.0)


@pytest.fixture
def tilted_incidence_45deg() -> Tuple[float, float, float]:
    """45° 倾斜入射方向（在 YZ 平面内）
    
    返回:
        方向余弦 (L, M, N)
    """
    angle = np.pi / 4
    return (0.0, np.sin(angle), np.cos(angle))


@pytest.fixture
def tilted_incidence_30deg() -> Tuple[float, float, float]:
    """30° 倾斜入射方向（在 YZ 平面内）
    
    返回:
        方向余弦 (L, M, N)
    """
    angle = np.pi / 6
    return (0.0, np.sin(angle), np.cos(angle))


@pytest.fixture
def origin_position() -> Tuple[float, float, float]:
    """原点位置
    
    返回:
        位置 (x, y, z) = (0, 0, 0)
    """
    return (0.0, 0.0, 0.0)


@pytest.fixture
def offset_position() -> Tuple[float, float, float]:
    """偏移位置
    
    返回:
        位置 (x, y, z) = (0, 0, 100)
    """
    return (0.0, 0.0, 100.0)


# =============================================================================
# 相位网格 fixtures
# =============================================================================

@pytest.fixture
def flat_phase_64() -> NDArray:
    """64x64 平坦相位网格
    
    返回:
        相位数组，形状 (64, 64)，值为 0
    """
    return np.zeros((64, 64), dtype=np.float64)


@pytest.fixture
def linear_phase_64() -> NDArray:
    """64x64 线性相位网格（倾斜波前）
    
    返回:
        相位数组，形状 (64, 64)，沿 X 方向线性变化
    """
    n = 64
    x = np.linspace(-np.pi, np.pi, n)
    y = np.linspace(-np.pi, np.pi, n)
    X, Y = np.meshgrid(x, y)
    return 0.5 * X  # 沿 X 方向的线性相位


@pytest.fixture
def quadratic_phase_64() -> NDArray:
    """64x64 二次相位网格（球面波前）
    
    返回:
        相位数组，形状 (64, 64)，二次变化
    """
    n = 64
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)
    R_sq = X**2 + Y**2
    return np.pi * R_sq  # 二次相位


@pytest.fixture
def random_phase_64() -> NDArray:
    """64x64 随机相位网格
    
    返回:
        相位数组，形状 (64, 64)，随机值在 [-π, π] 范围内
    """
    np.random.seed(42)  # 固定随机种子以保证可重复性
    return np.random.uniform(-np.pi, np.pi, (64, 64))


# =============================================================================
# 辅助函数
# =============================================================================

@pytest.fixture
def compute_total_energy():
    """计算复振幅总能量的辅助函数
    
    返回:
        计算能量的函数
    """
    def _compute(amplitude: NDArray) -> float:
        """计算复振幅的总能量（振幅平方和）"""
        return np.sum(np.abs(amplitude)**2)
    
    return _compute


@pytest.fixture
def compute_energy_ratio():
    """计算能量比的辅助函数
    
    返回:
        计算能量比的函数
    """
    def _compute(amplitude_out: NDArray, amplitude_in: NDArray) -> float:
        """计算输出与输入的能量比"""
        energy_in = np.sum(np.abs(amplitude_in)**2)
        energy_out = np.sum(np.abs(amplitude_out)**2)
        if energy_in == 0:
            return 0.0
        return energy_out / energy_in
    
    return _compute


@pytest.fixture
def assert_energy_conservation():
    """断言能量守恒的辅助函数
    
    返回:
        断言函数
    """
    def _assert(
        amplitude_out: NDArray,
        amplitude_in: NDArray,
        rtol: float = 0.01,
    ) -> None:
        """断言输出与输入能量守恒
        
        参数:
            amplitude_out: 输出复振幅
            amplitude_in: 输入复振幅
            rtol: 相对容差，默认 1%
        """
        energy_in = np.sum(np.abs(amplitude_in)**2)
        energy_out = np.sum(np.abs(amplitude_out)**2)
        ratio = energy_out / energy_in if energy_in > 0 else 0.0
        
        assert 1.0 - rtol <= ratio <= 1.0 + rtol, (
            f"能量不守恒：输出/输入能量比 = {ratio:.4f}，"
            f"期望在 [{1.0 - rtol:.4f}, {1.0 + rtol:.4f}] 范围内"
        )
    
    return _assert


# =============================================================================
# Hypothesis 策略 fixtures
# =============================================================================

@pytest.fixture
def tilt_angle_strategy():
    """倾斜角度策略
    
    返回:
        Hypothesis 策略，生成 [-π/4, π/4] 范围内的角度
    """
    from hypothesis import strategies as st
    return st.floats(
        min_value=-np.pi / 4,
        max_value=np.pi / 4,
        allow_nan=False,
        allow_infinity=False,
    )


@pytest.fixture
def wavelength_strategy():
    """波长策略
    
    返回:
        Hypothesis 策略，生成 [0.3, 2.0] μm 范围内的波长
    """
    from hypothesis import strategies as st
    return st.floats(
        min_value=0.3,
        max_value=2.0,
        allow_nan=False,
        allow_infinity=False,
    )


@pytest.fixture
def grid_size_strategy():
    """网格大小策略
    
    返回:
        Hypothesis 策略，生成 2 的幂次网格大小
    """
    from hypothesis import strategies as st
    return st.sampled_from([32, 64, 128, 256, 512])


@pytest.fixture
def physical_size_strategy():
    """物理尺寸策略
    
    返回:
        Hypothesis 策略，生成 [5.0, 100.0] mm 范围内的尺寸
    """
    from hypothesis import strategies as st
    return st.floats(
        min_value=5.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
    )
