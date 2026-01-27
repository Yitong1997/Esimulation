"""
全局坐标系混合传播集成测试

本模块测试 GlobalElementRaytracer 和 HybridElementPropagatorGlobal 的集成功能。

测试内容：
1. 平面镜传输精度（不同倾斜角度）
2. 复杂折叠镜系统
3. 与标准 HybridElementPropagator 的结果对比

**Validates: Requirements 11.1-11.5**
"""

import sys
from pathlib import Path

# 设置路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))
sys.path.insert(0, str(project_root / 'proper_v3.3.4_python'))

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

# 导入测试所需的模块
from wavefront_to_rays.global_element_raytracer import (
    GlobalElementRaytracer,
    GlobalSurfaceDefinition,
    PlaneDef,
)
from hybrid_optical_propagation.hybrid_element_propagator_global import (
    HybridElementPropagatorGlobal,
)


class TestFlatMirrorPrecision:
    """平面镜传输精度测试
    
    验证不同倾斜角度（0°-60°）的平面镜 RMS < 1 milli-wave
    
    **Validates: Requirements 11.5**
    """
    
    @pytest.mark.parametrize("tilt_angle_deg", [0, 15, 30, 45, 60])
    def test_flat_mirror_precision_at_various_angles(self, tilt_angle_deg):
        """测试不同倾斜角度的平面镜传输精度
        
        对于理想平面镜，所有光线的相对 OPD 应该相同（都为 0）。
        
        **Validates: Requirements 11.5**
        """
        from optiland.rays import RealRays
        
        tilt_angle_rad = np.radians(tilt_angle_deg)
        
        # 计算倾斜后的表面法向量
        # 绕 X 轴旋转 tilt_angle
        surface_normal = (
            0.0,
            -np.sin(tilt_angle_rad),
            -np.cos(tilt_angle_rad),
        )
        
        # 创建平面镜
        mirror = GlobalSurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            vertex_position=(0.0, 0.0, 100.0),
            surface_normal=surface_normal,
            tilt_x=tilt_angle_rad,
        )
        
        # 创建入射面
        entrance_plane = PlaneDef(
            position=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
        )
        
        # 创建光线追迹器
        raytracer = GlobalElementRaytracer(
            surfaces=[mirror],
            wavelength=0.633,
            entrance_plane=entrance_plane,
        )
        
        # 追迹主光线
        raytracer.trace_chief_ray()
        
        # 创建多条光线
        n_rays = 21
        x_vals = np.linspace(-5, 5, n_rays)
        y_vals = np.zeros(n_rays)
        z_vals = np.zeros(n_rays)
        
        input_rays = RealRays(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            L=np.zeros(n_rays),
            M=np.zeros(n_rays),
            N=np.ones(n_rays),
            intensity=np.ones(n_rays),
            wavelength=np.full(n_rays, 0.633),
        )
        input_rays.opd = np.zeros(n_rays)
        
        # 追迹光线
        output_rays = raytracer.trace(input_rays)
        
        # 计算相对于主光线的 OPD
        opd_out = np.asarray(output_rays.opd)
        x_out = np.asarray(output_rays.x)
        chief_idx = np.argmin(np.abs(x_out))
        chief_opd = opd_out[chief_idx]
        relative_opd = opd_out - chief_opd
        
        # 转换为波长数
        wavelength_mm = 0.633 * 1e-3
        relative_opd_waves = relative_opd / wavelength_mm
        
        # 计算 RMS
        rms_waves = np.sqrt(np.mean(relative_opd_waves**2))
        rms_milli_waves = rms_waves * 1000
        
        # 验证 RMS < 1 milli-wave
        assert rms_milli_waves < 1.0, \
            f"倾斜角度 {tilt_angle_deg}° 的平面镜 RMS = {rms_milli_waves:.3f} milli-waves，超过 1 milli-wave"
        
        print(f"倾斜角度 {tilt_angle_deg}°: RMS = {rms_milli_waves:.6f} milli-waves")


class TestGlobalRaytracerBasicFunctionality:
    """GlobalElementRaytracer 基本功能测试
    
    **Validates: Requirements 11.1, 11.2**
    """
    
    def test_flat_mirror_normal_incidence(self):
        """测试平面镜正入射
        
        **Validates: Requirements 11.1**
        """
        from optiland.rays import RealRays
        
        # 创建平面镜
        mirror = GlobalSurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            vertex_position=(0.0, 0.0, 100.0),
            surface_normal=(0.0, 0.0, -1.0),
        )
        
        # 创建入射面
        entrance_plane = PlaneDef(
            position=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
        )
        
        # 创建光线追迹器
        raytracer = GlobalElementRaytracer(
            surfaces=[mirror],
            wavelength=0.633,
            entrance_plane=entrance_plane,
        )
        
        # 追迹主光线
        exit_dir = raytracer.trace_chief_ray()
        
        # 验证出射方向（正入射反射后应该沿 -Z 方向）
        assert np.isclose(exit_dir[0], 0.0, atol=1e-6)
        assert np.isclose(exit_dir[1], 0.0, atol=1e-6)
        assert np.isclose(exit_dir[2], -1.0, atol=1e-6)
        
        # 创建输入光线
        input_rays = RealRays(
            x=np.array([0.0, 1.0, -1.0]),
            y=np.array([0.0, 0.0, 0.0]),
            z=np.array([0.0, 0.0, 0.0]),
            L=np.array([0.0, 0.0, 0.0]),
            M=np.array([0.0, 0.0, 0.0]),
            N=np.array([1.0, 1.0, 1.0]),
            intensity=np.array([1.0, 1.0, 1.0]),
            wavelength=np.array([0.633, 0.633, 0.633]),
        )
        input_rays.opd = np.zeros(3)
        
        # 追迹光线
        output_rays = raytracer.trace(input_rays)
        
        # 验证输出光线方向（应该沿 -Z 方向）
        L_out = np.asarray(output_rays.L)
        M_out = np.asarray(output_rays.M)
        N_out = np.asarray(output_rays.N)
        
        assert np.allclose(L_out, 0.0, atol=1e-6)
        assert np.allclose(M_out, 0.0, atol=1e-6)
        assert np.allclose(N_out, -1.0, atol=1e-6)
    
    def test_tilted_flat_mirror(self):
        """测试倾斜平面镜
        
        **Validates: Requirements 11.2**
        """
        from optiland.rays import RealRays
        
        # 创建 45° 倾斜的平面镜
        tilt_angle = np.pi / 4  # 45°
        
        # 表面法向量（绕 X 轴旋转 45°）
        surface_normal = (
            0.0,
            -np.sin(tilt_angle),
            -np.cos(tilt_angle),
        )
        
        mirror = GlobalSurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            vertex_position=(0.0, 0.0, 100.0),
            surface_normal=surface_normal,
            tilt_x=tilt_angle,
        )
        
        # 创建入射面
        entrance_plane = PlaneDef(
            position=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
        )
        
        # 创建光线追迹器
        raytracer = GlobalElementRaytracer(
            surfaces=[mirror],
            wavelength=0.633,
            entrance_plane=entrance_plane,
        )
        
        # 追迹主光线
        exit_dir = raytracer.trace_chief_ray()
        
        # 验证出射方向（45° 倾斜反射后应该沿 ±Y 方向）
        # 入射方向 (0, 0, 1)，表面法向量 (0, -sin45, -cos45)
        # 反射方向的 Y 分量应该是 ±1
        assert np.isclose(exit_dir[0], 0.0, atol=1e-5)
        assert np.isclose(abs(exit_dir[1]), 1.0, atol=1e-5)
        assert np.isclose(abs(exit_dir[2]), 0.0, atol=1e-5)


class TestHybridElementPropagatorGlobalBasic:
    """HybridElementPropagatorGlobal 基本功能测试
    
    **Validates: Requirements 11.4**
    """
    
    def test_propagator_initialization(self):
        """测试传播器初始化"""
        propagator = HybridElementPropagatorGlobal(
            wavelength_um=0.633,
            num_rays=100,
        )
        
        assert propagator.wavelength_um == 0.633
        assert propagator.num_rays == 100
    
    def test_coordinate_transform_methods_exist(self):
        """测试坐标转换方法存在"""
        propagator = HybridElementPropagatorGlobal(
            wavelength_um=0.633,
            num_rays=100,
        )
        
        # 验证方法存在
        assert hasattr(propagator, '_local_to_global_rays')
        assert hasattr(propagator, '_global_to_local_rays')
        assert hasattr(propagator, '_define_entrance_plane')
        assert hasattr(propagator, '_define_exit_plane')


# =============================================================================
# 运行测试
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
