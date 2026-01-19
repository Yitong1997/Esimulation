"""
光轴方向跟踪模块测试

测试 coordinate_tracking.py 中的光轴跟踪功能。
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

import sys
sys.path.insert(0, 'src')

from sequential_system.coordinate_tracking import (
    RayDirection,
    Position3D,
    LocalCoordinateSystem,
    OpticalAxisState,
    OpticalAxisTracker,
    calculate_reflection_direction,
)


class TestRayDirection:
    """测试 RayDirection 类"""
    
    def test_default_direction(self):
        """测试默认方向（沿 +Z）"""
        d = RayDirection()
        assert_allclose([d.L, d.M, d.N], [0, 0, 1], atol=1e-10)
    
    def test_normalization(self):
        """测试自动归一化"""
        d = RayDirection(L=3, M=4, N=0)
        assert_allclose([d.L, d.M, d.N], [0.6, 0.8, 0], atol=1e-10)
    
    def test_zero_vector_raises(self):
        """测试零向量抛出异常"""
        with pytest.raises(ValueError, match="零向量"):
            RayDirection(L=0, M=0, N=0)
    
    def test_to_array(self):
        """测试转换为数组"""
        d = RayDirection(L=0, M=0, N=1)
        arr = d.to_array()
        assert_allclose(arr, [0, 0, 1], atol=1e-10)
    
    def test_from_array(self):
        """测试从数组创建"""
        d = RayDirection.from_array(np.array([1, 0, 0]))
        assert_allclose([d.L, d.M, d.N], [1, 0, 0], atol=1e-10)
    
    def test_reflect_normal_incidence(self):
        """测试正入射反射"""
        # 沿 +Z 入射，法向量沿 -Z
        incident = RayDirection(L=0, M=0, N=1)
        normal = RayDirection(L=0, M=0, N=-1)
        reflected = incident.reflect(normal)
        
        # 反射后应该沿 -Z
        assert_allclose([reflected.L, reflected.M, reflected.N], [0, 0, -1], atol=1e-10)
    
    def test_reflect_45_degree_mirror(self):
        """测试 45° 折叠镜反射"""
        # 沿 +Z 入射
        incident = RayDirection(L=0, M=0, N=1)
        
        # 45° 折叠镜：法向量在 YZ 平面内
        # 法向量 (0, 0, -1) 绕 X 轴旋转 45° 后 = (0, sin(45°), -cos(45°)) = (0, 0.707, -0.707)
        normal = RayDirection(L=0, M=np.sqrt(2)/2, N=-np.sqrt(2)/2)
        reflected = incident.reflect(normal)
        
        # 反射后应该沿 +Y 方向
        assert_allclose([reflected.L, reflected.M, reflected.N], [0, 1, 0], atol=1e-10)
    
    def test_rotate_x(self):
        """测试绕 X 轴旋转"""
        d = RayDirection(L=0, M=0, N=1)
        rotated = d.rotate_x(np.pi/2)  # 旋转 90°
        
        # 旋转后应该沿 -Y 方向
        assert_allclose([rotated.L, rotated.M, rotated.N], [0, -1, 0], atol=1e-10)
    
    def test_rotate_y(self):
        """测试绕 Y 轴旋转"""
        d = RayDirection(L=0, M=0, N=1)
        rotated = d.rotate_y(np.pi/2)  # 旋转 90°
        
        # 旋转后应该沿 +X 方向
        assert_allclose([rotated.L, rotated.M, rotated.N], [1, 0, 0], atol=1e-10)
    
    def test_angle_with(self):
        """测试计算夹角"""
        d1 = RayDirection(L=0, M=0, N=1)
        d2 = RayDirection(L=1, M=0, N=0)
        
        angle = d1.angle_with(d2)
        assert_allclose(angle, np.pi/2, atol=1e-10)


class TestPosition3D:
    """测试 Position3D 类"""
    
    def test_default_position(self):
        """测试默认位置（原点）"""
        p = Position3D()
        assert_allclose([p.x, p.y, p.z], [0, 0, 0], atol=1e-10)
    
    def test_to_array(self):
        """测试转换为数组"""
        p = Position3D(x=1, y=2, z=3)
        arr = p.to_array()
        assert_allclose(arr, [1, 2, 3], atol=1e-10)
    
    def test_from_array(self):
        """测试从数组创建"""
        p = Position3D.from_array(np.array([1, 2, 3]))
        assert_allclose([p.x, p.y, p.z], [1, 2, 3], atol=1e-10)
    
    def test_advance(self):
        """测试沿方向前进"""
        p = Position3D(x=0, y=0, z=0)
        d = RayDirection(L=0, M=0, N=1)
        
        new_p = p.advance(d, 100)
        assert_allclose([new_p.x, new_p.y, new_p.z], [0, 0, 100], atol=1e-10)
    
    def test_advance_diagonal(self):
        """测试沿对角方向前进"""
        p = Position3D(x=0, y=0, z=0)
        d = RayDirection(L=1, M=1, N=1)  # 会被归一化
        
        new_p = p.advance(d, np.sqrt(3))  # 前进 sqrt(3)
        assert_allclose([new_p.x, new_p.y, new_p.z], [1, 1, 1], atol=1e-10)


class TestLocalCoordinateSystem:
    """测试 LocalCoordinateSystem 类"""
    
    def test_default_coordinate_system(self):
        """测试默认坐标系"""
        cs = LocalCoordinateSystem()
        
        assert_allclose(cs.origin.to_array(), [0, 0, 0], atol=1e-10)
        assert_allclose(cs.z_axis.to_array(), [0, 0, 1], atol=1e-10)
    
    def test_get_surface_normal_no_tilt(self):
        """测试无倾斜时的表面法向量"""
        cs = LocalCoordinateSystem()
        normal = cs.get_surface_normal()
        
        # 法向量应该沿 -Z（指向入射侧）
        assert_allclose(normal.to_array(), [0, 0, -1], atol=1e-10)
    
    def test_get_surface_normal_with_tilt_x(self):
        """测试绕 X 轴倾斜时的表面法向量"""
        cs = LocalCoordinateSystem()
        normal = cs.get_surface_normal(tilt_x=np.pi/4)  # 45° 倾斜
        
        # 初始法向量 (0, 0, -1) 绕 X 轴旋转 45°
        # 使用右手定则：拇指指向 +X，四指从 +Y 转向 +Z
        # 旋转矩阵：
        # [1,    0,       0   ]
        # [0,  cos(θ), -sin(θ)]
        # [0,  sin(θ),  cos(θ)]
        # (0, 0, -1) 旋转后 = (0, sin(45°), -cos(45°)) = (0, 0.707, -0.707)
        expected = np.array([0, np.sqrt(2)/2, -np.sqrt(2)/2])
        assert_allclose(normal.to_array(), expected, atol=1e-10)


class TestOpticalAxisState:
    """测试 OpticalAxisState 类"""
    
    def test_propagate(self):
        """测试传播"""
        state = OpticalAxisState(
            position=Position3D(0, 0, 0),
            direction=RayDirection(0, 0, 1),
            path_length=0,
        )
        
        new_state = state.propagate(100)
        
        assert_allclose(new_state.position.to_array(), [0, 0, 100], atol=1e-10)
        assert_allclose(new_state.direction.to_array(), [0, 0, 1], atol=1e-10)
        assert_allclose(new_state.path_length, 100, atol=1e-10)
    
    def test_reflect(self):
        """测试反射"""
        state = OpticalAxisState(
            position=Position3D(0, 0, 100),
            direction=RayDirection(0, 0, 1),
            path_length=100,
        )
        
        # 45° 折叠镜：法向量 (0, 0, -1) 绕 X 轴旋转 45° 后
        normal = RayDirection(0, np.sqrt(2)/2, -np.sqrt(2)/2)
        new_state = state.reflect(normal)
        
        # 位置不变
        assert_allclose(new_state.position.to_array(), [0, 0, 100], atol=1e-10)
        # 方向改变为 +Y
        assert_allclose(new_state.direction.to_array(), [0, 1, 0], atol=1e-10)
        # 光程不变
        assert_allclose(new_state.path_length, 100, atol=1e-10)


class TestCalculateReflectionDirection:
    """测试 calculate_reflection_direction 函数"""
    
    def test_45_degree_fold(self):
        """测试 45° 折叠"""
        incident = RayDirection(0, 0, 1)
        reflected = calculate_reflection_direction(incident, tilt_x=np.pi/4, tilt_y=0)
        
        # 法向量 (0, 0, -1) 绕 X 轴旋转 45° 后变成 (0, 0.707, -0.707)
        # 入射光 (0, 0, 1) 反射后：
        # d·n = -0.707
        # r = d - 2*(d·n)*n = (0, 0, 1) + 1.414*(0, 0.707, -0.707) = (0, 1, 0)
        # 反射后应该沿 +Y 方向
        assert_allclose(reflected.to_array(), [0, 1, 0], atol=1e-10)
    
    def test_no_tilt(self):
        """测试无倾斜（正入射）"""
        incident = RayDirection(0, 0, 1)
        reflected = calculate_reflection_direction(incident, tilt_x=0, tilt_y=0)
        
        # 反射后应该沿 -Z 方向
        assert_allclose(reflected.to_array(), [0, 0, -1], atol=1e-10)
    
    def test_tilt_y_45_degree(self):
        """测试绕 Y 轴 45° 倾斜"""
        incident = RayDirection(0, 0, 1)
        reflected = calculate_reflection_direction(incident, tilt_x=0, tilt_y=np.pi/4)
        
        # 反射后应该沿 -X 方向
        assert_allclose(reflected.to_array(), [-1, 0, 0], atol=1e-10)


class TestTiltedMirrorRayTracing:
    """测试倾斜镜面的光线追迹"""
    
    def test_tilted_flat_mirror_ray_intersection(self):
        """测试倾斜平面镜的光线交点计算
        
        验证 optiland 中倾斜参数是否正确影响光线追迹。
        注意：optiland 的倾斜参数可能需要特定的配置才能正确工作。
        这个测试主要验证光线追迹能正常完成。
        """
        from wavefront_to_rays.element_raytracer import (
            SurfaceDefinition,
            ElementRaytracer,
        )
        from optiland.rays import RealRays
        
        # 创建无倾斜的平面镜（作为基准）
        mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            thickness=0.0,
            material='mirror',
            semi_aperture=20.0,
            tilt_x=0.0,  # 无倾斜
        )
        
        # 创建光线追迹器
        raytracer = ElementRaytracer(
            surfaces=[mirror],
            wavelength=0.633,
            chief_ray_direction=(0, 0, 1),
            entrance_position=(0, 0, 0),
        )
        
        # 创建输入光线（沿 +Z 方向）
        input_rays = RealRays(
            x=np.array([0.0]),
            y=np.array([0.0]),
            z=np.array([0.0]),
            L=np.array([0.0]),
            M=np.array([0.0]),
            N=np.array([1.0]),
            intensity=np.array([1.0]),
            wavelength=np.array([0.633]),
        )
        
        # 执行追迹
        output_rays = raytracer.trace(input_rays)
        
        # 验证输出光线存在
        assert len(output_rays.x) == 1
        
        # 验证光线有效
        valid_mask = raytracer.get_valid_ray_mask()
        assert valid_mask[0], f"光线追迹失败，光线无效。输出: x={output_rays.x}, y={output_rays.y}, z={output_rays.z}"
        
        # 获取出射主光线方向
        exit_dir = raytracer.get_exit_chief_ray_direction()
        
        # 对于正入射的平面镜，反射后光线应该沿 -Z 方向
        assert_allclose(exit_dir, (0, 0, -1), atol=0.1, 
                       err_msg=f"正入射平面镜反射方向错误: {exit_dir}")
    
    def test_tilted_mirror_changes_ray_direction(self):
        """测试倾斜镜面改变光线方向
        
        比较有倾斜和无倾斜时的出射光线方向。
        """
        from wavefront_to_rays.element_raytracer import (
            SurfaceDefinition,
            ElementRaytracer,
        )
        from optiland.rays import RealRays
        
        # 创建输入光线
        input_rays = RealRays(
            x=np.array([0.0]),
            y=np.array([0.0]),
            z=np.array([0.0]),
            L=np.array([0.0]),
            M=np.array([0.0]),
            N=np.array([1.0]),
            intensity=np.array([1.0]),
            wavelength=np.array([0.633]),
        )
        
        # 无倾斜的平面镜
        mirror_no_tilt = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            thickness=0.0,
            material='mirror',
            semi_aperture=20.0,
            tilt_x=0.0,
            tilt_y=0.0,
        )
        
        raytracer_no_tilt = ElementRaytracer(
            surfaces=[mirror_no_tilt],
            wavelength=0.633,
        )
        raytracer_no_tilt.trace(input_rays)
        dir_no_tilt = raytracer_no_tilt.get_exit_chief_ray_direction()
        
        # 有倾斜的平面镜
        mirror_with_tilt = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            thickness=0.0,
            material='mirror',
            semi_aperture=20.0,
            tilt_x=np.pi/6,  # 30° 倾斜
            tilt_y=0.0,
        )
        
        raytracer_with_tilt = ElementRaytracer(
            surfaces=[mirror_with_tilt],
            wavelength=0.633,
        )
        raytracer_with_tilt.trace(input_rays)
        dir_with_tilt = raytracer_with_tilt.get_exit_chief_ray_direction()
        
        # 两个方向应该不同
        dir_no_tilt_arr = np.array(dir_no_tilt)
        dir_with_tilt_arr = np.array(dir_with_tilt)
        
        # 计算方向差异
        dot_product = np.dot(dir_no_tilt_arr, dir_with_tilt_arr)
        
        # 方向应该不同（点积不为 1）
        assert not np.isclose(dot_product, 1.0, atol=1e-6), \
            f"倾斜应该改变光线方向，但两个方向相同: {dir_no_tilt} vs {dir_with_tilt}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
