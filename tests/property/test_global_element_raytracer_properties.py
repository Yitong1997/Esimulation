"""
GlobalElementRaytracer 属性测试

本模块使用 Hypothesis 进行属性基测试，验证 GlobalElementRaytracer 的核心属性。

测试属性：
1. 法向量归一化验证
2. 方向余弦归一化保持
3. 旋转矩阵正交性
4. 坐标转换可逆性
5. OPD 坐标转换不变性
6. 主光线 OPD 为零

**Validates: Requirements 1.2, 1.3, 5.2, 5.3, 6.1, 6.2, 6.3, 7.1, 10.1, 10.4**
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.extra.numpy import arrays

from wavefront_to_rays.global_element_raytracer import (
    GlobalElementRaytracer,
    GlobalSurfaceDefinition,
    PlaneDef,
    _validate_normal_vector,
)


# =============================================================================
# 策略定义
# =============================================================================

@st.composite
def normalized_vector_strategy(draw):
    """生成归一化的 3D 向量"""
    # 生成非零向量
    x = draw(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))
    y = draw(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))
    z = draw(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))
    
    # 确保向量非零
    norm = np.sqrt(x**2 + y**2 + z**2)
    assume(norm > 1e-6)
    
    # 归一化
    return (x / norm, y / norm, z / norm)


@st.composite
def non_normalized_vector_strategy(draw):
    """生成非归一化的 3D 向量（长度不为 1）"""
    x = draw(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))
    y = draw(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))
    z = draw(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))
    
    norm = np.sqrt(x**2 + y**2 + z**2)
    assume(norm > 1e-6)
    
    # 缩放使其不为单位向量
    scale = draw(st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False))
    assume(abs(scale - 1.0) > 0.01)  # 确保不是单位向量
    
    return (x * scale / norm, y * scale / norm, z * scale / norm)


@st.composite
def position_strategy(draw):
    """生成有效的 3D 位置"""
    x = draw(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
    y = draw(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
    z = draw(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
    return (x, y, z)


@st.composite
def wavelength_strategy(draw):
    """生成有效的波长（μm）"""
    return draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))


# =============================================================================
# Property 1: 法向量归一化验证
# =============================================================================

class TestNormalVectorValidation:
    """法向量归一化验证测试
    
    **Validates: Requirements 1.2, 1.3, 10.1**
    """
    
    @given(normal=normalized_vector_strategy())
    @settings(max_examples=100)
    def test_accepts_normalized_vectors(self, normal):
        """验证系统正确接受归一化向量
        
        **Validates: Requirements 1.2**
        """
        # 不应抛出异常
        _validate_normal_vector(normal, name="测试向量")
    
    @given(normal=non_normalized_vector_strategy())
    @settings(max_examples=100)
    def test_rejects_non_normalized_vectors(self, normal):
        """验证系统正确拒绝非归一化向量
        
        **Validates: Requirements 1.3, 10.1**
        """
        with pytest.raises(ValueError) as exc_info:
            _validate_normal_vector(normal, name="测试向量")
        
        # 验证错误信息包含有用信息
        assert "归一化" in str(exc_info.value) or "normalized" in str(exc_info.value).lower()
    
    @given(
        position=position_strategy(),
        normal=non_normalized_vector_strategy(),
        wavelength=wavelength_strategy(),
    )
    @settings(max_examples=50)
    def test_plane_def_with_non_normalized_normal_raises_in_raytracer(
        self, position, normal, wavelength
    ):
        """验证 GlobalElementRaytracer 拒绝非归一化法向量的入射面
        
        **Validates: Requirements 10.1**
        """
        # 创建带有非归一化法向量的 PlaneDef
        entrance_plane = PlaneDef(position=position, normal=normal)
        
        # 创建一个简单的表面定义
        surface = GlobalSurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            vertex_position=(0.0, 0.0, 100.0),
        )
        
        # 应该在创建 GlobalElementRaytracer 时抛出 ValueError
        with pytest.raises(ValueError):
            GlobalElementRaytracer(
                surfaces=[surface],
                wavelength=wavelength,
                entrance_plane=entrance_plane,
            )


# =============================================================================
# Property 2: 方向余弦归一化保持
# =============================================================================

class TestDirectionCosineNormalization:
    """方向余弦归一化保持测试
    
    验证坐标转换后方向余弦仍满足 L² + M² + N² = 1
    
    **Validates: Requirements 5.2, 10.4**
    """
    
    @given(
        direction=normalized_vector_strategy(),
        entrance_normal=normalized_vector_strategy(),
    )
    @settings(max_examples=100)
    def test_direction_cosines_remain_normalized_after_transform(
        self, direction, entrance_normal
    ):
        """验证坐标转换后方向余弦保持归一化
        
        **Validates: Requirements 5.2, 10.4**
        """
        from hybrid_optical_propagation.hybrid_element_propagator_global import (
            HybridElementPropagatorGlobal,
        )
        from wavefront_to_rays.global_element_raytracer import PlaneDef
        from optiland.rays import RealRays
        
        # 创建传播器
        propagator = HybridElementPropagatorGlobal(
            wavelength_um=0.633,
            num_rays=10,
        )
        
        # 创建入射面定义
        entrance_plane = PlaneDef(
            position=(0.0, 0.0, 0.0),
            normal=entrance_normal,
        )
        
        # 创建模拟的光轴状态
        class MockOpticalAxisState:
            def __init__(self, pos, dir):
                self._pos = pos
                self._dir = dir
            
            @property
            def position(self):
                class Pos:
                    def to_array(self_inner):
                        return np.array(self._pos)
                return Pos()
            
            @property
            def direction(self):
                class Dir:
                    def to_array(self_inner):
                        return np.array(self._dir)
                return Dir()
        
        entrance_axis = MockOpticalAxisState(
            pos=(0.0, 0.0, 0.0),
            dir=entrance_normal,
        )
        
        # 创建局部坐标系光线
        L, M, N = direction
        local_rays = RealRays(
            x=np.array([0.0]),
            y=np.array([0.0]),
            z=np.array([0.0]),
            L=np.array([L]),
            M=np.array([M]),
            N=np.array([N]),
            intensity=np.array([1.0]),
            wavelength=np.array([0.633]),
        )
        local_rays.opd = np.array([0.0])
        
        # 转换到全局坐标系
        global_rays = propagator._local_to_global_rays(
            local_rays, entrance_plane, entrance_axis
        )
        
        # 验证方向余弦归一化
        L_global = np.asarray(global_rays.L)[0]
        M_global = np.asarray(global_rays.M)[0]
        N_global = np.asarray(global_rays.N)[0]
        
        norm_sq = L_global**2 + M_global**2 + N_global**2
        
        assert np.isclose(norm_sq, 1.0, rtol=1e-6), \
            f"方向余弦未归一化：L² + M² + N² = {norm_sq}"


# =============================================================================
# Property 3: 旋转矩阵正交性
# =============================================================================

class TestRotationMatrixOrthogonality:
    """旋转矩阵正交性测试
    
    验证 R × R^T = I 且 det(R) = 1
    
    **Validates: Requirements 5.3, 6.3**
    """
    
    @given(normal=normalized_vector_strategy())
    @settings(max_examples=100)
    def test_rotation_matrix_is_orthogonal(self, normal):
        """验证旋转矩阵是正交矩阵
        
        **Validates: Requirements 5.3, 6.3**
        """
        # 构建旋转矩阵（与 _local_to_global_rays 中相同的逻辑）
        z_axis = np.array(normal, dtype=np.float64)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        if abs(z_axis[2]) < 0.9:
            ref = np.array([0.0, 0.0, 1.0])
        else:
            ref = np.array([1.0, 0.0, 0.0])
        
        x_axis = np.cross(ref, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        R = np.column_stack([x_axis, y_axis, z_axis])
        
        # 验证 R × R^T = I
        RRT = R @ R.T
        assert np.allclose(RRT, np.eye(3), rtol=1e-6), \
            f"R × R^T ≠ I:\n{RRT}"
        
        # 验证 det(R) = 1（正交矩阵）
        det_R = np.linalg.det(R)
        assert np.isclose(det_R, 1.0, rtol=1e-6), \
            f"det(R) = {det_R}，期望为 1.0"


# =============================================================================
# Property 4: 坐标转换可逆性
# =============================================================================

class TestCoordinateTransformReversibility:
    """坐标转换可逆性测试
    
    验证局部→全局→局部转换后得到原始值
    
    **Validates: Requirements 5.1, 6.1**
    """
    
    @given(
        position=position_strategy(),
        direction=normalized_vector_strategy(),
        plane_normal=normalized_vector_strategy(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    def test_local_global_local_roundtrip(self, position, direction, plane_normal):
        """验证坐标转换的可逆性
        
        **Validates: Requirements 5.1, 6.1**
        """
        from hybrid_optical_propagation.hybrid_element_propagator_global import (
            HybridElementPropagatorGlobal,
        )
        from wavefront_to_rays.global_element_raytracer import PlaneDef
        from optiland.rays import RealRays
        
        # 解包位置
        x, y, z = position
        
        propagator = HybridElementPropagatorGlobal(
            wavelength_um=0.633,
            num_rays=10,
        )
        
        # 创建平面定义
        plane = PlaneDef(
            position=(0.0, 0.0, 0.0),
            normal=plane_normal,
        )
        
        # 创建模拟的光轴状态
        class MockOpticalAxisState:
            def __init__(self, pos, dir):
                self._pos = pos
                self._dir = dir
            
            @property
            def position(self):
                class Pos:
                    def to_array(self_inner):
                        return np.array(self._pos)
                return Pos()
            
            @property
            def direction(self):
                class Dir:
                    def to_array(self_inner):
                        return np.array(self._dir)
                return Dir()
        
        axis = MockOpticalAxisState(
            pos=(0.0, 0.0, 0.0),
            dir=plane_normal,
        )
        
        # 创建局部坐标系光线
        L, M, N = direction
        local_rays = RealRays(
            x=np.array([x]),
            y=np.array([y]),
            z=np.array([z]),
            L=np.array([L]),
            M=np.array([M]),
            N=np.array([N]),
            intensity=np.array([1.0]),
            wavelength=np.array([0.633]),
        )
        local_rays.opd = np.array([0.0])
        
        # 局部 → 全局
        global_rays = propagator._local_to_global_rays(local_rays, plane, axis)
        
        # 全局 → 局部
        recovered_rays = propagator._global_to_local_rays(global_rays, plane, axis)
        
        # 验证位置恢复
        assert np.isclose(np.asarray(recovered_rays.x)[0], x, rtol=1e-5, atol=1e-10), \
            f"x 坐标未恢复：原始 {x}，恢复 {np.asarray(recovered_rays.x)[0]}"
        assert np.isclose(np.asarray(recovered_rays.y)[0], y, rtol=1e-5, atol=1e-10), \
            f"y 坐标未恢复：原始 {y}，恢复 {np.asarray(recovered_rays.y)[0]}"
        assert np.isclose(np.asarray(recovered_rays.z)[0], z, rtol=1e-5, atol=1e-10), \
            f"z 坐标未恢复：原始 {z}，恢复 {np.asarray(recovered_rays.z)[0]}"
        
        # 验证方向恢复
        assert np.isclose(np.asarray(recovered_rays.L)[0], L, rtol=1e-5, atol=1e-10), \
            f"L 方向未恢复：原始 {L}，恢复 {np.asarray(recovered_rays.L)[0]}"
        assert np.isclose(np.asarray(recovered_rays.M)[0], M, rtol=1e-5, atol=1e-10), \
            f"M 方向未恢复：原始 {M}，恢复 {np.asarray(recovered_rays.M)[0]}"
        assert np.isclose(np.asarray(recovered_rays.N)[0], N, rtol=1e-5, atol=1e-10), \
            f"N 方向未恢复：原始 {N}，恢复 {np.asarray(recovered_rays.N)[0]}"


# =============================================================================
# Property 5: OPD 坐标转换不变性
# =============================================================================

class TestOPDCoordinateInvariance:
    """OPD 坐标转换不变性测试
    
    验证 OPD 在坐标转换过程中保持不变
    
    **Validates: Requirements 6.2**
    """
    
    @given(
        opd_value=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        plane_normal=normalized_vector_strategy(),
    )
    @settings(max_examples=100)
    def test_opd_preserved_during_transform(self, opd_value, plane_normal):
        """验证 OPD 在坐标转换中保持不变
        
        **Validates: Requirements 6.2**
        """
        from hybrid_optical_propagation.hybrid_element_propagator_global import (
            HybridElementPropagatorGlobal,
        )
        from wavefront_to_rays.global_element_raytracer import PlaneDef
        from optiland.rays import RealRays
        
        propagator = HybridElementPropagatorGlobal(
            wavelength_um=0.633,
            num_rays=10,
        )
        
        plane = PlaneDef(
            position=(0.0, 0.0, 0.0),
            normal=plane_normal,
        )
        
        class MockOpticalAxisState:
            def __init__(self, pos, dir):
                self._pos = pos
                self._dir = dir
            
            @property
            def position(self):
                class Pos:
                    def to_array(self_inner):
                        return np.array(self._pos)
                return Pos()
            
            @property
            def direction(self):
                class Dir:
                    def to_array(self_inner):
                        return np.array(self._dir)
                return Dir()
        
        axis = MockOpticalAxisState(
            pos=(0.0, 0.0, 0.0),
            dir=plane_normal,
        )
        
        # 创建带有 OPD 的光线
        local_rays = RealRays(
            x=np.array([1.0]),
            y=np.array([2.0]),
            z=np.array([0.0]),
            L=np.array([0.0]),
            M=np.array([0.0]),
            N=np.array([1.0]),
            intensity=np.array([1.0]),
            wavelength=np.array([0.633]),
        )
        local_rays.opd = np.array([opd_value])
        
        # 局部 → 全局
        global_rays = propagator._local_to_global_rays(local_rays, plane, axis)
        
        # 验证 OPD 保持不变
        assert np.isclose(np.asarray(global_rays.opd)[0], opd_value, rtol=1e-10), \
            f"OPD 在局部→全局转换中改变：原始 {opd_value}，转换后 {np.asarray(global_rays.opd)[0]}"
        
        # 全局 → 局部
        recovered_rays = propagator._global_to_local_rays(global_rays, plane, axis)
        
        # 验证 OPD 保持不变
        assert np.isclose(np.asarray(recovered_rays.opd)[0], opd_value, rtol=1e-10), \
            f"OPD 在全局→局部转换中改变：原始 {opd_value}，转换后 {np.asarray(recovered_rays.opd)[0]}"


# =============================================================================
# Property 6: 主光线 OPD 为零
# =============================================================================

class TestChiefRayOPDZero:
    """主光线 OPD 为零测试
    
    验证主光线的相对 OPD 为 0
    
    **Validates: Requirements 7.1**
    """
    
    @given(
        mirror_z=st.floats(min_value=50, max_value=500, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20)
    def test_chief_ray_opd_is_zero_for_flat_mirror(self, mirror_z):
        """验证平面镜反射后主光线 OPD 为零
        
        **Validates: Requirements 7.1**
        """
        from wavefront_to_rays.global_element_raytracer import (
            GlobalElementRaytracer,
            GlobalSurfaceDefinition,
            PlaneDef,
        )
        from optiland.rays import RealRays
        
        # 创建平面镜
        mirror = GlobalSurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            vertex_position=(0.0, 0.0, mirror_z),
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
        raytracer.trace_chief_ray()
        
        # 创建多条光线（包括主光线和边缘光线）
        n_rays = 5
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
        
        # 找到主光线（x=0 的光线）
        x_out = np.asarray(output_rays.x)
        chief_idx = np.argmin(np.abs(x_out))
        
        # 计算相对于主光线的 OPD
        opd_out = np.asarray(output_rays.opd)
        chief_opd = opd_out[chief_idx]
        relative_opd = opd_out - chief_opd
        
        # 验证主光线的相对 OPD 为 0
        assert np.isclose(relative_opd[chief_idx], 0.0, atol=1e-10), \
            f"主光线相对 OPD 不为零：{relative_opd[chief_idx]}"
        
        # 对于平面镜，所有光线的相对 OPD 应该相同（都为 0）
        # 因为平面镜不引入像差
        assert np.allclose(relative_opd, 0.0, atol=1e-6), \
            f"平面镜反射后存在非零相对 OPD：{relative_opd}"


# =============================================================================
# 运行测试
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
