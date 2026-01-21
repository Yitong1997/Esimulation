"""
坐标系统属性基测试

本模块使用 hypothesis 库对 Zemax 光轴追踪与坐标转换模块进行属性基测试。
每个测试验证设计文档中定义的正确性属性。

测试框架：pytest + hypothesis
最小迭代次数：100

作者：混合光学仿真项目
"""

import numpy as np
from hypothesis import given, strategies as st, settings, assume
import pytest

from sequential_system.coordinate_system import (
    CurrentCoordinateSystem,
    CoordinateBreakProcessor,
    GlobalSurfaceDefinition,
)


# =============================================================================
# 测试策略定义
# =============================================================================

# 角度策略（弧度），限制在合理范围内避免数值问题
angle_strategy = st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False)

# 小角度策略（用于测试数值稳定性）
small_angle_strategy = st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False)

# 长度策略（mm），限制在合理范围内
length_strategy = st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)

# 正半径策略（mm）
positive_radius_strategy = st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False)

# 圆锥常数策略
conic_strategy = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)

# 变换顺序策略
order_strategy = st.sampled_from([0, 1])

# 厚度策略（可正可负）
thickness_strategy = st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False)


# =============================================================================
# CurrentCoordinateSystem 属性测试
# =============================================================================

class TestCurrentCoordinateSystemProperties:
    """CurrentCoordinateSystem 类的属性基测试"""
    
    @given(
        tilt_x=angle_strategy,
        tilt_y=angle_strategy,
        tilt_z=angle_strategy
    )
    @settings(max_examples=100)
    def test_property_6_direction_cosines_unit_vector(
        self, tilt_x: float, tilt_y: float, tilt_z: float
    ):
        """
        **Feature: zemax-optical-axis-tracing, Property 6: 方向余弦单位向量**
        **Validates: Requirements 1.4, 5.3**
        
        对于任意旋转后的坐标系，轴向量 (x_axis, y_axis, z_axis) 
        应该始终为单位向量，即 |axis| = 1。
        """
        # Arrange: 创建初始坐标系并应用旋转
        cs = CurrentCoordinateSystem.identity()
        cs_rotated = cs.apply_rotation(tilt_x, tilt_y, tilt_z)
        
        # Assert: 所有轴向量应为单位向量
        x_norm = np.linalg.norm(cs_rotated.x_axis)
        y_norm = np.linalg.norm(cs_rotated.y_axis)
        z_norm = np.linalg.norm(cs_rotated.z_axis)
        
        np.testing.assert_allclose(x_norm, 1.0, rtol=1e-10)
        np.testing.assert_allclose(y_norm, 1.0, rtol=1e-10)
        np.testing.assert_allclose(z_norm, 1.0, rtol=1e-10)

    @given(thickness=thickness_strategy)
    @settings(max_examples=100)
    def test_property_8_thickness_advancement(self, thickness: float):
        """
        **Feature: zemax-optical-axis-tracing, Property 8: 厚度前进**
        **Validates: Requirements 3.1**
        
        对于任意厚度 t 和当前坐标系的 Z 轴方向 z，
        原点应该精确移动 t × z。
        """
        # Arrange: 创建初始坐标系
        cs = CurrentCoordinateSystem.identity()
        
        # Act: 沿 Z 轴前进
        cs_advanced = cs.advance_along_z(thickness)
        
        # Assert: 原点应该移动 thickness × z_axis
        expected_origin = cs.origin + thickness * cs.z_axis
        np.testing.assert_allclose(cs_advanced.origin, expected_origin, rtol=1e-10)
    
    @given(
        tilt_x=angle_strategy,
        tilt_y=angle_strategy,
        tilt_z=angle_strategy,
        thickness=thickness_strategy
    )
    @settings(max_examples=100)
    def test_property_8_thickness_after_rotation(
        self, tilt_x: float, tilt_y: float, tilt_z: float, thickness: float
    ):
        """
        **Feature: zemax-optical-axis-tracing, Property 8: 旋转后厚度前进**
        **Validates: Requirements 3.1, 3.2**
        
        旋转后沿新 Z 轴前进，原点应该沿旋转后的 Z 轴方向移动。
        """
        # Arrange: 创建旋转后的坐标系
        cs = CurrentCoordinateSystem.identity()
        cs_rotated = cs.apply_rotation(tilt_x, tilt_y, tilt_z)
        
        # Act: 沿新 Z 轴前进
        cs_advanced = cs_rotated.advance_along_z(thickness)
        
        # Assert: 原点应该移动 thickness × new_z_axis
        expected_origin = cs_rotated.origin + thickness * cs_rotated.z_axis
        np.testing.assert_allclose(cs_advanced.origin, expected_origin, rtol=1e-10)
    
    @given(
        dx=length_strategy,
        dy=length_strategy
    )
    @settings(max_examples=100)
    def test_property_4_decenter_translation(self, dx: float, dy: float):
        """
        **Feature: zemax-optical-axis-tracing, Property 4: 偏心平移正确性**
        **Validates: Requirements 2.4**
        
        对于任意偏心值 (dx, dy) 和当前坐标系的轴向量矩阵 A，
        原点应该精确移动 dx × A[:, 0] + dy × A[:, 1]。
        """
        # Arrange: 创建初始坐标系
        cs = CurrentCoordinateSystem.identity()
        
        # Act: 应用偏心
        cs_decentered = cs.apply_decenter(dx, dy)
        
        # Assert: 原点应该移动 dx × x_axis + dy × y_axis
        expected_origin = cs.origin + dx * cs.x_axis + dy * cs.y_axis
        np.testing.assert_allclose(cs_decentered.origin, expected_origin, rtol=1e-10)
    
    @given(
        tilt_x=angle_strategy,
        tilt_y=angle_strategy,
        tilt_z=angle_strategy,
        dx=length_strategy,
        dy=length_strategy
    )
    @settings(max_examples=100)
    def test_property_4_decenter_after_rotation(
        self, tilt_x: float, tilt_y: float, tilt_z: float, dx: float, dy: float
    ):
        """
        **Feature: zemax-optical-axis-tracing, Property 4: 旋转后偏心平移**
        **Validates: Requirements 2.4**
        
        旋转后应用偏心，原点应该沿旋转后的 X/Y 轴方向移动。
        """
        # Arrange: 创建旋转后的坐标系
        cs = CurrentCoordinateSystem.identity()
        cs_rotated = cs.apply_rotation(tilt_x, tilt_y, tilt_z)
        
        # Act: 应用偏心
        cs_decentered = cs_rotated.apply_decenter(dx, dy)
        
        # Assert: 原点应该移动 dx × new_x_axis + dy × new_y_axis
        expected_origin = cs_rotated.origin + dx * cs_rotated.x_axis + dy * cs_rotated.y_axis
        np.testing.assert_allclose(cs_decentered.origin, expected_origin, rtol=1e-10)


# =============================================================================
# CoordinateBreakProcessor 属性测试
# =============================================================================

class TestCoordinateBreakProcessorProperties:
    """CoordinateBreakProcessor 类的属性基测试"""
    
    @given(
        dx=length_strategy,
        dy=length_strategy,
        tilt_x=angle_strategy,
        tilt_y=angle_strategy,
        tilt_z=angle_strategy,
        thickness=thickness_strategy
    )
    @settings(max_examples=100)
    def test_property_1_order_0_transformation(
        self, dx: float, dy: float, 
        tilt_x: float, tilt_y: float, tilt_z: float,
        thickness: float
    ):
        """
        **Feature: zemax-optical-axis-tracing, Property 1: 坐标断点 Order=0 变换正确性**
        **Validates: Requirements 2.1, 8.4**
        
        对于 Order=0 的坐标断点，变换顺序应为：
        1. 先平移（使用当前轴）
        2. 后旋转
        3. 最后沿新 Z 轴前进厚度
        """
        # Arrange: 创建初始坐标系
        cs = CurrentCoordinateSystem.identity()
        
        # Act: 使用 Order=0 处理坐标断点
        cs_result = CoordinateBreakProcessor.process(
            cs, dx, dy, tilt_x, tilt_y, tilt_z, order=0, thickness=thickness
        )
        
        # 手动计算预期结果
        # 1. 先平移
        origin_after_decenter = cs.origin + dx * cs.x_axis + dy * cs.y_axis
        # 2. 后旋转
        R_xyz = CoordinateBreakProcessor.rotation_matrix_xyz(tilt_x, tilt_y, tilt_z)
        axes_after_rotation = cs.axes @ R_xyz
        # 3. 沿新 Z 轴前进
        new_z_axis = axes_after_rotation[:, 2]
        expected_origin = origin_after_decenter + thickness * new_z_axis
        expected_axes = axes_after_rotation
        
        # Assert
        np.testing.assert_allclose(cs_result.origin, expected_origin, rtol=1e-10)
        np.testing.assert_allclose(cs_result.axes, expected_axes, rtol=1e-10)

    @given(
        dx=length_strategy,
        dy=length_strategy,
        tilt_x=angle_strategy,
        tilt_y=angle_strategy,
        tilt_z=angle_strategy,
        thickness=thickness_strategy
    )
    @settings(max_examples=100)
    def test_property_2_order_1_transformation(
        self, dx: float, dy: float, 
        tilt_x: float, tilt_y: float, tilt_z: float,
        thickness: float
    ):
        """
        **Feature: zemax-optical-axis-tracing, Property 2: 坐标断点 Order=1 变换正确性**
        **Validates: Requirements 2.2, 8.5**
        
        对于 Order=1 的坐标断点，变换顺序应为：
        1. 先旋转
        2. 后平移（使用旋转后的轴）
        3. 最后沿新 Z 轴前进厚度
        """
        # Arrange: 创建初始坐标系
        cs = CurrentCoordinateSystem.identity()
        
        # Act: 使用 Order=1 处理坐标断点
        cs_result = CoordinateBreakProcessor.process(
            cs, dx, dy, tilt_x, tilt_y, tilt_z, order=1, thickness=thickness
        )
        
        # 手动计算预期结果
        # 1. 先旋转
        R_xyz = CoordinateBreakProcessor.rotation_matrix_xyz(tilt_x, tilt_y, tilt_z)
        axes_after_rotation = cs.axes @ R_xyz
        # 2. 后平移（使用旋转后的轴）
        new_x_axis = axes_after_rotation[:, 0]
        new_y_axis = axes_after_rotation[:, 1]
        origin_after_decenter = cs.origin + dx * new_x_axis + dy * new_y_axis
        # 3. 沿新 Z 轴前进
        new_z_axis = axes_after_rotation[:, 2]
        expected_origin = origin_after_decenter + thickness * new_z_axis
        expected_axes = axes_after_rotation
        
        # Assert
        np.testing.assert_allclose(cs_result.origin, expected_origin, rtol=1e-10)
        np.testing.assert_allclose(cs_result.axes, expected_axes, rtol=1e-10)
    
    @given(
        tilt_x=angle_strategy,
        tilt_y=angle_strategy,
        tilt_z=angle_strategy
    )
    @settings(max_examples=100)
    def test_property_3_rotation_order_xyz(
        self, tilt_x: float, tilt_y: float, tilt_z: float
    ):
        """
        **Feature: zemax-optical-axis-tracing, Property 3: 旋转顺序 X→Y→Z**
        **Validates: Requirements 2.3**
        
        组合旋转矩阵应等于 R_z × R_y × R_x，
        即先绕 X 轴旋转，再绕 Y 轴，最后绕 Z 轴。
        """
        # Act: 计算组合旋转矩阵
        R_xyz = CoordinateBreakProcessor.rotation_matrix_xyz(tilt_x, tilt_y, tilt_z)
        
        # 手动计算预期结果
        R_x = CoordinateBreakProcessor.rotation_matrix_x(tilt_x)
        R_y = CoordinateBreakProcessor.rotation_matrix_y(tilt_y)
        R_z = CoordinateBreakProcessor.rotation_matrix_z(tilt_z)
        expected_R = R_z @ R_y @ R_x
        
        # Assert
        np.testing.assert_allclose(R_xyz, expected_R, rtol=1e-10)
    
    @given(
        dx=length_strategy,
        dy=length_strategy,
        tilt_x=angle_strategy,
        tilt_y=angle_strategy,
        tilt_z=angle_strategy,
        thickness=thickness_strategy
    )
    @settings(max_examples=100)
    def test_property_9_coordinate_break_thickness(
        self, dx: float, dy: float, 
        tilt_x: float, tilt_y: float, tilt_z: float,
        thickness: float
    ):
        """
        **Feature: zemax-optical-axis-tracing, Property 9: 坐标断点厚度处理**
        **Validates: Requirements 3.4**
        
        对于任意坐标断点，完成偏心和旋转变换后，
        原点应该沿新 Z 轴方向前进 thickness。
        """
        # Arrange: 创建初始坐标系
        cs = CurrentCoordinateSystem.identity()
        
        # 计算不带厚度的结果
        cs_no_thickness = CoordinateBreakProcessor.process(
            cs, dx, dy, tilt_x, tilt_y, tilt_z, order=0, thickness=0.0
        )
        
        # 计算带厚度的结果
        cs_with_thickness = CoordinateBreakProcessor.process(
            cs, dx, dy, tilt_x, tilt_y, tilt_z, order=0, thickness=thickness
        )
        
        # Assert: 两者的差应该是 thickness × new_z_axis
        origin_diff = cs_with_thickness.origin - cs_no_thickness.origin
        expected_diff = thickness * cs_no_thickness.z_axis
        
        # 使用合理的数值容差（浮点运算累积误差）
        np.testing.assert_allclose(origin_diff, expected_diff, rtol=1e-9, atol=1e-12)
    
    @given(angle=angle_strategy)
    @settings(max_examples=100)
    def test_rotation_matrix_x_orthogonal(self, angle: float):
        """
        **Feature: zemax-optical-axis-tracing, 旋转矩阵正交性**
        **Validates: Requirements 4.1**
        
        绕 X 轴的旋转矩阵应该是正交矩阵（R^T × R = I）。
        """
        R = CoordinateBreakProcessor.rotation_matrix_x(angle)
        
        # 验证正交性（使用合理的数值容差）
        np.testing.assert_allclose(R.T @ R, np.eye(3), rtol=1e-10, atol=1e-15)
        np.testing.assert_allclose(R @ R.T, np.eye(3), rtol=1e-10, atol=1e-15)
        
        # 验证行列式为 1（旋转矩阵）
        np.testing.assert_allclose(np.linalg.det(R), 1.0, rtol=1e-10)
    
    @given(angle=angle_strategy)
    @settings(max_examples=100)
    def test_rotation_matrix_y_orthogonal(self, angle: float):
        """
        **Feature: zemax-optical-axis-tracing, 旋转矩阵正交性**
        **Validates: Requirements 4.2**
        
        绕 Y 轴的旋转矩阵应该是正交矩阵。
        """
        R = CoordinateBreakProcessor.rotation_matrix_y(angle)
        
        np.testing.assert_allclose(R.T @ R, np.eye(3), rtol=1e-10, atol=1e-15)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, rtol=1e-10)
    
    @given(angle=angle_strategy)
    @settings(max_examples=100)
    def test_rotation_matrix_z_orthogonal(self, angle: float):
        """
        **Feature: zemax-optical-axis-tracing, 旋转矩阵正交性**
        **Validates: Requirements 4.3**
        
        绕 Z 轴的旋转矩阵应该是正交矩阵。
        """
        R = CoordinateBreakProcessor.rotation_matrix_z(angle)
        
        np.testing.assert_allclose(R.T @ R, np.eye(3), rtol=1e-10, atol=1e-15)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, rtol=1e-10)


# =============================================================================
# GlobalSurfaceDefinition 属性测试
# =============================================================================

class TestGlobalSurfaceDefinitionProperties:
    """GlobalSurfaceDefinition 类的属性基测试"""
    
    @given(
        radius=positive_radius_strategy,
        vertex_x=length_strategy,
        vertex_y=length_strategy,
        vertex_z=length_strategy
    )
    @settings(max_examples=100)
    def test_property_12_curvature_center_calculation(
        self, radius: float, vertex_x: float, vertex_y: float, vertex_z: float
    ):
        """
        **Feature: zemax-optical-axis-tracing, Property 12: 曲率中心计算**
        **Validates: Requirements 6.1, 6.2, 6.5**
        
        对于任意有限半径 R 和姿态矩阵的 Z 轴 z，
        曲率中心应该位于 vertex_position + R × z。
        """
        # Arrange
        vertex = np.array([vertex_x, vertex_y, vertex_z])
        orientation = np.eye(3)  # 使用单位矩阵
        
        surface = GlobalSurfaceDefinition(
            index=1,
            surface_type='standard',
            vertex_position=vertex,
            orientation=orientation,
            radius=radius
        )
        
        # Act
        center = surface.curvature_center
        
        # Assert
        expected_center = vertex + radius * orientation[:, 2]
        np.testing.assert_allclose(center, expected_center, rtol=1e-10)
    
    @given(
        radius=positive_radius_strategy,
        tilt_x=angle_strategy,
        tilt_y=angle_strategy,
        tilt_z=angle_strategy
    )
    @settings(max_examples=100)
    def test_property_18_curvature_center_after_rotation(
        self, radius: float, tilt_x: float, tilt_y: float, tilt_z: float
    ):
        """
        **Feature: zemax-optical-axis-tracing, Property 18: 旋转坐标系后的曲率中心计算**
        **Validates: Requirements 6.5, 9.8, 9.9**
        
        对于旋转后的坐标系定义的光学表面，曲率中心应该使用
        旋转后的 Z 轴方向计算：vertex + R × rotated_z_axis。
        """
        # Arrange: 创建旋转后的姿态矩阵
        R_xyz = CoordinateBreakProcessor.rotation_matrix_xyz(tilt_x, tilt_y, tilt_z)
        orientation = np.eye(3) @ R_xyz
        vertex = np.array([0.0, 0.0, 100.0])
        
        surface = GlobalSurfaceDefinition(
            index=1,
            surface_type='standard',
            vertex_position=vertex,
            orientation=orientation,
            radius=radius
        )
        
        # Act
        center = surface.curvature_center
        
        # Assert: 曲率中心 = 顶点 + R × 旋转后的 Z 轴
        rotated_z_axis = orientation[:, 2]
        expected_center = vertex + radius * rotated_z_axis
        np.testing.assert_allclose(center, expected_center, rtol=1e-10)
    
    @given(conic=conic_strategy)
    @settings(max_examples=100)
    def test_property_13_conic_constant_preserved(self, conic: float):
        """
        **Feature: zemax-optical-axis-tracing, Property 13: 圆锥常数保持**
        **Validates: Requirements 7.1**
        
        圆锥常数 k 应该在转换过程中保持不变。
        """
        # Arrange
        surface = GlobalSurfaceDefinition(
            index=1,
            surface_type='standard',
            vertex_position=np.array([0.0, 0.0, 0.0]),
            orientation=np.eye(3),
            radius=100.0,
            conic=conic
        )
        
        # Assert: conic 值应该保持不变
        assert surface.conic == conic
    
    def test_curvature_center_none_for_flat_surface(self):
        """
        **Feature: zemax-optical-axis-tracing, 平面曲率中心**
        **Validates: Requirements 6.3**
        
        对于平面（无穷大半径），曲率中心应该返回 None。
        """
        surface = GlobalSurfaceDefinition(
            index=1,
            surface_type='flat',
            vertex_position=np.array([0.0, 0.0, 0.0]),
            orientation=np.eye(3),
            radius=np.inf
        )
        
        assert surface.curvature_center is None
    
    @given(
        radius=st.one_of(
            positive_radius_strategy,
            st.floats(min_value=-10000.0, max_value=-1.0, allow_nan=False, allow_infinity=False)
        )
    )
    @settings(max_examples=100)
    def test_property_12_negative_radius(self, radius: float):
        """
        **Feature: zemax-optical-axis-tracing, Property 12: 负半径曲率中心**
        **Validates: Requirements 6.2**
        
        对于负半径（凸面），曲率中心应该在 -Z 方向。
        """
        assume(radius != 0)  # 排除零半径
        
        vertex = np.array([0.0, 0.0, 0.0])
        orientation = np.eye(3)
        
        surface = GlobalSurfaceDefinition(
            index=1,
            surface_type='standard',
            vertex_position=vertex,
            orientation=orientation,
            radius=radius
        )
        
        center = surface.curvature_center
        expected_center = vertex + radius * orientation[:, 2]
        
        np.testing.assert_allclose(center, expected_center, rtol=1e-10)


# =============================================================================
# SurfaceTraversalAlgorithm 属性测试
# =============================================================================

class TestSurfaceTraversalProperties:
    """SurfaceTraversalAlgorithm 类的属性基测试"""
    
    @given(
        tilt_x_1=angle_strategy,
        tilt_x_2=angle_strategy
    )
    @settings(max_examples=100)
    def test_property_14_consecutive_coordinate_breaks(
        self, tilt_x_1: float, tilt_x_2: float
    ):
        """
        **Feature: zemax-optical-axis-tracing, Property 14: 连续坐标断点累积**
        **Validates: Requirements 5.2, 5.6**
        
        对于 N 个连续坐标断点，最终坐标系状态应该等于
        按顺序应用所有 N 个变换的结果。
        """
        # Arrange: 创建初始坐标系
        cs = CurrentCoordinateSystem.identity()
        
        # Act: 依次应用两个坐标断点
        cs_after_1 = CoordinateBreakProcessor.process(
            cs, 0, 0, tilt_x_1, 0, 0, order=0, thickness=0
        )
        cs_after_2 = CoordinateBreakProcessor.process(
            cs_after_1, 0, 0, tilt_x_2, 0, 0, order=0, thickness=0
        )
        
        # 验证：两次旋转的累积效果
        # 第一次旋转后的 Z 轴
        R1 = CoordinateBreakProcessor.rotation_matrix_xyz(tilt_x_1, 0, 0)
        expected_axes_1 = np.eye(3) @ R1
        
        # 第二次旋转后的 Z 轴（在第一次旋转的基础上）
        R2 = CoordinateBreakProcessor.rotation_matrix_xyz(tilt_x_2, 0, 0)
        expected_axes_2 = expected_axes_1 @ R2
        
        np.testing.assert_allclose(cs_after_2.axes, expected_axes_2, rtol=1e-10)
    
    @given(thickness=st.floats(min_value=-500.0, max_value=-0.1, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_property_15_negative_thickness(self, thickness: float):
        """
        **Feature: zemax-optical-axis-tracing, Property 15: 负厚度处理**
        **Validates: Requirements 3.3, 10.7**
        
        对于负厚度 t，原点应该沿 Z 轴负方向移动 |t| 距离。
        """
        # Arrange
        cs = CurrentCoordinateSystem.identity()
        
        # Act
        cs_advanced = cs.advance_along_z(thickness)
        
        # Assert: 原点应该在 Z 轴负方向
        expected_origin = np.array([0.0, 0.0, thickness])
        np.testing.assert_allclose(cs_advanced.origin, expected_origin, rtol=1e-10)
        
        # 验证移动方向
        assert cs_advanced.origin[2] < 0  # Z 坐标应该为负
    
    def test_property_16_virtual_vs_optical_surface(self):
        """
        **Feature: zemax-optical-axis-tracing, Property 16: 虚拟表面与光学表面分类**
        **Validates: Requirements 5.5, 10.4, 10.5, 10.6**
        
        表面应该根据 surface_type 字段正确分类：
        - coordinate_break -> 虚拟表面（不生成 GlobalSurfaceDefinition）
        - standard/even_asphere -> 光学表面（生成 GlobalSurfaceDefinition）
        """
        from sequential_system.zmx_parser import ZmxSurfaceData, ZmxDataModel
        from sequential_system.coordinate_system import SurfaceTraversalAlgorithm
        
        # 创建测试数据模型
        zmx_data = ZmxDataModel()
        
        # 添加坐标断点（虚拟表面）
        zmx_data.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=0.0
        )
        
        # 添加标准表面（光学表面）
        zmx_data.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=100.0,
            thickness=50.0,
            is_mirror=True
        )
        
        # 添加另一个坐标断点
        zmx_data.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=0.0
        )
        
        # 遍历
        traversal = SurfaceTraversalAlgorithm(zmx_data)
        global_surfaces = traversal.traverse()
        
        # Assert: 只有光学表面被转换
        assert len(global_surfaces) == 1
        assert global_surfaces[0].index == 1
        assert global_surfaces[0].surface_type == 'standard'


# =============================================================================
# 反射镜行为属性测试
# =============================================================================

class TestMirrorBehaviorProperties:
    """反射镜行为的属性基测试"""
    
    def test_property_10_mirror_does_not_change_coordinate_system(self):
        """
        **Feature: zemax-optical-axis-tracing, Property 10: 反射镜不改变当前坐标系**
        **Validates: Requirements 9.1, 9.2**
        
        处理反射镜表面时，当前坐标系不应该自动旋转。
        坐标系只应该沿 Z 轴前进厚度。
        """
        from sequential_system.zmx_parser import ZmxSurfaceData, ZmxDataModel
        from sequential_system.coordinate_system import SurfaceTraversalAlgorithm
        
        # 创建测试数据模型
        zmx_data = ZmxDataModel()
        
        # 添加反射镜表面
        zmx_data.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='standard',
            radius=np.inf,
            thickness=100.0,
            is_mirror=True
        )
        
        # 遍历
        traversal = SurfaceTraversalAlgorithm(zmx_data)
        global_surfaces = traversal.traverse()
        
        # 获取遍历后的坐标系状态
        final_cs = traversal.current_coordinate_system
        
        # Assert: 坐标系轴向量应该保持不变（单位矩阵）
        np.testing.assert_allclose(final_cs.axes, np.eye(3), rtol=1e-10)
        
        # Assert: 原点应该沿 Z 轴前进 100mm
        np.testing.assert_allclose(final_cs.origin, [0, 0, 100], rtol=1e-10)
    
    def test_property_11_surface_vertex_and_orientation(self):
        """
        **Feature: zemax-optical-axis-tracing, Property 11: 表面顶点和姿态转换**
        **Validates: Requirements 9.1, 9.2**
        
        转换后的 GlobalSurfaceDefinition 应该具有：
        1. vertex_position = 转换时的 current_origin
        2. orientation = 转换时的 current_axes
        """
        from sequential_system.zmx_parser import ZmxSurfaceData, ZmxDataModel
        from sequential_system.coordinate_system import SurfaceTraversalAlgorithm
        
        # 创建测试数据模型：先旋转 45 度，再添加表面
        zmx_data = ZmxDataModel()
        
        # 坐标断点：绕 X 轴旋转 45 度
        zmx_data.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=100.0  # 沿新 Z 轴前进 100mm
        )
        
        # 反射镜表面
        zmx_data.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=200.0,
            thickness=50.0,
            is_mirror=True
        )
        
        # 遍历
        traversal = SurfaceTraversalAlgorithm(zmx_data)
        global_surfaces = traversal.traverse()
        
        # Assert: 应该有一个光学表面
        assert len(global_surfaces) == 1
        surface = global_surfaces[0]
        
        # 计算预期的顶点位置
        # 坐标断点后：原点沿旋转后的 Z 轴前进 100mm
        angle = np.deg2rad(45.0)
        rotated_z = np.array([0, -np.sin(angle), np.cos(angle)])
        expected_vertex = 100.0 * rotated_z
        
        np.testing.assert_allclose(surface.vertex_position, expected_vertex, rtol=1e-10)
        
        # 计算预期的姿态矩阵
        R_x = CoordinateBreakProcessor.rotation_matrix_x(angle)
        expected_orientation = np.eye(3) @ R_x
        
        np.testing.assert_allclose(surface.orientation, expected_orientation, rtol=1e-10)


# =============================================================================
# ZemaxToOptilandConverter 属性测试
# =============================================================================

class TestOptilandConverterProperties:
    """ZemaxToOptilandConverter 类的属性基测试"""
    
    @given(
        radius=positive_radius_strategy,
        conic=conic_strategy
    )
    @settings(max_examples=100)
    def test_property_17_optiland_surface_params(self, radius: float, conic: float):
        """
        **Feature: zemax-optical-axis-tracing, Property 17: optiland 表面参数传递**
        **Validates: Requirements 10.2, 10.3, 10.4**
        
        转换到 optiland 时，半径和 conic 常数应该正确传递。
        """
        # Arrange
        surface = GlobalSurfaceDefinition(
            index=1,
            surface_type='standard',
            vertex_position=np.array([0.0, 0.0, 0.0]),
            orientation=np.eye(3),
            radius=radius,
            conic=conic,
            is_mirror=True,
            semi_aperture=25.0
        )
        
        # Act
        params = surface.to_optiland_params()
        
        # Assert
        assert params['radius'] == radius
        assert params['conic'] == conic
        assert params['is_mirror'] == True
        assert params['semi_diameter'] == 25.0
    
    def test_property_17_mirror_material(self):
        """
        **Feature: zemax-optical-axis-tracing, Property 17: 反射镜材料设置**
        **Validates: Requirements 10.3**
        
        反射镜表面应该正确设置 mirror 材料。
        """
        from sequential_system.coordinate_system import ZemaxToOptilandConverter
        
        # Arrange
        surface = GlobalSurfaceDefinition(
            index=1,
            surface_type='standard',
            vertex_position=np.array([0.0, 0.0, 0.0]),
            orientation=np.eye(3),
            radius=100.0,
            is_mirror=True,
            semi_aperture=25.0,
            thickness=50.0
        )
        
        # Act
        converter = ZemaxToOptilandConverter([surface])
        # 注意：实际转换需要 optiland 库，这里只测试参数准备
        params = surface.to_optiland_params()
        
        # Assert
        assert params['is_mirror'] == True
    
    @given(
        radius=positive_radius_strategy,
        tilt_x=angle_strategy,
        tilt_y=angle_strategy,
        tilt_z=angle_strategy
    )
    @settings(max_examples=100)
    def test_property_18_rotated_curvature_center(
        self, radius: float, tilt_x: float, tilt_y: float, tilt_z: float
    ):
        """
        **Feature: zemax-optical-axis-tracing, Property 18: 旋转坐标系后的曲率中心计算**
        **Validates: Requirements 6.5, 9.8, 9.9**
        
        对于旋转后的坐标系定义的光学表面，曲率中心应该使用
        旋转后的 Z 轴方向计算。
        """
        # Arrange: 创建旋转后的姿态矩阵
        R_xyz = CoordinateBreakProcessor.rotation_matrix_xyz(tilt_x, tilt_y, tilt_z)
        orientation = np.eye(3) @ R_xyz
        vertex = np.array([10.0, 20.0, 30.0])
        
        surface = GlobalSurfaceDefinition(
            index=1,
            surface_type='standard',
            vertex_position=vertex,
            orientation=orientation,
            radius=radius
        )
        
        # Act
        center = surface.curvature_center
        
        # Assert: 曲率中心 = 顶点 + R × 旋转后的 Z 轴
        rotated_z_axis = orientation[:, 2]
        expected_center = vertex + radius * rotated_z_axis
        
        np.testing.assert_allclose(center, expected_center, rtol=1e-10)
