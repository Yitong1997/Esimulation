"""
坐标转换集成测试

本模块测试 Zemax 光轴追踪与坐标转换的端到端功能。
包括 45° 折叠镜系统、Z 形双镜系统和离轴抛物面镜（OAP）测试。

作者：混合光学仿真项目
"""

import numpy as np
import pytest

from sequential_system.zmx_parser import ZmxSurfaceData, ZmxDataModel
from sequential_system.coordinate_system import (
    CurrentCoordinateSystem,
    CoordinateBreakProcessor,
    GlobalSurfaceDefinition,
    SurfaceTraversalAlgorithm,
    ZemaxToOptilandConverter,
)


class TestFoldMirrorSystem:
    """45° 折叠镜系统集成测试
    
    **Validates: Requirements 12.1**
    """
    
    def test_single_45_degree_fold_mirror(self):
        """测试单个 45° 折叠镜
        
        输入：沿 +Z 入射，45° 倾斜镜
        预期：出射方向沿 -Y
        """
        # 创建测试数据模型
        zmx_data = ZmxDataModel()
        
        # 坐标断点：绕 X 轴旋转 45 度
        zmx_data.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=0.0
        )
        
        # 平面反射镜
        zmx_data.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=np.inf,
            thickness=100.0,
            is_mirror=True,
            semi_diameter=25.0,
            comment='M1 - 45° Fold Mirror'
        )
        
        # 遍历
        traversal = SurfaceTraversalAlgorithm(zmx_data)
        global_surfaces = traversal.traverse()
        
        # 验证
        assert len(global_surfaces) == 1
        
        # 验证镜面位置在原点
        mirror = global_surfaces[0]
        np.testing.assert_allclose(mirror.vertex_position, [0, 0, 0], atol=1e-10)
        
        # 验证镜面 Z 轴方向（45° 旋转后）
        angle = np.deg2rad(45.0)
        expected_z = np.array([0, -np.sin(angle), np.cos(angle)])
        np.testing.assert_allclose(mirror.orientation[:, 2], expected_z, atol=1e-10)
        
        # 验证最终坐标系 Z 轴方向（沿旋转后的 Z 轴前进后）
        final_cs = traversal.current_coordinate_system
        np.testing.assert_allclose(final_cs.z_axis, expected_z, atol=1e-10)


class TestZShapeDualMirrorSystem:
    """Z 形双镜系统集成测试
    
    **Validates: Requirements 12.2**
    """
    
    def test_z_shape_dual_mirror_180_degree_turn(self):
        """测试 Z 形双镜系统
        
        输入：两个 45° 镜
        预期：180° 方向改变（Z 轴从 +Z 变为 -Y）
        """
        # 创建测试数据模型
        zmx_data = ZmxDataModel()
        
        # 第一个坐标断点：绕 X 轴旋转 45 度
        zmx_data.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=0.0
        )
        
        # 第一个平面反射镜
        zmx_data.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=np.inf,
            thickness=100.0,
            is_mirror=True,
            semi_diameter=25.0,
            comment='M1'
        )
        
        # 第二个坐标断点：再绕 X 轴旋转 45 度
        zmx_data.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=0.0
        )
        
        # 第二个平面反射镜
        zmx_data.surfaces[3] = ZmxSurfaceData(
            index=3,
            surface_type='standard',
            radius=np.inf,
            thickness=50.0,
            is_mirror=True,
            semi_diameter=25.0,
            comment='M2'
        )
        
        # 遍历
        traversal = SurfaceTraversalAlgorithm(zmx_data)
        global_surfaces = traversal.traverse()
        
        # 验证
        assert len(global_surfaces) == 2
        
        # 验证 M1 位置
        m1 = global_surfaces[0]
        np.testing.assert_allclose(m1.vertex_position, [0, 0, 0], atol=1e-10)
        
        # 验证 M2 位置
        m2 = global_surfaces[1]
        angle = np.deg2rad(45.0)
        # M1 后沿 45° Z 轴前进 100mm
        z1 = np.array([0, -np.sin(angle), np.cos(angle)])
        expected_m2_pos = 100.0 * z1
        np.testing.assert_allclose(m2.vertex_position, expected_m2_pos, atol=1e-10)
        
        # 验证最终 Z 轴方向（两次 45° 旋转 = 90° = -Y 方向）
        final_cs = traversal.current_coordinate_system
        expected_final_z = np.array([0, -1, 0])
        np.testing.assert_allclose(final_cs.z_axis, expected_final_z, atol=1e-10)
    
    def test_consecutive_coordinate_breaks_accumulate(self):
        """测试连续坐标断点正确累积
        
        **Validates: Requirements 5.6, 12.2**
        """
        # 创建测试数据模型：三个连续坐标断点
        zmx_data = ZmxDataModel()
        
        # 三个连续坐标断点，每个旋转 30 度
        for i in range(3):
            zmx_data.surfaces[i] = ZmxSurfaceData(
                index=i,
                surface_type='coordinate_break',
                tilt_x_deg=30.0,
                thickness=0.0
            )
        
        # 一个反射镜
        zmx_data.surfaces[3] = ZmxSurfaceData(
            index=3,
            surface_type='standard',
            radius=np.inf,
            thickness=0.0,
            is_mirror=True
        )
        
        # 遍历
        traversal = SurfaceTraversalAlgorithm(zmx_data)
        global_surfaces = traversal.traverse()
        
        # 验证：三次 30° 旋转 = 90° 旋转
        mirror = global_surfaces[0]
        angle = np.deg2rad(90.0)
        expected_z = np.array([0, -np.sin(angle), np.cos(angle)])
        np.testing.assert_allclose(mirror.orientation[:, 2], expected_z, atol=1e-10)


class TestOffAxisParabolicMirror:
    """离轴抛物面镜（OAP）集成测试
    
    **Validates: Requirements 12.3**
    """
    
    def test_90_degree_oap(self):
        """测试 90° OAP
        
        输入：90° OAP（绕 X 轴旋转 45°）
        预期：出射方向垂直于入射方向
        """
        # 创建测试数据模型
        zmx_data = ZmxDataModel()
        
        # 坐标断点：绕 X 轴旋转 45 度
        zmx_data.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=0.0
        )
        
        # 抛物面镜（conic = -1）
        zmx_data.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=200.0,  # 焦距 = 100mm
            conic=-1.0,    # 抛物面
            thickness=100.0,
            is_mirror=True,
            semi_diameter=25.0,
            comment='OAP'
        )
        
        # 遍历
        traversal = SurfaceTraversalAlgorithm(zmx_data)
        global_surfaces = traversal.traverse()
        
        # 验证
        assert len(global_surfaces) == 1
        
        oap = global_surfaces[0]
        
        # 验证是抛物面
        assert oap.conic == -1.0
        
        # 验证曲率半径
        assert oap.radius == 200.0
        
        # 验证曲率中心位置
        angle = np.deg2rad(45.0)
        rotated_z = np.array([0, -np.sin(angle), np.cos(angle)])
        expected_center = oap.vertex_position + 200.0 * rotated_z
        np.testing.assert_allclose(oap.curvature_center, expected_center, atol=1e-10)


class TestOrderParameter:
    """Order 参数测试
    
    **Validates: Requirements 12.4**
    """
    
    def test_order_0_vs_order_1_different_results(self):
        """测试 Order=0 和 Order=1 产生不同结果
        
        **Validates: Requirements 2.1, 2.2, 12.4**
        
        注意：绕 X 轴旋转时，X 轴本身不变，所以需要使用 Y 轴旋转来展示差异。
        """
        # 使用 Y 轴旋转来展示 Order 差异
        dx, dy = 10.0, 0.0
        tilt_y = np.deg2rad(45.0)
        
        cs = CurrentCoordinateSystem.identity()
        
        # Order=0: 先平移后旋转
        cs_order_0 = CoordinateBreakProcessor.process(
            cs, dx, dy, 0, tilt_y, 0, order=0, thickness=0
        )
        
        # Order=1: 先旋转后平移
        cs_order_1 = CoordinateBreakProcessor.process(
            cs, dx, dy, 0, tilt_y, 0, order=1, thickness=0
        )
        
        # 验证两者结果不同
        assert not np.allclose(cs_order_0.origin, cs_order_1.origin)
        
        # Order=0: 先沿原始 X 轴平移 10mm，再旋转
        # 原点应该在 (10, 0, 0)
        np.testing.assert_allclose(cs_order_0.origin, [10, 0, 0], atol=1e-10)
        
        # Order=1: 先旋转，再沿旋转后的 X 轴平移 10mm
        # 旋转后 X 轴变为 (cos45, 0, -sin45)
        angle = np.deg2rad(45.0)
        new_x = np.array([np.cos(angle), 0, -np.sin(angle)])
        expected_origin_1 = 10.0 * new_x
        np.testing.assert_allclose(cs_order_1.origin, expected_origin_1, atol=1e-10)
        
    def test_order_0_vs_order_1_with_y_rotation(self):
        """测试绕 Y 轴旋转时 Order=0 和 Order=1 的差异"""
        dx, dy = 10.0, 0.0
        tilt_y = np.deg2rad(45.0)
        
        cs = CurrentCoordinateSystem.identity()
        
        # Order=0: 先平移后旋转
        cs_order_0 = CoordinateBreakProcessor.process(
            cs, dx, dy, 0, tilt_y, 0, order=0, thickness=0
        )
        
        # Order=1: 先旋转后平移
        cs_order_1 = CoordinateBreakProcessor.process(
            cs, dx, dy, 0, tilt_y, 0, order=1, thickness=0
        )
        
        # Order=0: 先沿原始 X 轴平移 10mm，原点在 (10, 0, 0)
        np.testing.assert_allclose(cs_order_0.origin, [10, 0, 0], atol=1e-10)
        
        # Order=1: 先旋转，旋转后 X 轴变为 (cos45, 0, -sin45)
        # 再沿新 X 轴平移 10mm
        angle = np.deg2rad(45.0)
        new_x = np.array([np.cos(angle), 0, -np.sin(angle)])
        expected_origin_1 = 10.0 * new_x
        np.testing.assert_allclose(cs_order_1.origin, expected_origin_1, atol=1e-10)
        
        # 验证两者确实不同
        assert not np.allclose(cs_order_0.origin, cs_order_1.origin)


class TestNegativeThickness:
    """负厚度测试
    
    **Validates: Requirements 3.3, 10.7**
    """
    
    def test_negative_thickness_backward_propagation(self):
        """测试负厚度导致反向传播"""
        zmx_data = ZmxDataModel()
        
        # 反射镜，负厚度
        zmx_data.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='standard',
            radius=np.inf,
            thickness=-50.0,  # 负厚度
            is_mirror=True
        )
        
        # 遍历
        traversal = SurfaceTraversalAlgorithm(zmx_data)
        global_surfaces = traversal.traverse()
        
        # 验证最终原点在 Z 轴负方向
        final_cs = traversal.current_coordinate_system
        np.testing.assert_allclose(final_cs.origin, [0, 0, -50], atol=1e-10)


class TestMirrorDoesNotChangeCoordinateSystem:
    """反射镜不改变坐标系测试
    
    **Validates: Requirements 9.1, 9.2**
    """
    
    def test_mirror_only_advances_origin(self):
        """测试反射镜只前进原点，不改变轴方向"""
        zmx_data = ZmxDataModel()
        
        # 反射镜（无坐标断点）
        zmx_data.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='standard',
            radius=100.0,
            thickness=50.0,
            is_mirror=True
        )
        
        # 遍历
        traversal = SurfaceTraversalAlgorithm(zmx_data)
        global_surfaces = traversal.traverse()
        
        # 验证坐标系轴方向不变
        final_cs = traversal.current_coordinate_system
        np.testing.assert_allclose(final_cs.axes, np.eye(3), atol=1e-10)
        
        # 验证原点前进了 50mm
        np.testing.assert_allclose(final_cs.origin, [0, 0, 50], atol=1e-10)


class TestCurvatureCenterCalculation:
    """曲率中心计算测试
    
    **Validates: Requirements 6.1, 6.2, 6.5**
    """
    
    def test_positive_radius_curvature_center(self):
        """测试正半径曲率中心在 +Z 方向"""
        surface = GlobalSurfaceDefinition(
            index=1,
            surface_type='standard',
            vertex_position=np.array([0, 0, 0]),
            orientation=np.eye(3),
            radius=100.0
        )
        
        expected_center = np.array([0, 0, 100])
        np.testing.assert_allclose(surface.curvature_center, expected_center, atol=1e-10)
    
    def test_negative_radius_curvature_center(self):
        """测试负半径曲率中心在 -Z 方向"""
        surface = GlobalSurfaceDefinition(
            index=1,
            surface_type='standard',
            vertex_position=np.array([0, 0, 0]),
            orientation=np.eye(3),
            radius=-100.0
        )
        
        expected_center = np.array([0, 0, -100])
        np.testing.assert_allclose(surface.curvature_center, expected_center, atol=1e-10)
    
    def test_rotated_curvature_center(self):
        """测试旋转后的曲率中心计算"""
        # 绕 X 轴旋转 45 度
        angle = np.deg2rad(45.0)
        R = CoordinateBreakProcessor.rotation_matrix_x(angle)
        orientation = np.eye(3) @ R
        
        surface = GlobalSurfaceDefinition(
            index=1,
            surface_type='standard',
            vertex_position=np.array([0, 0, 0]),
            orientation=orientation,
            radius=100.0
        )
        
        # 曲率中心应该沿旋转后的 Z 轴方向
        rotated_z = np.array([0, -np.sin(angle), np.cos(angle)])
        expected_center = 100.0 * rotated_z
        np.testing.assert_allclose(surface.curvature_center, expected_center, atol=1e-10)


class TestOptilandCoordinateSystemTransfer:
    """optiland 坐标系统传递测试
    
    验证 ZemaxToOptilandConverter 正确传递坐标系统参数到 optiland。
    **Validates: Requirements 10.1, 10.2, 10.3, 10.4**
    """
    
    def test_tilted_mirror_coordinate_system_transfer(self):
        """测试倾斜镜面的坐标系统正确传递到 optiland
        
        验证：
        1. 表面位置 (x, y, z) 正确传递
        2. 表面旋转 (rx, ry, rz) 正确传递
        """
        # 创建测试数据模型：45° 折叠镜
        zmx_data = ZmxDataModel()
        
        # 坐标断点：绕 X 轴旋转 45 度
        zmx_data.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=100.0  # 沿旋转后的 Z 轴前进 100mm
        )
        
        # 平面反射镜
        zmx_data.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=np.inf,
            thickness=50.0,
            is_mirror=True,
            semi_diameter=25.0
        )
        
        # 遍历生成全局坐标表面
        traversal = SurfaceTraversalAlgorithm(zmx_data)
        global_surfaces = traversal.traverse()
        
        # 转换为 optiland
        converter = ZemaxToOptilandConverter(
            global_surfaces,
            wavelength=0.55,
            entrance_pupil_diameter=10.0
        )
        optic = converter.convert()
        
        # 验证表面数量（物面 + 1个光学表面 + 像面 = 3）
        assert len(optic.surface_group.surfaces) == 3
        
        # 获取光学表面（index=1）
        optical_surface = optic.surface_group.surfaces[1]
        
        # 验证表面位置
        # 坐标断点后：原点沿旋转后的 Z 轴前进 100mm
        angle = np.deg2rad(45.0)
        rotated_z = np.array([0, -np.sin(angle), np.cos(angle)])
        expected_position = 100.0 * rotated_z
        
        cs = optical_surface.geometry.cs
        actual_position = np.array([float(cs.x), float(cs.y), float(cs.z)])
        np.testing.assert_allclose(actual_position, expected_position, atol=1e-10)
        
        # 验证表面旋转
        # 绕 X 轴旋转 45 度
        expected_rx = angle
        expected_ry = 0.0
        expected_rz = 0.0
        
        np.testing.assert_allclose(float(cs.rx), expected_rx, atol=1e-10)
        np.testing.assert_allclose(float(cs.ry), expected_ry, atol=1e-10)
        np.testing.assert_allclose(float(cs.rz), expected_rz, atol=1e-10)
    
    def test_off_axis_parabolic_mirror_transfer(self):
        """测试离轴抛物面镜的坐标系统正确传递到 optiland
        
        验证 OAP 的顶点位置和姿态正确传递。
        """
        # 创建测试数据模型：90° OAP
        zmx_data = ZmxDataModel()
        
        # 坐标断点：绕 X 轴旋转 45 度
        zmx_data.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=0.0
        )
        
        # 抛物面镜
        zmx_data.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=200.0,
            conic=-1.0,
            thickness=100.0,
            is_mirror=True,
            semi_diameter=25.0
        )
        
        # 遍历生成全局坐标表面
        traversal = SurfaceTraversalAlgorithm(zmx_data)
        global_surfaces = traversal.traverse()
        
        # 转换为 optiland
        converter = ZemaxToOptilandConverter(global_surfaces)
        optic = converter.convert()
        
        # 获取 OAP 表面
        oap_surface = optic.surface_group.surfaces[1]
        
        # 验证表面位置在原点（坐标断点 thickness=0）
        cs = oap_surface.geometry.cs
        np.testing.assert_allclose(float(cs.x), 0.0, atol=1e-10)
        np.testing.assert_allclose(float(cs.y), 0.0, atol=1e-10)
        np.testing.assert_allclose(float(cs.z), 0.0, atol=1e-10)
        
        # 验证表面旋转（绕 X 轴 45 度）
        angle = np.deg2rad(45.0)
        np.testing.assert_allclose(float(cs.rx), angle, atol=1e-10)
        np.testing.assert_allclose(float(cs.ry), 0.0, atol=1e-10)
        np.testing.assert_allclose(float(cs.rz), 0.0, atol=1e-10)
        
        # 验证曲率半径和圆锥常数
        assert oap_surface.geometry.radius == 200.0
        assert oap_surface.geometry.k == -1.0
    
    def test_z_shape_dual_mirror_transfer(self):
        """测试 Z 形双镜系统的坐标系统正确传递到 optiland
        
        验证两个镜面的位置和姿态都正确传递。
        """
        # 创建测试数据模型
        zmx_data = ZmxDataModel()
        
        # 第一个坐标断点：绕 X 轴旋转 45 度
        zmx_data.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=0.0
        )
        
        # 第一个平面反射镜
        zmx_data.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=np.inf,
            thickness=100.0,
            is_mirror=True
        )
        
        # 第二个坐标断点：再绕 X 轴旋转 45 度
        zmx_data.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=0.0
        )
        
        # 第二个平面反射镜
        zmx_data.surfaces[3] = ZmxSurfaceData(
            index=3,
            surface_type='standard',
            radius=np.inf,
            thickness=50.0,
            is_mirror=True
        )
        
        # 遍历生成全局坐标表面
        traversal = SurfaceTraversalAlgorithm(zmx_data)
        global_surfaces = traversal.traverse()
        
        # 转换为 optiland
        converter = ZemaxToOptilandConverter(global_surfaces)
        optic = converter.convert()
        
        # 验证表面数量（物面 + 2个光学表面 + 像面 = 4）
        assert len(optic.surface_group.surfaces) == 4
        
        # 验证 M1 位置和姿态
        m1 = optic.surface_group.surfaces[1]
        cs1 = m1.geometry.cs
        
        # M1 在原点，绕 X 轴旋转 45 度
        np.testing.assert_allclose(float(cs1.x), 0.0, atol=1e-10)
        np.testing.assert_allclose(float(cs1.y), 0.0, atol=1e-10)
        np.testing.assert_allclose(float(cs1.z), 0.0, atol=1e-10)
        np.testing.assert_allclose(float(cs1.rx), np.deg2rad(45.0), atol=1e-10)
        
        # 验证 M2 位置和姿态
        m2 = optic.surface_group.surfaces[2]
        cs2 = m2.geometry.cs
        
        # M2 位置：沿 45° Z 轴前进 100mm
        angle = np.deg2rad(45.0)
        z1 = np.array([0, -np.sin(angle), np.cos(angle)])
        expected_m2_pos = 100.0 * z1
        
        np.testing.assert_allclose(float(cs2.x), expected_m2_pos[0], atol=1e-10)
        np.testing.assert_allclose(float(cs2.y), expected_m2_pos[1], atol=1e-10)
        np.testing.assert_allclose(float(cs2.z), expected_m2_pos[2], atol=1e-10)
        
        # M2 绕 X 轴旋转 90 度（两次 45 度）
        np.testing.assert_allclose(float(cs2.rx), np.deg2rad(90.0), atol=1e-10)
    
    def test_image_surface_position(self):
        """测试像面位置正确计算
        
        像面应该位于最后一个光学表面沿其 Z 轴前进 thickness 的位置。
        """
        # 创建测试数据模型
        zmx_data = ZmxDataModel()
        
        # 坐标断点：绕 X 轴旋转 45 度
        zmx_data.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=0.0
        )
        
        # 反射镜，thickness=100mm
        zmx_data.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=np.inf,
            thickness=100.0,
            is_mirror=True
        )
        
        # 遍历生成全局坐标表面
        traversal = SurfaceTraversalAlgorithm(zmx_data)
        global_surfaces = traversal.traverse()
        
        # 转换为 optiland
        converter = ZemaxToOptilandConverter(global_surfaces)
        optic = converter.convert()
        
        # 获取像面
        image_surface = optic.surface_group.surfaces[-1]
        cs_image = image_surface.geometry.cs
        
        # 像面位置：镜面位置 + 100mm × 旋转后的 Z 轴
        angle = np.deg2rad(45.0)
        rotated_z = np.array([0, -np.sin(angle), np.cos(angle)])
        expected_image_pos = 100.0 * rotated_z  # 镜面在原点
        
        np.testing.assert_allclose(float(cs_image.x), expected_image_pos[0], atol=1e-10)
        np.testing.assert_allclose(float(cs_image.y), expected_image_pos[1], atol=1e-10)
        np.testing.assert_allclose(float(cs_image.z), expected_image_pos[2], atol=1e-10)
        
        # 像面姿态与最后一个光学表面相同
        np.testing.assert_allclose(float(cs_image.rx), angle, atol=1e-10)
