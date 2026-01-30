import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from sequential_system.zmx_converter import CoordinateTransform

class TestZmxOrder(unittest.TestCase):
    def test_identity(self):
        ct = CoordinateTransform()
        self.assertTrue(np.allclose(ct.matrix, np.eye(4)))
        self.assertTrue(np.isclose(ct.decenter_x, 0))
        self.assertTrue(np.isclose(ct.tilt_x_rad, 0))

    def test_order_0_decenter_then_tilt(self):
        # Order 0: Decenter then Tilt
        # Apply pure decenter then pure tilt in one step
        # M = R @ T
        ct = CoordinateTransform()
        ct.apply_coordinate_break(
            dx=10, dy=0, dz=0,
            rx_deg=90, ry_deg=0, rz_deg=0,
            order=0
        )
        
        # Expected:
        # T = translation(10, 0, 0)
        # R = rot_x(90) -> y->z, z->-y
        # M = R @ T
        # P_in = (0,0,0) -> P_out = R(T(0)) = R(10, 0, 0) = (10, 0, 0) because rot x doesn't affect x
        
        # Let's check decenter properties
        self.assertTrue(np.isclose(ct.decenter_x, 10.0))
        self.assertTrue(np.isclose(ct.decenter_y, 0.0))
        self.assertTrue(np.isclose(ct.decenter_z, 0.0))
        
        # Tilt should be 90 deg x
        self.assertTrue(np.isclose(ct.tilt_x_rad, np.deg2rad(90)))

    def test_order_1_tilt_then_decenter(self):
        # Order 1: Tilt then Decenter
        # M = T @ R
        ct = CoordinateTransform()
        ct.apply_coordinate_break(
            dx=10, dy=0, dz=0,
            rx_deg=90, ry_deg=0, rz_deg=0,
            order=1
        )
        
        # Expected:
        # Tilt occurs at origin, THEN decenter (10,0,0) is added in the NEW frame? 
        # Wait, standard interpretation:
        # The decenter is applied AFTER rotation.
        # But CoordinateTransform usually tracks Global Position relative to start.
        # Or relative to component local?
        
        # With T @ R:
        # [ I D ] [ R 0 ]   [ R  D ]
        # [ 0 1 ] [ 0 1 ] = [ 0  1 ]
        # Decenter is simply added to the translation column.
        
        self.assertTrue(np.isclose(ct.decenter_x, 10.0))
        self.assertTrue(np.isclose(ct.tilt_x_rad, np.deg2rad(90)))
        
        # Wait, if result is same for pure x decenter and pure x tilt, try mixed.
        # Try dy=10, rx=90.
        
    def test_order_difference(self):
        # Case: dy=10, rx=90
        # Order 0 (Decenter then Tilt):
        # T(0, 10, 0) -> P=(0, 10, 0)
        # R_x(90): (x, y, z) -> (x, -z, y)
        # R(0, 10, 0) -> (0, 0, 10)
        # So final decenter should be (0, 0, 10)
        
        ct0 = CoordinateTransform()
        ct0.apply_coordinate_break(
            dx=0, dy=10, dz=0,
            rx_deg=90, ry_deg=0, rz_deg=0,
            order=0
        )
        # Verify
        # R_x(90) = [[1,0,0],[0,0,-1],[0,1,0]]
        # T = [0, 10, 0]
        # R @ T = [0, -0, 10] = [0, 0, 10]
        self.assertTrue(np.isclose(ct0.decenter_z, 10.0), f"Order 0: Expected dz=10, got {ct0.decenter_z}")
        self.assertTrue(np.isclose(ct0.decenter_y, 0.0), f"Order 0: Expected dy=0, got {ct0.decenter_y}")

        # Order 1 (Tilt then Decenter):
        # R_x(90) first.
        # Then T(0, 10, 0).
        # M = T @ R = [ R  D ]
        # Translation part is just D = (0, 10, 0)
        ct1 = CoordinateTransform()
        ct1.apply_coordinate_break(
            dx=0, dy=10, dz=0,
            rx_deg=90, ry_deg=0, rz_deg=0,
            order=1
        )
        self.assertTrue(np.isclose(ct1.decenter_y, 10.0), f"Order 1: Expected dy=10, got {ct1.decenter_y}")
        self.assertTrue(np.isclose(ct1.decenter_z, 0.0), f"Order 1: Expected dz=0, got {ct1.decenter_z}")

    def test_accumulation(self):
        # Test accumulation of transforms
        ct = CoordinateTransform()
        # 1. Move Z+10
        ct.apply_coordinate_break(0,0,10, 0,0,0, order=0)
        self.assertTrue(np.isclose(ct.decenter_z, 10.0))
        
        # 2. Tilt X 90 (at new position)
        # Order 0: Decenter (0) then Tilt (90)
        # M_new = R_x(90)
        # M_total = M_old @ M_new
        # [ I  Z10 ] @ [ Rx  0 ] = [ Rx  Z10 ]
        # [ 0   1  ]   [ 0   1 ]   [ 0    1  ]
        # Position should remain (0,0,10), Orientation changed.
        ct.apply_coordinate_break(0,0,0, 90,0,0, order=0)
        self.assertTrue(np.isclose(ct.decenter_z, 10.0))
        self.assertTrue(np.isclose(ct.tilt_x_rad, np.deg2rad(90)))
        
        # 3. Move Z+10 (in new local frame)
        # New Z axis is old -Y axis.
        # So moving +z in local frame should decrease Y in global frame?
        # Global: (0,0,10). Local Z points to Global -Y.
        # Move +10 in local Z -> Move -10 in Global Y.
        # Result Global: (0, -10, 10).
        
        ct.apply_coordinate_break(0,0,10, 0,0,0, order=0)
        # M_new_2 = T(0,0,10)
        # M_total = [ Rx  Z10 ] @ [ I  Z10 ] = [ Rx   Rx*Z10 + Z10 ]
        #                                       [ 0          1       ]
        # Rx * [0,0,10]^T = [0, -10, 0]^T
        # Pos = [0, -10, 0] + [0, 0, 10] = [0, -10, 10]
        
        self.assertTrue(np.isclose(ct.decenter_x, 0))
        self.assertTrue(np.isclose(ct.decenter_y, -10.0), f"Got {ct.decenter_y}")
        self.assertTrue(np.isclose(ct.decenter_z, 10.0), f"Got {ct.decenter_z}")

if __name__ == '__main__':
    unittest.main()
