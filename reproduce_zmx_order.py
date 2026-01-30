
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from sequential_system.zmx_parser import ZmxParser

# Create a temporary ZMX file with a Coordinate Break and PARM 6 (Order)
zmx_content = """
MODE SEQ
SURF 0
  TYPE STANDARD
SURF 1
  TYPE COORDBRK
  PARM 1 1.0
  PARM 2 2.0
  PARM 3 10.0
  PARM 4 20.0
  PARM 5 30.0
  PARM 6 1.0  
  COMM Test Coordinate Break
SURF 2
  TYPE STANDARD
"""

with open('temp_test.zmx', 'w') as f:
    f.write(zmx_content)

try:
    parser = ZmxParser('temp_test.zmx')
    model = parser.parse()
    
    cb_surface = model.get_surface(1)
    print(f"Surface 1 Type: {cb_surface.surface_type}")
    print(f"Decenter X: {cb_surface.decenter_x}")
    print(f"Decenter Y: {cb_surface.decenter_y}")
    print(f"Tilt X: {cb_surface.tilt_x_deg}")
    print(f"Tilt Y: {cb_surface.tilt_y_deg}")
    print(f"Tilt Z: {cb_surface.tilt_z_deg}")
    
    # Check if 'order' attribute exists and is set correctly
    if hasattr(cb_surface, 'order'):
        print(f"Order: {cb_surface.order}")
    else:
        print("Attribute 'order' NOT FOUND on ZmxSurfaceData")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if os.path.exists('temp_test.zmx'):
        os.remove('temp_test.zmx')
