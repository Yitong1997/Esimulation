
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))
sys.path.insert(0, str(project_root / 'proper_v3.3.4_python'))

import bts

zmx_dir = project_root / 'optiland-master' / 'tests' / 'zemax_files'
zmx_file = zmx_dir / 'simple_fold_mirror_up.zmx'

print(f"Loading {zmx_file}")
system = bts.load_zmx(str(zmx_file))
system.print_info()
