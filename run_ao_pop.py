"""使用 ao_core 模块对 biconic_focus_test.zmx 执行 POP 仿真并可视化"""

import matplotlib.pyplot as plt
from ao_core import init_system, get_pop_field, plot_wavefront, cleanup

zos, oss = init_system("Zemax_baseline/biconic_focus_test.zmx")

amplitude, phase, extent_info = get_pop_field(oss,sampling = 512, beam_width=128.0)

fig = plot_wavefront(amplitude, phase, title="POP 仿真结果", extent_info=extent_info)
plt.show()

cleanup(zos)
