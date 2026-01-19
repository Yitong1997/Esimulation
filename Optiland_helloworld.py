import optiland
from optiland.samples.objectives import ReverseTelephoto
import matplotlib.pyplot as plt
lens = ReverseTelephoto()
lens.draw3D()
lens.draw()
plt.show()