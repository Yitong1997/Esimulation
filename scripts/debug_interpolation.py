"""调试插值问题"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# 创建简单的测试数据
grid_size = 5
physical_size = 10.0
half_size = physical_size / 2.0

coords = np.linspace(-half_size, half_size, grid_size)
print(f"坐标: {coords}")

X, Y = np.meshgrid(coords, coords)
print(f"\nX 网格:\n{X}")
print(f"\nY 网格:\n{Y}")

# 创建相位网格：φ = x + 2*y
# 这样可以验证 x 和 y 的对应关系
phase = X + 2 * Y
print(f"\n相位网格 (φ = x + 2*y):\n{phase}")

# 创建插值器
# RegularGridInterpolator 期望的格式：
# - 第一个参数是 (y_coords, x_coords)
# - 数据的形状是 (len(y_coords), len(x_coords))
# - 查询点的格式是 (y, x)

interpolator = RegularGridInterpolator(
    (coords, coords),  # (y_coords, x_coords)
    phase,
    method='linear',
    bounds_error=False,
    fill_value=0.0,
)

# 测试点
test_points = [
    (0, 0),      # 中心，期望 0 + 0 = 0
    (2.5, 0),    # y=2.5, x=0，期望 0 + 5 = 5
    (0, 2.5),    # y=0, x=2.5，期望 2.5 + 0 = 2.5
    (-2.5, 2.5), # y=-2.5, x=2.5，期望 2.5 - 5 = -2.5
]

print("\n插值测试（查询格式：(y, x)）：")
for y, x in test_points:
    result = interpolator([[y, x]])[0]
    expected = x + 2 * y
    print(f"  点 (y={y}, x={x}): 插值={result:.4f}, 期望={expected:.4f}, 差异={result-expected:.4f}")

# 如果上面的测试失败，尝试交换 x 和 y
print("\n插值测试（查询格式：(x, y)）：")
for y, x in test_points:
    result = interpolator([[x, y]])[0]
    expected = x + 2 * y
    print(f"  点 (y={y}, x={x}): 插值={result:.4f}, 期望={expected:.4f}, 差异={result-expected:.4f}")
