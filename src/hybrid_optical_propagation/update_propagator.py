
import sys
import numpy as np

file_path = r'd:\BTS\src\hybrid_optical_propagation\hybrid_propagator.py'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Locate the function _compute_reflection_direction
start_line = -1
for i, line in enumerate(lines):
    if 'def _compute_reflection_direction' in line:
        start_line = i
        break

if start_line == -1:
    print('Function not found')
    sys.exit(1)

# Find the end of the function
end_line = -1
for i in range(start_line + 1, len(lines)):
    if line.strip().startswith('def '): # Careful with indentation
        pass
    
    # Simple heuristic: find the next 'def ' with same indentation or less?
    # Actually, the method is inside a class, so it has indentation.
    # The next method starts with '    def '
    if lines[i].startswith('    def '):
        end_line = i
        break
    if i == len(lines) - 1:
        end_line = len(lines)

# Reconstruct the function
# Note: Indentation must be preserved (4 spaces)
new_func_lines = [
    '    def _compute_reflection_direction(\n',
    '        self,\n',
    '        incident_dir: np.ndarray,\n',
    '        intersection: np.ndarray,\n',
    '        surface: "GlobalSurfaceDefinition",\n',
    '    ) -> np.ndarray:\n',
    '        """计算反射方向\n',
    '        \n',
    '        参数:\n',
    '            incident_dir: 入射方向（归一化）\n',
    '            intersection: 交点位置 (mm)\n',
    '            surface: 表面定义\n',
    '        \n',
    '        返回:\n',
    '            反射方向（归一化）\n',
    '        \n',
    '        注意:\n',
    '            已移除对抛物面的特殊处理。现在所有表面（包括离轴抛物面 OAP）\n',
    '            都遵循严格的几何反射定律，使用局部法向量计算反射方向。\n',
    '            这允许正确仿真对准误差和表面误差。\n',
    '        """\n',
    '        # 对于所有表面（包括抛物面），使用标准的几何法向量方法\n',
    '        # 计算表面法向量\n',
    '        normal = self._compute_surface_normal(intersection, surface)\n',
    '        \n',
    '        # 确保法向量指向入射光来的方向\n',
    '        if np.dot(normal, incident_dir) > 0:\n',
    '            normal = -normal\n',
    '        \n',
    '        # 反射公式：r = d - 2(d·n)n\n',
    '        dot_dn = np.dot(incident_dir, normal)\n',
    '        reflected_dir = incident_dir - 2 * dot_dn * normal\n',
    '        \n',
    '        # 归一化\n',
    '        norm = np.linalg.norm(reflected_dir)\n',
    '        if norm > 1e-10:\n',
    '            reflected_dir = reflected_dir / norm\n',
    '        \n',
    '        return reflected_dir\n',
    '\n'
]

# Replace the lines
new_lines = lines[:start_line] + new_func_lines + lines[end_line:]

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('Successfully updated hybrid_propagator.py')
