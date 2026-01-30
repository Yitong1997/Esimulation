
import sys
import inspect
import os

# Add src to path
sys.path.insert(0, r'd:\BTS\src')

try:
    from wavefront_to_rays import element_raytracer
    print(f"Module file: {element_raytracer.__file__}")
    
    if hasattr(element_raytracer, 'compute_rotation_matrix'):
        func = element_raytracer.compute_rotation_matrix
        print(f"Function found: {func}")
        print(f"Source file: {inspect.getsourcefile(func)}")
        lines, lineno = inspect.getsourcelines(func)
        print(f"Line number: {lineno}")
        print("Source code:")
        print("".join(lines))
    else:
        print("compute_rotation_matrix not found in module")

except Exception as e:
    print(f"Error: {e}")
    
# Also write to a file to avoid encoding issues
with open('func_source.txt', 'w', encoding='utf-8') as f:
    if 'func' in locals():
        f.write(f"Line: {lineno}\n")
        f.write("".join(lines))
