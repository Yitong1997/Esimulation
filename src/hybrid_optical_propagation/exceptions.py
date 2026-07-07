"""
混合光学传播系统异常类

定义混合光学传播系统中使用的所有异常类型。

**Validates: Requirements 14.1-14.6**
"""


class HybridPropagationError(Exception):
    """混合传播基础异常
    
    所有混合光学传播相关异常的基类。
    """
    pass


class RayTracingError(HybridPropagationError):
    """光线追迹错误
    
    当光线追迹失败时抛出，例如：
    - 光线与表面无交点
    - 有效光线数量不足
    
    **Validates: Requirements 14.1, 14.2**
    """
    pass


class PhaseUnwrappingError(HybridPropagationError):
    """相位解包裹错误
    
    当相位解包裹失败或残差过大时抛出。
    
    **Validates: Requirements 14.3**
    """
    pass


class MaterialError(HybridPropagationError):
    """材质相关错误
    
    当材质折射率无效或材质定义错误时抛出。
    
    **Validates: Requirements 14.5**
    """
    pass


class GridSamplingError(HybridPropagationError):
    """网格采样错误
    
    当网格尺寸不匹配或采样参数无效时抛出。
    
    **Validates: Requirements 17.6**
    """
    pass


class InsufficientRaysError(RayTracingError):
    """有效光线数量不足错误
    
    当有效光线数量少于最小要求时抛出。
    
    **Validates: Requirements 14.2**
    """
    
    def __init__(self, valid_count: int, min_required: int = 4):
        self.valid_count = valid_count
        self.min_required = min_required
        message = (
            f"有效光线数量不足：{valid_count} < {min_required}。"
            f"建议增加采样密度或检查光学系统配置。"
        )
        super().__init__(message)


class NoIntersectionError(RayTracingError):
    """光线无交点错误
    
    当光线与表面无交点时抛出。
    
    **Validates: Requirements 14.1**
    """
    
    def __init__(self, surface_index: int, ray_position: tuple = None):
        self.surface_index = surface_index
        self.ray_position = ray_position
        if ray_position:
            message = (
                f"光线追迹失败：光线在位置 {ray_position} 处与表面 {surface_index} 无交点。"
            )
        else:
            message = f"光线追迹失败：光线与表面 {surface_index} 无交点。"
        super().__init__(message)
