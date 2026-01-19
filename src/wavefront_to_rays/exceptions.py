"""
波前到光线模块异常类定义

本模块定义了波前到光线转换和复振幅重建过程中使用的异常类。
所有异常都继承自 ReconstructionError 基类，并提供中文错误信息。

异常类层次：
- ReconstructionError（基类）
  - InsufficientRaysError（有效光线数量不足）

使用示例：
    >>> from wavefront_to_rays.exceptions import InsufficientRaysError
    >>> raise InsufficientRaysError(
    ...     f"有效光线数量不足：3 < 4。"
    ...     f"无法进行复振幅重建，请检查光学系统配置或增加采样光线数量。"
    ... )
"""


class ReconstructionError(Exception):
    """复振幅重建错误基类
    
    所有复振幅重建相关异常的基类。
    用于捕获所有与光线到波前重建相关的错误。
    
    属性:
        message: 错误信息（中文）
    
    示例:
        >>> try:
        ...     # 某些重建操作
        ...     pass
        ... except ReconstructionError as e:
        ...     print(f"复振幅重建错误: {e}")
    """
    
    def __init__(self, message: str) -> None:
        """初始化异常
        
        参数:
            message: 错误信息（中文）
        """
        self.message = message
        super().__init__(message)


class InsufficientRaysError(ReconstructionError):
    """有效光线数量不足错误
    
    当有效光线数量少于最小要求（4 条）时抛出此异常。
    复振幅重建需要至少 4 条有效光线才能进行插值和雅可比矩阵计算。
    
    对应需求: 需求 6.1
    
    常见触发条件：
    - 光学元件遮挡了大部分光线
    - 采样范围设置不当
    - 光线追迹过程中大量光线失效
    
    建议解决方案：
    - 增加采样光线数量
    - 检查光学系统配置
    - 调整采样范围
    
    示例:
        >>> valid_count = 3
        >>> min_required = 4
        >>> raise InsufficientRaysError(
        ...     f"有效光线数量不足：{valid_count} < {min_required}。"
        ...     f"无法进行复振幅重建，请检查光学系统配置或增加采样光线数量。"
        ... )
    """
    pass

