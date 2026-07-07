"""
统一仿真入口异常类

定义仿真过程中可能抛出的异常。
"""


class ConfigurationError(Exception):
    """配置错误
    
    当仿真配置不完整或无效时抛出。
    
    示例：
        - 未定义光学系统
        - 未定义光源
        - 参数超出有效范围
    """
    pass


class SimulationError(Exception):
    """仿真错误
    
    当仿真执行过程中发生错误时抛出。
    
    示例：
        - 光线追迹失败
        - 传播计算失败
        - 数值溢出
    """
    pass
