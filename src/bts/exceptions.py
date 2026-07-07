"""
BTS 模块异常定义

定义 MATLAB 风格 API 中使用的异常类。
"""

# 重导出现有异常
from hybrid_simulation.exceptions import ConfigurationError, SimulationError


class ParseError(Exception):
    """解析错误
    
    当 ZMX 文件或其他配置文件解析失败时抛出。
    
    示例：
        - ZMX 文件格式无效
        - 缺少必要的表面定义
        - 参数值格式错误
    
    属性:
        message: 错误描述
        file_path: 相关文件路径（可选）
        line_number: 错误发生的行号（可选）
    """
    
    def __init__(
        self,
        message: str,
        file_path: str = None,
        line_number: int = None
    ):
        """创建解析错误异常
        
        参数:
            message: 错误描述
            file_path: 相关文件路径（可选）
            line_number: 错误发生的行号（可选）
        """
        self.message = message
        self.file_path = file_path
        self.line_number = line_number
        
        # 构建完整的错误信息
        full_message = message
        if file_path:
            full_message = f"{file_path}: {full_message}"
        if line_number is not None:
            full_message = f"{full_message} (行 {line_number})"
        
        super().__init__(full_message)


# 导出所有异常类
__all__ = [
    'ParseError',
    'ConfigurationError',
    'SimulationError',
]
