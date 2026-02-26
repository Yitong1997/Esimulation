"""
序列光学系统异常类定义

本模块定义了序列光学系统中使用的异常类和警告类层次结构。
所有异常都继承自 SequentialSystemError 基类，并提供中文错误信息。

异常类层次：
- SequentialSystemError（基类）
  - SourceConfigurationError（光源配置错误）
  - SurfaceConfigurationError（光学面配置错误）
  - SimulationError（仿真执行错误）
  - SamplingError（采样面错误）

警告类：
- PilotBeamWarning（Pilot Beam 适用性警告）
  - 继承自 UserWarning
  - 用于混合元件传播中的 Pilot Beam 方法适用性检测

使用示例：
    >>> from sequential_system.exceptions import SourceConfigurationError
    >>> raise SourceConfigurationError(
    ...     f"光源参数 'wavelength' 无效：期望正值，实际为 -0.5 μm。"
    ...     f"请确保波长为正有限值。"
    ... )
"""


class SequentialSystemError(Exception):
    """序列光学系统基础异常
    
    所有序列光学系统相关异常的基类。
    用于捕获所有与序列光学系统相关的错误。
    
    属性:
        message: 错误信息（中文）
    
    示例:
        >>> try:
        ...     # 某些操作
        ...     pass
        ... except SequentialSystemError as e:
        ...     print(f"序列光学系统错误: {e}")
    """
    
    def __init__(self, message: str) -> None:
        """初始化异常
        
        参数:
            message: 错误信息（中文）
        """
        self.message = message
        super().__init__(message)


class SourceConfigurationError(SequentialSystemError):
    """光源配置错误
    
    当光源参数无效时抛出此异常。
    
    常见触发条件：
    - 波长为负值、零或无穷大
    - 束腰半径为负值、零或无穷大
    - M² 因子小于 1.0
    - 参数为 NaN
    
    示例:
        >>> wavelength = -0.633
        >>> raise SourceConfigurationError(
        ...     f"光源参数 'wavelength' 无效：期望正值，实际为 {wavelength} μm。"
        ...     f"请确保波长为正有限值。"
        ... )
    """
    pass


class SurfaceConfigurationError(SequentialSystemError):
    """光学面配置错误
    
    当光学面参数无效时抛出此异常。
    
    常见触发条件：
    - 半口径为负值或零
    - 厚度为负值（某些情况下）
    - 曲率半径为零
    - 材料名称无效
    - 倾斜角度超出合理范围
    
    示例:
        >>> index = 1
        >>> name = "M1"
        >>> semi_aperture = -5.0
        >>> raise SurfaceConfigurationError(
        ...     f"光学面 {index} ('{name}') 配置错误：半口径 ({semi_aperture} mm) "
        ...     f"必须为正值。"
        ... )
    """
    pass


class SimulationError(SequentialSystemError):
    """仿真执行错误
    
    当仿真过程中发生错误时抛出此异常。
    
    常见触发条件：
    - 波前传播过程中的数值问题
    - 光线追迹失败
    - 采样网格不足
    - 内存不足
    - PROPER 或 optiland 库内部错误
    
    示例:
        >>> path_length = 150.5
        >>> z_position = 75.2
        >>> reason = "波前振幅出现 NaN 值"
        >>> raise SimulationError(
        ...     f"仿真在光程距离 {path_length:.2f} mm 处失败：{reason}。"
        ...     f"当前位置 z = {z_position:.2f} mm。"
        ... )
    """
    pass


class SamplingError(SequentialSystemError):
    """采样面错误
    
    当采样面配置或数据有问题时抛出此异常。
    
    常见触发条件：
    - 采样面位置超出光路范围
    - 采样面位置为负值
    - 采样面名称重复
    - 采样数据无效
    
    示例:
        >>> distance = 500.0
        >>> max_distance = 300.0
        >>> raise SamplingError(
        ...     f"采样面位置 ({distance:.2f} mm) 超出光路范围。"
        ...     f"有效范围为 0 到 {max_distance:.2f} mm。"
        ... )
    """
    pass


# =============================================================================
# 警告类
# =============================================================================


class PilotBeamWarning(UserWarning):
    """Pilot Beam 适用性警告
    
    当 Pilot Beam 方法的适用性条件不满足时发出此警告。
    警告不会中断程序执行，但提示用户可能需要调整参数。
    
    常见触发条件：
    - 相邻像素相位差超过 π/2（采样不足）
      建议：增加网格分辨率
    - 光束发散角过大（超过设定阈值）
      建议：使用更小的传播步长
    - Pilot Beam 与实际光束尺寸差异超过 50%
      建议：调整 Pilot Beam 参数
    
    使用示例:
        >>> import warnings
        >>> from sequential_system.exceptions import PilotBeamWarning
        >>> 
        >>> # 发出采样不足警告
        >>> max_phase_diff = 1.8  # 弧度，超过 π/2
        >>> warnings.warn(
        ...     f"Pilot Beam 相位采样不足：相邻像素最大相位差为 {max_phase_diff:.3f} 弧度，"
        ...     f"超过阈值 π/2 ({np.pi/2:.3f} 弧度)。"
        ...     f"建议增加网格分辨率以获得更准确的结果。",
        ...     PilotBeamWarning
        ... )
        >>> 
        >>> # 发出发散角过大警告
        >>> divergence = 0.15  # 弧度
        >>> threshold = 0.1
        >>> warnings.warn(
        ...     f"光束发散角过大：{divergence:.4f} 弧度，超过阈值 {threshold:.4f} 弧度。"
        ...     f"建议使用更小的传播步长。",
        ...     PilotBeamWarning
        ... )
        >>> 
        >>> # 发出光束尺寸不匹配警告
        >>> pilot_size = 10.0  # mm
        >>> actual_size = 18.0  # mm
        >>> ratio = actual_size / pilot_size
        >>> warnings.warn(
        ...     f"Pilot Beam 与实际光束尺寸差异过大：比值为 {ratio:.2f}，超过 1.5。"
        ...     f"建议调整 Pilot Beam 参数以更好地匹配实际光束。",
        ...     PilotBeamWarning
        ... )
    
    捕获警告示例:
        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as w:
        ...     warnings.simplefilter("always")
        ...     # 执行可能产生警告的操作
        ...     for warning in w:
        ...         if issubclass(warning.category, PilotBeamWarning):
        ...             print(f"Pilot Beam 警告: {warning.message}")
    """
    pass
