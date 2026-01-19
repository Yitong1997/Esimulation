"""
混合元件传播测试模块

本模块包含 hybrid_propagation 模块的所有测试，包括：
- TiltedPropagation 倾斜平面传播测试
- PilotBeamValidator Pilot Beam 验证器测试
- PilotBeamCalculator Pilot Beam 计算器测试
- PhaseCorrector 相位修正器测试
- AmplitudeReconstructor 复振幅重建器测试
- HybridElementPropagator 混合元件传播器测试
- 集成测试

测试策略：
- 单元测试：验证各组件的独立功能
- 属性测试：使用 Hypothesis 验证通用属性
- 集成测试：验证完整的传播流程
"""
