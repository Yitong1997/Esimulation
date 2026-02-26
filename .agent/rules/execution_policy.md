# 执行规则
**严禁**直接运行 `python <existing_file>.py`。

1. **必须**：创建新脚本（如 `_run_test.py`），用 `import` 或 `runpy` 调用目标。
2. **好处**：新脚本可设 `SafeToAutoRun: true` 实现自动执行。
