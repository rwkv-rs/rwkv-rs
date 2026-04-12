# 提交与 PR 规范

## 提交信息
提交历史采用 Conventional Commits 风格, 建议格式如下:

```text
<type>(<scope>): <summary>
```

示例:
- `fix(rwkv-infer): skip guided decoding during prefill`
- `feat(rwkv-nn/kernels): add guided token mask kernel`
- `chore(deps): bump workspace dependencies`

## 约定式提交类型
- `feat`: 新增一个功能
- `fix`: 修复一个 Bug
- `docs`: 文档变更
- `style`: 代码格式调整, 不影响功能
- `refactor`: 代码重构
- `perf`: 改善性能
- `test`: 测试变更
- `build`: 变更项目构建或外部依赖
- `ci`: 更改持续集成配置或脚本命令
- `chore`: 变更构建流程或辅助工具
- `revert`: 代码回退

## PR 要求
- 说明改动目的与影响范围.
- 列出受影响的 crate, 示例或部署流程.
- 写明实际执行过的验证命令.
- 若依赖额外配置, token, 硬件或驱动环境, 需明确标注.
- 只有涉及界面或文档展示变化时才需要截图.
