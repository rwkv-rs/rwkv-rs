# Repository Guidelines

## 使用方式
本文件只保留高优先级约束与导航. 遇到具体任务时, 先按当前需求判断是否需要继续阅读 `note_for_agents/` 下对应专题文档, 不要一次性加载全部说明.

建议按需阅读:
- 项目结构与各 crate 职责: `note_for_agents/project_structure.md`
- 构建, 测试, 基准与常用命令: `note_for_agents/development_workflow.md`
- 测试与基准镜像目录规范: `note_for_agents/testing_layout.md`
- 编译, 测试, 基准与回归检查规范: `note_for_agents/verification_standards.md`
- 代码风格, 命名与实现约束: `note_for_agents/coding_conventions.md`
- Code review 约束与 reviewer 行为规范: `note_for_agents/code_review.md`
- 提交与 PR 规范: `note_for_agents/commit_and_pr.md`

## 高优先级规则
- 先做目标对齐, 再展开方案设计或实现. 若用户强调“按官方/社区常见做法”, 不要先发明仓库私有流程, 工具或门禁.
- 优先采用 Rust 官方与主流社区共识中的最小充分方案; 只有在现有主流做法明确不满足目标时, 才考虑额外机制, 且必须先说明缺口.
- 本仓库是 Rust workspace, 依赖版本统一由 workspace 管理; 新增或调整依赖时, 优先修改根 `Cargo.toml`.
- 不要提交任何密钥或敏感信息, 例如 `HF_TOKEN`, 代理配置, 部署环境变量.
- 修改代码前先选择最小必要范围阅读说明; 修改后至少执行与变更范围匹配的验证命令. 涉及测试/基准的目录放置与命名时, 以 `note_for_agents/testing_layout.md` 为准; 涉及验收门槛与回归检查时, 以 `note_for_agents/verification_standards.md` 为准.
- 进行 code review, PR review 或 pre-merge 检查时, 默认主动使用 `pragmatic-clean-code-reviewer`, 并额外遵守 `note_for_agents/code_review.md` 中对 Rust 项目和本仓库约定的补充约束.
- 提交信息使用约定式提交, 格式保持简短, 可定位, 例如 `fix(rwkv-infer): ...`.

## 补充说明
`note_for_agents/` 中的文档是本仓库面向代理与贡献者的详细规范. 若主文档与专题文档存在冲突, 以更具体, 离任务更近的专题文档为准.
