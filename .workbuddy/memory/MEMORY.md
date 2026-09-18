# LongHorizon-Harness 项目长期记忆

## 项目定位
面向边缘/资源受限环境的轻量级可靠 Agent 运行时。技术选型（用户拍板）：Python 纯标准库（零第三方依赖）+ OpenAI 兼容 HTTP(urllib) + 嵌入式 SQLite 单文件(WAL)。目标：最终具备行为规范、反思纠错、熔断限流等进阶能力，对标 dsh。

## 构建方式（关键约定）
- **分阶段教学式构建 P0–P9**（路线见 README.md）。每阶段只加一个能力、保持可运行、用文字讲清"为什么"。P0–P5 已完成。
- **面试速记文档**：`docs/INTERVIEW_NOTES.md`。每完成一个 Phase（P6+），就在该文档「分阶段精华」追加对应小节、并勾掉末尾「后续阶段」勾选框。面试素材随项目演进持续积累，不另起文件。
- 全局设计哲学：每个高级能力都是对 `plan-act-observe` 主循环的一次小增强，而非推倒重来（持久化=多写事件、重试=把失败再喂一次、护栏=工具执行前多一道检查、可观测=多计一次时）。

## 运行环境
- 项目专属虚拟环境：`D:\LongHorizon-Harness\.venv\Scripts\python.exe`（PyCharm 右下角可切）。
- 本机 bash PATH 异常，跑脚本用 venv 绝对路径：`D:/LongHorizon-Harness/.venv/Scripts/python.exe examples/<demo>.py`。
- 调试/跑测试用托管 Python 绝对路径：`C:/Users/asus/.workbuddy/binaries/python/versions/3.13.12/python.exe`。

## 文件地图（P0–P5）
- `harness/types.py` 共享词汇(Role/Message/ToolCall/ToolResult/ToolSpec, Role 对齐 OpenAI)；`Message.usage` 载 Token
- `harness/models.py` Model ABC + DummyModel（离线确定性，填 usage 估算）
- `harness/tools.py` Tool/ToolRegistry + RetryPolicy(退避+抖动+fail_fast) + ToolExecutor
- `harness/state.py` StateStore(SQLite 事件溯源, append-only events 表, redactor 钩子)
- `harness/engine.py` plan-act-observe 主循环 + resume() + _heal_dangling + 护栏检查 + 四处 span 埋点 + export_trace()
- `harness/guardrails.py` Guardrails(白名单/schema校验/脱敏) + validate_args + redact_obj
- `harness/tracing.py` Span/Tracer/estimate_cost/spans_from_events/OTelExporter(懒加载, 零依赖核心)
- `examples/` hello_agent / resume_demo / flaky_demo / guardrails_demo / trace_demo（各 Phase 演示）
