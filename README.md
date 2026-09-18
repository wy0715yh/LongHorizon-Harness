# Agent Harness

> 项目地址：https://github.com/wy0715yh/LongHorizon-Harness

一个**从零开始、分阶段搭建**的轻量级可靠 Agent 运行时。每一阶段都产出可运行
代码，并解释"为什么这么写"。目标：最终具备行为规范、反思纠错、熔断限流等
进阶能力，对标主流生产级 Agent 运行时（如 dsh）。

## 构建路线（每阶段可运行）

| 阶段 | 状态 | 产出 |
| --- | --- | --- |
| P0 工程地基 | ✅ | 项目结构 / 打包 / 零运行时依赖 |
| P1 核心抽象 + 最小 Agent | ✅ | Message/Tool/Model 抽象 + plan-act-observe 主循环 |
| P2 状态持久化与恢复 | ✅ | SQLite 事件溯源 + checkpoint + `resume()` + 崩溃自愈 |
| P3 工具失败恢复 | ✅ | 退避重试 + 抖动 + 策略 + 双层恢复（执行器自愈 / 回灌模型重规划） |
| P4 行为规范与护栏 | ✅ | 系统策略 prompt / 输入输出校验 / 工具白名单 / 密钥脱敏 |
| P5 可观测性 | ✅ | 结构化 trace（延迟/Token/成本）+ waterfall 导出 + 耐久 span + 可选 OTel 导出 |
| P6 反思与自我纠错 | ✅ | Critic 评审 / 自省重规划 / 错误记忆（循环检测 + 跨任务失败记忆） |
| P7 熔断与限流 | ⬜ | 单工具熔断器 / 模型级限流器 / 背压队列 |
| P8 高级编排 | ⬜ | 子任务委派 / 并行工具 / 人在环 / 流式 |
| P9 服务化与评测 | ⬜ | CLI / HTTP 服务 / 基准测试 |

## 当前结构（P0–P6）

```
harness/
  __init__.py   版本号（0.6.0）
  types.py      Message / ToolCall / ToolResult / ToolSpec / Role —— 共享词汇
  models.py     Model 抽象 + DummyModel（离线，用于学习主循环）
  tools.py      Tool + ToolRegistry + calculator + RetryPolicy + ToolExecutor（重试/退避/抖动）
  state.py      StateStore：SQLite 事件溯源（tasks + events 表）+ errors 表（错误记忆）；redactor 钩子
  engine.py     plan-act-observe 主循环（store= 持久化 + resume + 自愈 + executor 执行 + 护栏 + Critic 反思）
  guardrails.py  Guardrails：系统策略 prompt + 工具白名单 + 输入 schema 校验 + 密钥脱敏（redact_obj）
  tracing.py     Tracer/Span：每步延迟 + Token + 成本度量；waterfall 可读导出；可选 OTelExporter（懒加载）
  critic.py      Critic 抽象 + RuleCritic（循环检测/错误记忆）+ LLMCritic（LLM-as-judge，可选）
examples/
  hello_agent.py      最小端到端演示（P1，无持久化）
  resume_demo.py      崩溃 -> 重开文件 -> resume -> 完成（P2）
  flaky_demo.py       工具失败双层恢复演示（P3）
  guardrails_demo.py  白名单/校验/脱敏三道栏 E2E 演示 + 落盘校验（P4）
  trace_demo.py       重试+崩溃续跑的可观测 trace（waterfall/JSON/耐久/OTel）（P5）
  reflection_demo.py  循环检测 + 错误记忆，自省重规划（P6）
```

## 运行

```bash
# P1：最小 Agent（不需要持久化，无需 API key）
python examples/hello_agent.py

# P2：崩溃后续跑（证明状态落盘、可恢复）
python examples/resume_demo.py

# P3：工具失败恢复（执行器重试自愈 + 回灌模型重规划）
python examples/flaky_demo.py

# P4：行为规范与护栏（白名单/校验/脱敏，带落盘断言）
python examples/guardrails_demo.py

# P5：可观测性（重试+崩溃的 trace，waterfall/JSON/耐久/OTel）
python examples/trace_demo.py

# P6：反思与自我纠错（循环检测拉回钻牛角尖的 agent + 错误记忆）
python examples/reflection_demo.py
```

## 核心概念：主循环

```
用户问题
  -> 调模型
  -> 若模型返回 tool_calls：经护栏检查后逐个执行工具，把结果作为 TOOL 消息追加回对话；
     若 Critic 判定本轮行为需反思，则把批评作为消息追加、不执行工具、重新调模型（自省重规划）
  -> 若模型返回纯文本：这就是最终答案，结束
```

关键机制：**工具结果（或 Critic 的批评）被追加回同一条对话**，模型据此决定下一步。
重试(P3)、护栏(P4)、反思(P6)、崩溃恢复(P2) 本质上都是同一件事的变体——**往对话里多追加几条消息，再调一次模型**。
