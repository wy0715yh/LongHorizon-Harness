# LongHorizon-Harness 面试速记 / 设计精华

> **用途**：面试时讲清「这个项目为什么这么设计、每一层解决什么问题、背后是什么工程思想」。
> **维护约定**：本文件随项目迭代持续追加。每完成一个 Phase（P6+），就在「分阶段精华」里补一节，并在末尾「后续阶段」勾掉对应项。当前覆盖 **P0–P6**。

---

## 0. 一句话电梯演讲（背下来）

> 我构建了一个面向**边缘/资源受限环境**的轻量级 Agent 运行时，只用 Python 标准库、零第三方依赖，在单 SQLite 文件上落地了**工具调用失败恢复、任务状态崩溃续跑、执行轨迹可观测**三项生产级工程能力，并进一步具备行为规范护栏与**反思自我纠错**。

**为什么是差异化选题**：主流框架（LangGraph / AutoGen / CrewAI）默认假设有云 GPU 和无限内存；本报告切的是它们照顾不好的「轻量 + 可靠」场景，讲的可不是「又包一层聊天机器人」，而是 Agent 真实的工程痛点。

---

## 1. 简历可用版本（STAR 浓缩，直接抄）

- **背景(S)**：生产级 Agent 在边缘/开发机上最怕三件事——工具调用动不动超时限流脏数据、长任务进程一崩就从头跑、线上等于黑盒没法 debug；更进一步，模型还可能钻牛角尖重复调同一个坏工具、或反复踩同一个坑。
- **动作(A)**：从零设计了一个 stdlib-only 运行时，事件溯源做持久化、执行器退避重试 + 失败回灌模型做自愈、结构化 Span 做可观测、四道护栏做行为规范；并加了 Critic——循环检测 + 跨任务错误记忆，把「失败回灌」升级成「模型自己批评自己、自省重规划」。
- **结果(R)**：零外部依赖、单文件存储约束下，进程崩溃后从断点秒级续跑零数据丢失；工具瞬态失败对模型透明自愈；每步可量化延迟/Token/成本并可对接 OTel；钻牛角尖的 agent 被 Critic 拉回、自动换工具收尾。
- **技术栈**：Python 3.13（仅标准库）/ SQLite(WAL) / OpenAI 兼容 HTTP 抽象 / 事件溯源 / 指数退避+抖动 / 可选 OpenTelemetry / Critic（规则 + LLM-as-judge）。

**锋利版一句话（放简历 bullet）**：
> 构建了面向边缘/开发机的轻量级 Agent 运行时（Python 标准库、零依赖），在单 SQLite 文件上实现工具调用失败自愈、任务状态断点续跑、端到端执行轨迹回溯与 Critic 反思纠错；进程崩溃后从断点秒级恢复、零数据丢失。

---

## 2. 技术选型与理由（必考题：为什么这么选）

| 决策 | 选择 | 一句话理由 |
| --- | --- | --- |
| 语言 | Python + **仅标准库** | `sqlite3`/`urllib`/`json` 内置 → 「装完 Python 直接跑，无需 pip install」本身就是最硬的「轻量」证据 |
| LLM 后端 | OpenAI 兼容 HTTP 抽象 | 一个客户端兼容 OpenAI/vLLM/Ollama/国产模型；引擎面向 `Model` 接口编程，换模型零改引擎 |
| 持久化 | 嵌入式 SQLite 单文件（WAL） | 零外部服务、崩溃恢复天然；WAL 让 trace 导出不阻塞写入 |

> 面试点：**为什么不用 LangGraph/AutoGen？** 它们默认依赖重运行时、假设云端资源；本项目刻意在「常驻内存极低、无外部服务」约束下解决同样问题，且每一步可运行可演示，面试官能 `python examples/reflection_demo.py` 直接跑起来。

---

## 3. 分阶段精华（每个 Phase = 一个可讲的设计决策）

### P0 工程地基
- **做了什么**：项目结构 / `pyproject.toml`（零依赖，仅声明 `python>=3.10`）/ 项目专属 `.venv`（与机器上其他环境隔离）/ `.gitignore`（排除 `.venv`、缓存、运行时 DB）。
- **可说点**：工程化从虚拟环境隔离开始——依赖只装进本项目 `.venv`，绝不污染系统；`.gitignore` 保证推 GitHub 不会传几百 MB 环境。

### P1 核心抽象 + 最小可跑 Agent（地基中的地基）
**文件**：`types.py` / `models.py` / `tools.py` / `engine.py`
- **types.py — 先定「词汇表」**：`Message / ToolCall / ToolResult / ToolSpec / Role`。`Role` 直接对齐 OpenAI 的 `system/user/assistant/tool` 枚举 → 接真模型时几乎不用翻译。面试词汇：**契约先行 / schema 设计**。
- **models.py — 依赖反转**：`Model` 是抽象基类，`DummyModel` 是实现。引擎不关心「答案从哪来」，今天用离线 `DummyModel` 学主循环，明天换 `OpenAICompatibleModel` 引擎零改动。面试词汇：**面向接口编程 / 依赖倒置（DIP）**。
- **tools.py — 声明与实现分离**：`Tool` 把「给模型的声明 ToolSpec」和「真正执行的函数 fn」绑在一起；`Tool.run` 里**工具抛异常绝不炸 agent，而是包成 `ToolResult(ok=False)`**。`ok` 字段是 P3 重试/恢复的种子。
- **engine.py — 主循环（项目心脏）**：逻辑四行——调模型 → 若返回 tool_calls 就逐个执行、把结果作为 `TOOL` 消息追加回对话、再调模型 → 若返回纯文本就结束。

> **全局必须内化的心智模型**：「工具结果被追加回同一条对话，模型据此决定下一步」。重试(P3)、反思(P6)、崩溃恢复(P2) 本质上都是同一件事的变体——**往对话里多追加几条消息，再调一次模型**。先吃透这个循环，后面每个高级功能都只是往循环里塞一个处理步骤。

### P2 状态持久化与恢复 — 事件溯源
**文件**：`state.py`（`StateStore`）+ `engine.py`（`resume()` / `_heal_dangling`）
- **核心决策①：状态靠「重放日志」重建，不存快照**。`events` 表只增不删，对话不是存出来的，是 `load_messages` 重放事件重建的。很多人第一反应是「每步存整份对话快照」，但崩溃若发生在「执行工具」和「存快照」之间，整份状态就没了。事件溯源下「崩溃」只是「停止追加」，重启从上次落盘那笔往后走——**快照永不过期，因为根本没有快照**。面试词汇：**Event Sourcing**。
- **核心决策②：崩溃自愈（`_heal_dangling`）**。真实 LLM API 要求 `tool_calls` 必须都有回包，否则非法输入。引擎重启后扫描出「悬空 tool_call（模型要调但无结果）」，**自动补跑一遍**把对话修成合法状态再继续。这是「可靠」与「玩具」的分水岭。
- **一句话故事（简历素材）**：进程在第 1 步崩溃，同一 DB 文件换进程重开，`_heal_dangling` 从事件日志重建对话、补跑悬空调用、算出 42——零数据丢失，长任务不用从头重跑。

### P3 工具失败恢复 — 双层自愈
**文件**：`tools.py`（`RetryPolicy` / `ToolExecutor`）+ `types.py`（`ToolResult.attempts`）+ `engine.py`
- **Layer 1 — 执行器重试（对模型透明）**：`flaky` 前 2 次失败、第 3 次成功，执行器按 `delay = min(cap, base·2ⁿ) · jitter` 退避重试，**模型全程没看到失败**，直接拿 42。解决瞬态错误（超时/限流/503）别浪费一次 LLM 回合。
- **Layer 2 — 回灌模型重规划（不崩溃）**：`boom` 永远失败，重试 3 次耗尽后把 `FAIL: ... (after 3 attempts)` 作为 `TOOL` 消息喂回，模型选兜底结论而不是让 agent 崩。
- **为什么加 jitter（抖动）**：无抖动时一堆 agent 同时重试会「惊群（thundering herd）」把已脆弱的依赖打死；随机化把重试错开。分布式系统经典 lesson。
- **`fail_fast` 策略给谁用**：给**不可重入**工具——如「扣款」重复执行就重复收费，宁可立即把失败交给模型判断也不自动重试。`RetryPolicy` 的 `strategy` 字段就是为此存在。
- **面试点**：双层恢复几乎免费——Layer 2 就是 P1 那句话「结果追加回对话、再调一次模型」的自然结果。重试/反思/恢复底层全是同一个循环。

### P4 行为规范与护栏 — 把玩具关进围栏
**文件**：`guardrails.py` + `engine.py` + `state.py`（`redactor`）
- **四道栏**：① 系统策略 prompt（`build_system_prompt` 把行为规则渲染成 `# Behavior policy` 分隔块，人可审计）② 工具白名单（`allowed_tools`，危险工具直接拦）③ 输入 schema 校验（`validate_args`，坏参数到不了真实工具）④ 密钥脱敏（`redact_obj`，落盘前 scrub）。
- **核心决策：护栏违规 = NON-FATAL = 结构化失败回灌模型重规划**。与 P3「工具失败」同构——都是往循环里多塞一条失败消息、再调一次模型，只是语义从「没做成」变成「别做」。P4 没引入新控制流，只在工具执行前多一道检查，**深度复用 plan-act-observe 心智模型**。
- **关键设计：脱敏放在持久化层（`store.redactor`），不在各调用点**。这样 `message` / `tool_result` / `guardrail` **所有**落盘事件被统一 scrub，密钥永不落盘。证据：`tool_result "recorded: deploy using ***REDACTED***"`。
- **验证铁证（事件日志）**：白名单拦截 `shell`（从未执行）→ 校验拦 `save_note` 类型错误 → 合法调用落盘已脱敏 → `calculator` 正常出 42。

### P5 可观测性 — 从「打日志」到「可度量」
**文件**：`tracing.py`（`Span` / `Tracer` / `estimate_cost` / `OTelExporter`）+ `types.py`（`Message.usage`）+ `engine.py`（四处埋点）
- **决策①：Span 是「可测量的工作单元」，且是耐久事件**。每次模型/工具调用生成 `Span`（延迟/Token/成本/状态）并作为 `span` 事件和 `message` 平级落盘。→ **trace 跨崩溃存活**：进程第 1 步崩，换进程 resume 后从 store 重建的 trace 仍含崩溃前的 model span。可观测性若随进程一起死，恰是你要 debug 崩溃时最没用的东西。
- **决策②：成本 = 模型名 + Token 数，价格表可配**。`estimate_cost("gpt-4o-mini", ...)` 查内置表；真模型把 API `usage` 填进 `Message.usage`，引擎自动算 `cost_usd`。这是简历「X→Y 数字」的直接来源。
- **决策③：埋点嵌入主循环，不另起炉灶**。模型调用前后 `time.perf_counter()`，无新线程/装饰器/采样配置。每加一层能力顺手补一个 span。
- **决策④：OTel 可选 + 中性格式 + 可插拔 exporter**。核心零依赖；仅 `pip install opentelemetry-sdk` 后 `OTelExporter` 才把中性 `Span` 翻成 OTel span 发往 Jaeger/Tempo/OTLP；SDK 未装时优雅跳过。标准可观测性架构模式。
- **验证**：`model_calls=3 tool_calls=2 retries=2 cost=$0.000015`；崩溃后从 store 重建 trace 仍含 3 个 model span；OTel 未装时提示安装命令。

### P6 反思与自我纠错 — Critic 评审 + 自省重规划 + 错误记忆
**文件**：`critic.py`（`Critic` / `RuleCritic` / `LLMCritic`）+ `engine.py`（critic 块）+ `state.py`（`errors` 表）
- **核心决策①：反思 = 同一种循环增强，只是「触发者」变了**。P3 的「失败回灌」触发者是「工具出错」，P6 的「批评回灌」触发者是「Critic 判定 agent 行为不当」。两者都往对话追加一条消息再调一次模型——**零新控制流**。这是全局设计哲学（每一层能力都是主循环的一次小增强）的最强证据。
- **核心决策②：Critic 抽象 + 两种实现**。引擎只依赖 `Critic` 接口，具体实现可换：
  - `RuleCritic`——确定性规则（循环检测 + 错误记忆），离线可跑、零二次 LLM 调用、可解释。
  - `LLMCritic`——第二个 `Model` 当 judge（LLM-as-a-judge），看最近对话回 `OK` 或 `CRITIQUE: ...`。更聪明但多一次调用、可能不稳定。面试词汇：**依赖倒置 / 元认知**。
- **RuleCritic 两项检查**：① 循环检测——最近 K 步发出**完全相同**的 tool_call（name+参数）→ "你重复 N 次没进展，换方法或给结论"；② 错误记忆——某工具历史失败次数达阈值 → "别再调它"。
- **核心决策③：错误记忆跨任务、跨崩溃持久化**。`state.py` 新增 `errors` 表，`record_error` 用 `ON CONFLICT ... DO UPDATE SET n=n+1` 做幂等计数；`_classify_error` 把数字/引号归一化，让 `Timeout after 3s` 与 `Timeout after 7s` 归为同一桶——记忆的是「失败的种类」而非精确文本。所以 agent 能从「上次也失败过」学习，而不只是当次重试。
- **核心决策④：自省重规划不崩溃、有上限**。`max_reflections` 限制反思次数，防「顽固 agent」无限反思死循环；达上限后让循环正常执行/返回。Critic 的批评作为一条 `[Critic]` 消息追加回对话，模型下一轮据此改策略——正是 P1 那句话「结果追加回对话、再调一次模型」。
- **一句话故事（简历素材）**：agent 钻牛角尖连调坏工具 `boom` 两次，RuleCritic 循环检测触发，把批评注入对话，模型下一轮改调 `calculator` 算出 42 收尾——全程没崩、没手动介入。
- **验证铁证**：`reflection_demo.py` 输出 `[critic] loop: You have called the same tool 'boom' 2 times...` → 模型改 `calculator` → `FINAL ANSWER: ... 42`；场景 B 直接证明 `error_memory` critique 在工具历史失败 2 次时触发。

---

## 4. 贯穿全局的设计哲学（面试金句）

> **这个 Harness 的演进方式很朴素：每一层能力，都是对 `plan-act-observe` 主循环的一次「小增强」，而不是推倒重来。**
> 持久化 = 循环里多写几笔事件；重试 = 循环里把失败再喂一次；护栏 = 循环里工具执行前多一道检查；反思 = 循环里多塞一条批评再调一次模型；可观测 = 循环里多计一次时。所以代码始终小、始终可跑、始终讲得清。

---

## 5. 面试高频 Q&A（预设问题 + 回答要点）

**Q：工具调用失败你怎么处理？**
A：双层。执行器层做指数退避+抖动重试，覆盖瞬态错误，对模型透明；重试耗尽仍失败，把结构化失败（含尝试次数）回灌模型让其重规划或兜底，绝不让 agent 崩。不可重入工具（如扣款）用 `fail_fast` 策略跳过自动重试。

**Q：进程崩了任务怎么办？**
A：事件溯源。所有步骤只追加进 SQLite `events` 表，对话靠重放重建。重启后用同一 task_id `resume()`，引擎 `_heal_dangling` 自动补跑崩溃前悬空的工具调用，从断点续跑、零丢失。

**Q：你的 agent 会不会乱调工具/泄露密钥？**
A：四道栏。系统策略 prompt 约束行为；白名单限制可调工具；输入 schema 校验挡住坏参数；脱敏放在持久化层，所有落盘事件统一 scrub 密钥。护栏违规不崩溃，作为结构化失败回灌模型重规划。

**Q：模型钻牛角尖（反复调同一个坏工具）怎么办？**
A：Phase 6 的 Critic。RuleCritic 做循环检测——最近 K 步发出完全相同的 tool_call 就判定死循环，把批评作为消息回灌、不执行工具、让模型自省重规划（换工具或给结论）。`max_reflections` 防止无限反思。另有跨任务的错误记忆：某工具历史失败多次，Critic 直接建议别再调它。反思和 P3 的重试同源——都是往对话多追加消息再调一次模型，零新控制流。

**Q：怎么衡量 agent 跑得好不好？**
A：结构化 Span 度量每次模型/工具调用的延迟、Token、成本，作为耐久事件落盘，所以崩溃后 trace 仍在；可导出 waterfall/JSON，可选接 OTel 发往 Jaeger/Tempo。成本是模型名+Token 数查价格表算的。

**Q：为什么零依赖 / 为什么不直接用现成框架？**
A：目标场景是边缘/受限环境，要求常驻内存极低、无外部服务。stdlib-only 意味着装完 Python 即跑；SQLite 单文件自带崩溃恢复。现成框架默认重运行时、假设云端资源，本项目在更紧约束下解决同样问题且每步可演示。

**Q：模型抽象怎么设计的？**
A：`Model` 抽象基类 + 具体实现（离线 `DummyModel` 学主循环，后续 `OpenAICompatibleModel`）。引擎面向接口编程，换后端零改引擎——依赖倒置。Critic 同理：引擎只依赖 `Critic` 接口，规则版和 LLM 版可换。

---

## 6. 后续阶段（待补充，完成后在此追加小节并勾掉）

- [x] **P6 反思与自我纠错** — Critic 评审 / 自省重规划 / 错误记忆（把 P3/P4「失败回灌模型」升级成「模型自己批评自己」）
- [ ] **P7 熔断与限流** — 单工具熔断器 / 模型级限流器 / 背压队列
- [ ] **P8 高级编排** — 子任务委派 / 并行工具 / 人在环 / 流式
- [ ] **P9 服务化与评测** — CLI / HTTP 服务 / 基准测试（量化冷启动、内存峰值、恢复耗时）

> 更新日志：2026-09-18 创建，覆盖 P0–P5；20:45 补充 P6（反思与自我纠错：Critic / 循环检测 / 错误记忆 / 自省重规划）。
