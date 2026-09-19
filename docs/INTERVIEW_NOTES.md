# LongHorizon-Harness 面试速记 / 设计精华

> **用途**：面试时讲清「这个项目为什么这么设计、每一层解决什么问题、背后是什么工程思想」。
> **维护约定**：本文件随项目迭代持续追加。每完成一个 Phase（P6+），就在「分阶段精华」里补一节，并在末尾「后续阶段」勾掉对应项。当前覆盖 **P0–P9**。

---

## 0. 一句话电梯演讲（背下来）

> 我构建了一个面向**边缘/资源受限环境**的轻量级 Agent 运行时，只用 Python 标准库、零第三方依赖，在单 SQLite 文件上落地了**工具调用失败恢复、任务状态崩溃续跑、执行轨迹可观测**三项生产级工程能力，并进一步具备行为规范护栏、**反思自我纠错**、面向依赖与 API 的**熔断/限流/背压**三道弹性防线、**高级编排**（并行工具、人在环、子任务委派、流式输出），以及把这一切**包装成可服务化入口（CLI / HTTP 服务）并有基准量化（冷启动/内存峰值/恢复耗时）**——全程零第三方依赖，装完 Python 即跑。

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

### P7 熔断与限流 — 三道弹性防线（Circuit Breaker / Rate Limiter / Backpressure）
**文件**：`resilience.py`（`CircuitBreaker` / `CircuitBreakerSet` / `RateLimiter` / `BackpressureQueue`）+ `engine.py`（三处接入点）+ `tracing.py`（三类 span）

- **为什么需要（又一类不可靠对象）**：P2–P6 处理了「工具会失败、进程会崩、模型会钻牛角尖」，但还有两类不可靠没管——**工具依赖**（一个 500 会把整个循环拖垮、还反复锤打已着火的服务）和**模型 API**（429 限流 / 突发账单）。P7 给这两类各加一道经典弹性模式，且全是**主循环的小增强、零新控制流**。
- **防线①：每工具熔断器（`CircuitBreaker`）。状态机 CLOSED→OPEN→HALF_OPEN→CLOSED**。连续失败达阈值就 OPEN，之后**快速失败（fast-fail）**并回灌模型「CIRCUIT OPEN」结构化结果 → 模型换工具而不是锤进虚空；冷却后 HALF_OPEN 放一个探测，成功则 CLOSED、再失败则重新 OPEN 并重置冷却。**关键决策：熔断器只数「重试耗尽后」的失败**——它坐在 P3 重试层之后，单次瞬态抖动（执行器已重试过）不计数，只有**持续失败**才 trip。这是它为何必须放在重试层之后而非之前。
- **防线②：模型级限流器（`RateLimiter`，令牌桶）**。`acquire()` 阻塞到令牌补充；这等待**本身就是背压**——压力向上游（模型调用）传播，而不是让我们撞 429 墙。时钟/`sleep` 可注入（虚拟时钟），demo 瞬时确定性跑完。`capacity`/`refill_per_s` 控 QPS；`acquire` 的等待单独记一个 `ratelimit` span，使 model span 只反映纯推理耗时。
- **防线③：背压队列（`BackpressureQueue`）**。`maxsize` 限制**单回合**准入的工具调用数；溢出即**丢弃（shed）**并转软失败回灌「重试下一回合」，而不是无界缓冲赌内存/锁死。每回合开始 `clear()`，所以只在「模型一次性要并发调很多工具」的突发下才咬人。
- **三道防线如何串起来 + 复用前面所有能力**：① 熔断 OPEN 的快速失败与 P4 护栏违规**同构**——都是「结构化失败回灌模型重规划」，零新控制流；② 熔断 trip 时**写入 P6 的跨任务错误记忆**（`store.record_error`），于是 Critic 能警告模型别再调死工具——**快速本地反应（熔断）vs 慢速跨任务学习（错误记忆）**；③ 限流的等待和背压的 shed 都落 `span`，进 P5 可观测。
- **可讲的设计张力（面试加分）**：背压「丢弃」语义很容易写错成「丢旧的腾地方」——但本框架下引擎已承诺本回合要跑那些已准入的调用，丢旧的会静默丢工作。正确语义是「满了就拒绝新调用（返回 False），由引擎转软失败」。这就是 `circuit_demo.py` 里一个真实踩坑：最初 `admit()` 永远返回 True，背压形同虚设，修正后才真正 shed。
- **验证铁证（`circuit_demo.py`）**：A) `boom` 连续失败 2 次 → 熔断 OPEN、fast-fail 1 次、模型改 `calculator` 出 42、且 `store.error_count('boom')>=1`；A2) 虚拟时钟跑通状态机 CLOSED→OPEN→HALF_OPEN→CLOSED；B) 令牌桶容量 2 取 5 个令牌需等 1.5s 虚拟时间、引擎 ratelimit span 记录到等待；C) 5 并发 echo / maxsize=2 → 3 个被 shed、软失败回灌、模型重试直到 `all 5 done`。

### P8 高级编排 — 并行工具 / 人在环 / 子任务委派 / 流式
**文件**：`orchestration.py`（`execute_calls` / `Delegate` / `HumanInput`）+ `engine.py`（gate→execute→commit 三段式工具循环、`_call_model` 流式、`_ask_human`）+ `tools.py`（`Tool.human`）+ `models.py`（`chat_stream`）+ `state.py`/`tracing.py`（线程安全改造）

- **全局决策：把「编排」拆成主循环上的三段式工具处理**，而不是另起一套调度器。**每回合工具调用 = PASS1 准入(GATE，串行) → PASS2 执行(EXECUTE，可并行) → PASS3 落盘(COMMIT，主线程)**。这个拆分是 P8 四个能力全部能干净挂上的关键骨架。
- **能力①：并行工具（`execute_calls`，线程池）**。模型一次发出多个**互相独立**的工具调用时，用 `ThreadPoolExecutor` 并发跑，墙钟时间从「N×单调用」降到「≈1×单调用」。`orchestration_demo.py` 实测 3 个各睡 0.1s 的慢调用：串行 0.30s、并行 0.10s（~3× 加速）。**关键边界**：准入闸门（护栏/熔断/背压）仍是**串行**的——只有「执行」并行；结果回主线程**顺序落盘**，对话日志保持单写者，所以并发写安全（靠 P8 给 `StateStore`/`Tracer` 加的锁 + `check_same_thread=False`）。
- **能力②：人在环（`Tool.human=True` + `_ask_human`）**。工具注册时标 `human=True`，模型一旦调它，引擎**暂停自主循环**、调用 `human_input(question, args)` 回调要答案，再把答案作为 `TOOL` 消息回灌——agent 懂得「何时该停下来问人」。回调默认 `input()`（真实 TTY 交互），没接回调则返回结构化 `HUMAN INPUT UNAVAILABLE` 优雅降级，绝不崩。这正是「human-in-the-loop」的本质：不是全自主，而是知道何时求助。
- **能力③：子任务委派（`Delegate` 元工具 / 嵌套 Engine）**。模型调 `delegate` 工具（参数 `sub_prompt`），引擎**起一个嵌套 Engine** 跑子任务、把最终答案折回主对话。这是**分层规划（hierarchical planning）**：boss 拆解、委派、汇总。嵌套引擎被刻意做成**叶子节点**——不递归委派、无 Critic、无流式、且**内存态无持久化**——保证永不无限递归、子任务有界。父引擎只记一个 `delegate` span，子任务内部全被一条 `TOOL` 消息藏住。
- **能力④：流式输出（`Model.chat_stream` + `_call_model`）**。`Model` 增加 `chat_stream` 默认实现（一次性 yield 完整消息），真模型可覆盖成 SSE 分片；`_call_model` 在 `stream=True` 时逐 delta 打印并**现场拼回**最终 `Message`，`tool_calls`/`usage` 以**最后一个 delta** 为准。主循环其余逻辑与离线调用完全同构——流式只是「拿消息」的方式变了。
- **设计张力（面试加分）**：并行执行最容易踩的坑是「并发写持久化层」。本项目没为了并行去改事件日志语义，而是给 `StateStore`/`Tracer` 加锁并放开 `check_same_thread`，让并行 worker 各写各的 `tool_attempt`/`tool` span 仍安全——**线程池只在「执行」这一段，落盘始终主线程**，既拿到并发加速又不破坏事件溯源的单写者不变式。委派同理：子任务不碰父 store，避免嵌套任务簿记与跨线程写。
- **验证铁证（`orchestration_demo.py`）**：A) 3 慢调用并行 ~3× 加速；B) 模型调 `confirm(human=True)` → 引擎暂停、`human_input` 返回 `yes, approved` → 答案回灌、继续；C) boss 调 `delegate` → 嵌套引擎算出 42 并折回、boss 出 `boss final: ...42`；D) `stream=True` 下最终回答逐字打印且拼装正确。

### P9 服务化与评测 — CLI / HTTP 服务 / 基准测试
**文件**：`server.py`（HTTP 薄包装）/ `cli.py` + `__main__.py`（命令行）/ `models.py`（`OpenAICompatibleModel`，补齐了此前一直宣称但未实现的真模型客户端）/ `examples/benchmark_demo.py`（评测）

- **全局决策：服务化 = 引擎的「薄包装」，不是新控制流**。P9 的三件事里，CLI 和 HTTP 服务都**不改主循环**——它们只做三件事：① 按配置构造一个 `Engine`；② 调 `engine.run()`；③ 把 `engine.export_trace()` 序列化成 JSON。整个项目到「可部署」这步，内核一行没动。这本身就是「每一层能力都是主循环的小增强」哲学在服务层的回响——只不过服务层增强的是「调用方式」而非「循环内部」。`harness/server.py` 用 **stdlib `http.server` + `ThreadingHTTPServer`**，零 Flask/FastAPI，让「零第三方依赖」从内核一直成立到上线。
- **能力①：CLI（`python -m harness`）**。`cli.py` 用 `argparse` 提供 `run`（单 prompt 跑任务 + `--trace` 打印轨迹）和 `serve`（起 HTTP 服务）两个子命令，`--model dummy|openai` 切换后端、`--store` 可选持久化。最快的端到端演示入口：`python -m harness run "what is 6*7" --trace`。
- **能力②：HTTP 服务（`server.py`）**。`POST /run` 收 `{prompt, model, max_steps, db?, task_id?}` 返回 `{answer, task_id, trace}`；`GET /health` 健康检查。每请求一线程（线程安全已在 P8 打底），**单任务无状态、带 `db` 即支持 resume**（同一 SQLite 文件跨请求续跑）。一个坏任务只会返回 `500 {"error":...}`，**绝不拖垮服务进程**——服务的健壮性复用的是引擎「失败不崩」的同一条契约。
- **补完真模型客户端（`OpenAICompatibleModel`）**：此前路线一直宣称「OpenAI 兼容 HTTP 抽象」，但 `models.py` 实际只有 `DummyModel`。P9 用 **stdlib `urllib`** 补齐了真客户端（chat + Message/ToolCall 与 OpenAI 格式的双向转换），让「换模型零改引擎」从话术变成可跑的事实——`Engine(OpenAICompatibleModel(...), ...)` 一行切换，引擎代码零改动。这同时坐实了「零依赖」：连真模型 HTTP 调用都不用 `requests`。
- **能力③：基准测试（`benchmark_demo.py`）——把「轻量 + 可靠」量化成数字**：
  - **冷启动**：子进程真测 `import harness` 耗时 + 进程内 Engine 构造 + 单任务端到端延迟（实测 import 亚毫秒级、构造 0.03ms、单任务 ~1ms）。
  - **内存峰值**：`tracemalloc` 测一次任务峰值（实测 ~0.03MB）——直接回应「轻量」主张。
  - **恢复耗时**：`crash_after=1` 模拟第 1 步崩溃，再用**全新 Engine 打开同一 SQLite 文件** resume 到完成，量化墙钟（实测 ~12ms）——这就是事件溯源设计的硬 payoff。
  - **附：离线 HTTP 往返**：起后台服务 `POST /run` 断言返回 `answer+task_id+trace`，证明「可服务化」不是嘴上说的。
- **为什么评测是收尾而不是炫技**：前面 P2–P8 讲的是「可靠」和「弹性」的工程手段，但面试官会问「到底多轻、多快恢复」。P9 用项目自带的可观测轨迹（P5 Span）+ 崩溃恢复（P2）直接算出这三项指标，**复用已有能力做度量，不另造一套 benchmark 框架**——又一次「小增强」哲学。
- **验证铁证（`benchmark_demo.py` ALL CHECKS PASSED）**：import 0.0003s / 构造 0.03ms / 单任务 1.34ms / 内存峰值 0.029MB / 崩溃→recovery 12.17ms 且答案正确 / HTTP `POST /run` 返回 `answer+task_id+trace` + `GET /health` ok。

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

**Q：工具依赖挂了 / 模型 API 被限流（429）你怎么兜底？**
A：两道防线（P7）。工具侧：每工具熔断器——连续失败达阈值就 OPEN 快速失败，回灌模型让它换工具；冷却后放探测，成功则闭合。关键是熔断器坐在 P3 重试层之后，只对「重试仍失败」的持续故障计数，瞬态抖动不计。模型侧：令牌桶限流器在调模型前 `acquire()`，没令牌就阻塞等待——这等待本身就是背压，避免撞 429。突发并发工具调用用背压队列 shed 溢出、软失败回灌下一回合。三道防线都只是主循环的小增强，且复用 P4（结构化失败回灌）/P5（span 可观测）/P6（错误记忆）。

**Q：为什么零依赖 / 为什么不直接用现成框架？**
A：目标场景是边缘/受限环境，要求常驻内存极低、无外部服务。stdlib-only 意味着装完 Python 即跑；SQLite 单文件自带崩溃恢复。现成框架默认重运行时、假设云端资源，本项目在更紧约束下解决同样问题且每步可演示。

**Q：模型抽象怎么设计的？**
A：`Model` 抽象基类 + 具体实现（离线 `DummyModel` 学主循环，后续 `OpenAICompatibleModel`）。引擎面向接口编程，换后端零改引擎——依赖倒置。Critic 同理：引擎只依赖 `Critic` 接口，规则版和 LLM 版可换。

**Q：多个工具调用你怎么并发 / 怎么让人介入 / 怎么做分层？**
A：都挂在 P8 的「gate→execute→commit」三段式工具循环上。并发：互相独立的调用丢进线程池跑，准入闸门仍串行、结果回主线程顺序落盘——既加速又不破坏事件溯源单写者不变式（持久化层为此加了锁）。人在环：工具标 `human=True`，引擎暂停调 `human_input` 回调取答案再回灌，而非永远自主。分层：模型调 `delegate` 元工具时引擎起嵌套 Engine 跑子任务并把答案折回，嵌套引擎是叶子、不递归、内存态，保证有界。流式：`chat_stream` 逐 delta 打印并拼回消息，主循环其余逻辑同构。

**Q：并行执行怎么保证线程安全？**
A：事件日志与 trace 都是共享状态，原本单线程写。P8 让工具执行进线程池，于是给 `StateStore` 加 `threading.RLock` 并放开 `check_same_thread=False`、给 `Tracer` 的 span 记录加锁，worker 各写各的事件/span 仍安全；而「把结果折回对话 + 落盘 tool_result/message」只发生在主线程，所以对话顺序确定、不竞态。这比「为并行改事件溯源语义」小得多、稳得多。

---

**Q：怎么把它跑成服务 / 怎么对接真模型？**
A：服务化是引擎的薄包装，不改主循环——`server.py` 用 stdlib `http.server` 起 `POST /run`（收 prompt 返回 answer+trace），CLI 用 `python -m harness run/serve` 即可。真模型客户端 `OpenAICompatibleModel` 也用纯 urllib 实现，构造时一行换成 `Engine(OpenAICompatibleModel(...))`，引擎零改动，连 HTTP 调用都零第三方依赖。

**Q：你说「轻量、可靠」，有数字吗？**
A：有，`benchmark_demo.py` 把主张量化了——`import harness` 亚毫秒级、Engine 构造 ~0.03ms、单任务端到端 ~1ms、一次任务内存峰值 ~0.03MB（tracemalloc）、第 1 步崩溃后开同一 SQLite 文件 resume 完成 ~12ms 且答案正确。冷启动/内存来自可观测 Span，恢复耗时来自 P2 事件溯源——都是复用已有能力度量，不另造框架。

## 6. 后续阶段（待补充，完成后在此追加小节并勾掉）

- [x] **P6 反思与自我纠错** — Critic 评审 / 自省重规划 / 错误记忆（把 P3/P4「失败回灌模型」升级成「模型自己批评自己」）
- [x] **P7 熔断与限流** — 单工具熔断器（CLOSED/OPEN/HALF_OPEN）/ 模型级令牌桶限流 / 背压队列 shed 溢出
- [x] **P8 高级编排** — 并行工具（线程池）/ 人在环（human_input 暂停）/ 子任务委派（嵌套 Engine）/ 流式（chat_stream）
- [x] **P9 服务化与评测** — CLI（`python -m harness`）/ HTTP 服务（stdlib http.server）/ 基准测试（冷启动·内存峰值·恢复耗时）+ 补齐 OpenAICompatibleModel 真模型客户端

> 更新日志：2026-09-18 创建，覆盖 P0–P5；20:45 补充 P6（反思与自我纠错：Critic / 循环检测 / 错误记忆 / 自省重规划）；2026-09-19 补充 P7（熔断与限流：CircuitBreaker / RateLimiter 令牌桶 / BackpressureQueue 背压 shed）；2026-09-19 补充 P8（高级编排：gate→execute→commit 三段式工具循环 / execute_calls 线程池并行 / Tool.human 人在环 / Delegate 嵌套引擎委派 / Model.chat_stream 流式 / StateStore·Tracer 线程安全）；2026-09-19 补充 P9（服务化：CLI `python -m harness` / HTTP 服务 `server.py` stdlib http.server / 基准测试 `benchmark_demo.py` 量化冷启动·内存峰值·恢复耗时；并补齐此前缺失的 `OpenAICompatibleModel` urllib 真模型客户端，坐实「换模型零改引擎」与「零依赖」）。
