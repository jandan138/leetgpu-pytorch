# Agent Memory: researcher

## Stable Preferences
- 

## Known Pitfalls
- 

## Proven Patterns
- 

## Anti-Patterns
- 

## Tooling Notes
- 

## Recent Deltas
- Run: none
  - Summary: 
  - Source: 



- Run: run_2026_03_04_backend_decouple_phase2
  - Summary: merged delta at 2026-03-04T15:09:56Z
  - Source: runs/run_2026_03_04_backend_decouple_phase2/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_04_backend_decouple_phase2
    - Source Agent: researcher
    - Recorder: doc-writer
    
    ## 1) Delta Summary
    - phase2 建议聚焦 backend capability metadata 与无 MuJoCo 单测。
    
    ## 2) New Stable Preferences
    - 先提高可观测性和测试覆盖，再推进真实外部引擎适配。
    
    ## 3) Pitfalls Learned
    - 只有工厂路由不足以保障演进，需要能力矩阵与自动化验证配套。
    
    ## 4) Reusable Patterns
    - 每个 backend class 显式提供 `implemented` 标记，并导出列表 API。
    
    ## 5) Evidence Links
    - agent_team/runs/run_2026_03_04_backend_decouple_phase2/logs/researcher.md

- Run: run_2026_03_04_backend_decouple_phase2
  - Summary: merged delta at 2026-03-04T15:11:00Z
  - Source: runs/run_2026_03_04_backend_decouple_phase2/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_04_backend_decouple_phase2
    - Source Agent: researcher
    - Recorder: doc-writer
    
    ## 1) Delta Summary
    - phase2 建议聚焦 backend capability metadata 与无 MuJoCo 单测。
    
    ## 2) New Stable Preferences
    - 先提高可观测性和测试覆盖，再推进真实外部引擎适配。
    
    ## 3) Pitfalls Learned
    - 只有工厂路由不足以保障演进，需要能力矩阵与自动化验证配套。
    
    ## 4) Reusable Patterns
    - 每个 backend class 显式提供 `implemented` 标记，并导出列表 API。
    
    ## 5) Evidence Links
    - agent_team/runs/run_2026_03_04_backend_decouple_phase2/logs/researcher.md

- Run: run_2026_03_04_backend_decouple_phase3
  - Summary: merged delta at 2026-03-04T15:27:19Z
  - Source: runs/run_2026_03_04_backend_decouple_phase3/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_04_backend_decouple_phase3
    - Source Agent: researcher
    - Recorder: doc-writer
    
    ## 1) Delta Summary
    - 已确认 phase3 最优先下一步是“运行时插件加载”而非继续堆内置分支。
    - 给出验收方向：配置化加载、默认兼容、错误可诊断、无 MuJoCo 单测可跑。
    
    ## 2) New Stable Preferences
    - 优先做“扩展路径能力”而不是一次性接某个具体引擎。
    
    ## 3) Pitfalls Learned
    - 只有 `register_*` API 不够，缺入口会导致第三方集成仍需改核心代码。
    
    ## 4) Reusable Patterns
    - 插件模块导入即注册（import-time registration）是低复杂度高收益方案。
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase3/logs/researcher.md`

- Run: run_2026_03_04_backend_decouple_phase4
  - Summary: merged delta at 2026-03-04T15:33:07Z
  - Source: runs/run_2026_03_04_backend_decouple_phase4/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_04_backend_decouple_phase4
    - Source Agent: researcher
    - Recorder: doc-writer
    
    ## 1) Delta Summary
    - 推荐下一步聚焦“插件加载线程安全与并发重复调用可靠性”，保持接口不变，仅增强内部保证。
    
    ## 2) New Stable Preferences
    - 优先做小步、可测、接口不变的稳定性改进（先 reliability，后 feature）。
    
    ## 3) Pitfalls Learned
    - 仅有“去重集合”不等于并发安全；多线程重复加载时仍可能出现竞态。
    
    ## 4) Reusable Patterns
    - 用模块级锁 + 原子状态更新实现“exactly-once registration”语义。
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase4/logs/researcher.md`
    

- Run: run_2026_03_04_backend_decouple_phase5
  - Summary: merged delta at 2026-03-04T15:40:48Z
  - Source: runs/run_2026_03_04_backend_decouple_phase5/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_04_backend_decouple_phase5
    - Source Agent: researcher
    - Recorder: doc-writer
    
    ## 1) Delta Summary
    - 明确了“真实后端切换测试”的最早引入时点：首个非 MuJoCo 后端可实例化并完成 env 映射后的同一天。
    - 给出了 5 条仓库内可执行 gating conditions（实现、映射、渲染、依赖、smoke 配置）。
    
    ## 2) New Stable Preferences
    - 采用“两层测试策略”：先框架级常驻测试，再真实后端 smoke 自动激活。
    
    ## 3) Pitfalls Learned
    - 只实现 backend class 不足以开展真实切换测试，必须同时具备 env registry 映射与可运行依赖环境。
    
    ## 4) Reusable Patterns
    - 用“无后端则 skip、有后端则强制执行”的测试骨架降低引入成本并避免误报。
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase5/logs/researcher.md`

- Run: run_2026_03_04_backend_decouple_phase5
  - Summary: merged delta at 2026-03-04T15:44:36Z
  - Source: runs/run_2026_03_04_backend_decouple_phase5/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_04_backend_decouple_phase5
    - Source Agent: researcher
    - Recorder: doc-writer
    
    ## 1) Delta Summary
    - 明确了“真实后端切换测试”的最早引入时点：首个非 MuJoCo 后端可实例化并完成 env 映射后的同一天。
    - 给出了 5 条仓库内可执行 gating conditions（实现、映射、渲染、依赖、smoke 配置）。
    
    ## 2) New Stable Preferences
    - 采用“两层测试策略”：先框架级常驻测试，再真实后端 smoke 自动激活。
    
    ## 3) Pitfalls Learned
    - 只实现 backend class 不足以开展真实切换测试，必须同时具备 env registry 映射与可运行依赖环境。
    
    ## 4) Reusable Patterns
    - 用“无后端则 skip、有后端则强制执行”的测试骨架降低引入成本并避免误报。
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase5/logs/researcher.md`

- Run: run_2026_03_04_backend_decouple_phase6
  - Summary: merged delta at 2026-03-04T15:48:33Z
  - Source: runs/run_2026_03_04_backend_decouple_phase6/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_04_backend_decouple_phase6
    - Source Agent: researcher
    - Recorder: doc-writer
    
    ## 1) Delta Summary
    - 建议新增统一的 backend readiness API，避免依赖异常文本来驱动测试门控。
    - 建议把真实 Genesis 切换测试改成“readiness 通过才执行，否则结构化 skip”。
    
    ## 2) New Stable Preferences
    - 对可选依赖后端（Genesis/Isaac/Newton）优先采用“状态探测 + 门控执行”，不要把缺依赖当失败。
    
    ## 3) Pitfalls Learned
    - 仅靠 `ImportError` 字符串判断会导致 CI 不稳定（信息变化即误报）。
    
    ## 4) Reusable Patterns
    - 三层测试矩阵：
      - L1: synthetic always-on
      - L2: real-backend gated by readiness + env var
      - L3: explicit skip with machine-readable reason
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase6/logs/researcher.md`

- Run: run_2026_03_04_backend_decouple_phase7
  - Summary: merged delta at 2026-03-04T15:58:52Z
  - Source: runs/run_2026_03_04_backend_decouple_phase7/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_04_backend_decouple_phase7
    - Source Agent: researcher
    - Recorder: researcher
    
    ## 1) Delta Summary
    - Repo already has backend decoupling, plugin loading, Genesis physics dependency check, and gated real-switch test entrypoint.
    - Core gap identified: no unified readiness diagnostic for actual backend switch (dependency + env mapping + render readiness).
    - Recommended next incremental step: add `backend_switch_readiness(...)` API and use it in real backend switch tests for deterministic skip/fail reasons without requiring Genesis installation.
    
    ## 2) New Stable Preferences
    - For backend bring-up, prioritize deterministic diagnostics that work in environments without optional runtimes installed.
    
    ## 3) Pitfalls Learned
    - `physics_backend_readiness("genesis")` alone can be misleading; it does not verify `ENV_REGISTRY` mapping or render backend executability.
    - Real-switch test gating currently depends on dynamic conditions and can skip without rich reason granularity.
    
    ## 4) Reusable Patterns
    - Compose readiness in layers:
      1) backend implementation/dependency,
      2) env mapping existence,
      3) render backend readiness,
      4) optional runtime gate env var.
    - Use test-only dummy env mappings to validate framework behavior independent of external engine installs.
    
    ## 5) Evidence Links
    - `mbrl/environments/backends/physics.py`
    - `mbrl/environments/backends/render.py`
    - `mbrl/environments/backends/registry.py`
    - `tests/test_environment_backends.py`
    - `requirements.txt`
    - `requirements.no_mujoco.txt`

- Run: run_2026_03_05_backend_decouple_phase8
  - Summary: merged delta at 2026-03-04T16:10:02Z
  - Source: runs/run_2026_03_05_backend_decouple_phase8/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_05_backend_decouple_phase8
    - Source Agent: researcher
    - Recorder: researcher
    
    ## 1) Delta Summary
    - Confirmed phase7 already has structured physics readiness and real-switch precheck, but still lacks unified switch readiness that composes physics + render + env mapping.
    - Proposed phase8 additive contract:
      - add `render_backend_readiness(...)`
      - add `backend_switch_readiness(...)`
      - extend diagnostics with `candidate_switch_readiness` while preserving existing phase7 keys/semantics.
    - Verified current workspace behavior: genesis physics not ready (`ImportError`), no non-MuJoCo mapping candidate, backend tests pass with expected gated skip.
    
    ## 2) New Stable Preferences
    - For backend-switch diagnostics, prefer additive schema evolution over in-place key semantics changes.
    - Keep real-switch default viability checks aligned with the actual test constructor path (`render_backend="none"`).
    
    ## 3) Pitfalls Learned
    - `physics_backend_readiness` alone can produce false confidence for switchability because mapping and render constraints are external to that API.
    - Replacing `candidate_readiness` payload shape would risk phase7 test/API breakage; introduce `candidate_switch_readiness` instead.
    
    ## 4) Reusable Patterns
    - Readiness composition pattern for backend bring-up:
      - `mapping.exists` check
      - `physics.ready` check
      - `render.ready` check
      - aggregate to `ready` + `blocking_checks[]`
    - Backward-compatible diagnostics extension pattern:
      - preserve existing keys and meanings
      - add new keys for richer phase8 detail.
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_05_backend_decouple_phase8/logs/researcher.md`
    - `mbrl/environments/backends/diagnostics.py`
    - `mbrl/environments/backends/physics.py`
    - `mbrl/environments/backends/render.py`
    - `mbrl/environments/backends/registry.py`
    - `tests/test_environment_backends.py`
