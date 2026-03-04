# Agent Memory: architect

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
  - Source: runs/run_2026_03_04_backend_decouple_phase2/memory/architect.delta.md
  - Notes:
    # Memory Delta: architect
    
    - Run ID: run_2026_03_04_backend_decouple_phase2
    - Source Agent: architect
    - Recorder: architect
    
    ## 1) Delta Summary
    - phase2 设计聚焦在 backend readiness metadata 与统一查询 API。
    
    ## 2) New Stable Preferences
    - 采用“接口先行 + 测试兜底 + 逐后端落地”的节奏。
    
    ## 3) Pitfalls Learned
    - 仅注册后端名称不够，缺少能力状态会导致配置治理困难。
    
    ## 4) Reusable Patterns
    - 后端类统一暴露 `display_name` 与 `implemented`。
    
    ## 5) Evidence Links
    - agent_team/runs/run_2026_03_04_backend_decouple_phase2/logs/architect.md

- Run: run_2026_03_04_backend_decouple_phase2
  - Summary: merged delta at 2026-03-04T15:11:00Z
  - Source: runs/run_2026_03_04_backend_decouple_phase2/memory/architect.delta.md
  - Notes:
    # Memory Delta: architect
    
    - Run ID: run_2026_03_04_backend_decouple_phase2
    - Source Agent: architect
    - Recorder: architect
    
    ## 1) Delta Summary
    - phase2 设计聚焦在 backend readiness metadata 与统一查询 API。
    
    ## 2) New Stable Preferences
    - 采用“接口先行 + 测试兜底 + 逐后端落地”的节奏。
    
    ## 3) Pitfalls Learned
    - 仅注册后端名称不够，缺少能力状态会导致配置治理困难。
    
    ## 4) Reusable Patterns
    - 后端类统一暴露 `display_name` 与 `implemented`。
    
    ## 5) Evidence Links
    - agent_team/runs/run_2026_03_04_backend_decouple_phase2/logs/architect.md

- Run: run_2026_03_04_backend_decouple_phase3
  - Summary: merged delta at 2026-03-04T15:27:19Z
  - Source: runs/run_2026_03_04_backend_decouple_phase3/memory/architect.delta.md
  - Notes:
    # Memory Delta: architect
    
    - Run ID: run_2026_03_04_backend_decouple_phase3
    - Source Agent: architect
    - Recorder: architect
    
    ## 1) Delta Summary
    - 产出最小插件加载机制设计：通过配置导入第三方模块，在导入期调用 `register_*` 完成后端注册。
    - 明确启动接入点：`mbrl/main.py` 中 `env_from_string(...)` 之前执行插件加载。
    
    ## 2) New Stable Preferences
    - 优先使用“模块导入即注册”的插件协议，避免引入复杂插件框架。
    - 默认不覆盖现有 backend，冲突由显式 `override=True` 控制。
    
    ## 3) Pitfalls Learned
    - 仅有注册函数不够，缺少统一加载入口时第三方无法无侵入接入。
    - 插件导入失败必须保留原始异常上下文，便于定位依赖缺失。
    
    ## 4) Reusable Patterns
    - `load_backend_plugins(modules: Iterable[str]) -> Sequence[str]`
    - 在主入口启动早期（构建 env 前）完成插件导入与注册。
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase3/logs/architect.md`

- Run: run_2026_03_04_backend_decouple_phase4
  - Summary: merged delta at 2026-03-04T15:33:07Z
  - Source: runs/run_2026_03_04_backend_decouple_phase4/memory/architect.delta.md
  - Notes:
    # Memory Delta: architect
    
    - Run ID: run_2026_03_04_backend_decouple_phase4
    - Source Agent: architect
    - Recorder: architect
    
    ## 1) Delta Summary
    - What changed in this run:
    
    ## 2) New Stable Preferences
    - 
    
    ## 3) Pitfalls Learned
    - 
    
    ## 4) Reusable Patterns
    - 
    
    ## 5) Evidence Links
    - 

- Run: run_2026_03_04_backend_decouple_phase5
  - Summary: merged delta at 2026-03-04T15:40:48Z
  - Source: runs/run_2026_03_04_backend_decouple_phase5/memory/architect.delta.md
  - Notes:
    # Memory Delta: architect
    
    - Run ID: run_2026_03_04_backend_decouple_phase5
    - Source Agent: architect
    - Recorder: architect
    
    ## 1) Delta Summary
    - What changed in this run:
    
    ## 2) New Stable Preferences
    - 
    
    ## 3) Pitfalls Learned
    - 
    
    ## 4) Reusable Patterns
    - 
    
    ## 5) Evidence Links
    - 

- Run: run_2026_03_04_backend_decouple_phase5
  - Summary: merged delta at 2026-03-04T15:44:36Z
  - Source: runs/run_2026_03_04_backend_decouple_phase5/memory/architect.delta.md
  - Notes:
    # Memory Delta: architect
    
    - Run ID: run_2026_03_04_backend_decouple_phase5
    - Source Agent: architect
    - Recorder: architect
    
    ## 1) Delta Summary
    - What changed in this run:
    
    ## 2) New Stable Preferences
    - 
    
    ## 3) Pitfalls Learned
    - 
    
    ## 4) Reusable Patterns
    - 
    
    ## 5) Evidence Links
    - 

- Run: run_2026_03_04_backend_decouple_phase6
  - Summary: merged delta at 2026-03-04T15:48:33Z
  - Source: runs/run_2026_03_04_backend_decouple_phase6/memory/architect.delta.md
  - Notes:
    # Memory Delta: architect
    
    - Run ID: run_2026_03_04_backend_decouple_phase6
    - Source Agent: architect
    - Recorder: architect
    
    ## 1) Delta Summary
    - What changed in this run:
    
    ## 2) New Stable Preferences
    - 
    
    ## 3) Pitfalls Learned
    - 
    
    ## 4) Reusable Patterns
    - 
    
    ## 5) Evidence Links
    - 

- Run: run_2026_03_04_backend_decouple_phase7
  - Summary: merged delta at 2026-03-04T15:58:52Z
  - Source: runs/run_2026_03_04_backend_decouple_phase7/memory/architect.delta.md
  - Notes:
    # Memory Delta: architect
    
    - Run ID: run_2026_03_04_backend_decouple_phase7
    - Source Agent: architect
    - Recorder: architect
    
    ## 1) Delta Summary
    - What changed in this run:
    
    ## 2) New Stable Preferences
    - 
    
    ## 3) Pitfalls Learned
    - 
    
    ## 4) Reusable Patterns
    - 
    
    ## 5) Evidence Links
    - 

- Run: run_2026_03_05_backend_decouple_phase8
  - Summary: merged delta at 2026-03-04T16:10:02Z
  - Source: runs/run_2026_03_05_backend_decouple_phase8/memory/architect.delta.md
  - Notes:
    # Memory Delta: architect
    
    - Run ID: run_2026_03_05_backend_decouple_phase8
    - Source Agent: architect
    - Recorder: architect
    
    ## 1) Delta Summary
    - What changed in this run:
    
    ## 2) New Stable Preferences
    - 
    
    ## 3) Pitfalls Learned
    - 
    
    ## 4) Reusable Patterns
    - 
    
    ## 5) Evidence Links
    - 
