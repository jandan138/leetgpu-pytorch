# Agent Memory: test-engineer

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
  - Source: runs/run_2026_03_04_backend_decouple_phase2/memory/test-engineer.delta.md
  - Notes:
    # Memory Delta: test-engineer
    
    - Run ID: run_2026_03_04_backend_decouple_phase2
    - Source Agent: test-engineer
    - Recorder: test-engineer
    
    ## 1) Delta Summary
    - 本轮采用 compileall 快速验证后端改动无语法回归。
    
    ## 2) New Stable Preferences
    - 在无外部依赖时，先做编译/导入检查，再补纯 Python 单测。
    
    ## 3) Pitfalls Learned
    - 若立即收口，需在日志中明确“未执行的测试项与后续计划”。
    
    ## 4) Reusable Patterns
    - 对后端抽象改动至少执行 compileall 与工厂函数导入检查。
    
    ## 5) Evidence Links
    - agent_team/runs/run_2026_03_04_backend_decouple_phase2/logs/test-engineer.md

- Run: run_2026_03_04_backend_decouple_phase2
  - Summary: merged delta at 2026-03-04T15:11:00Z
  - Source: runs/run_2026_03_04_backend_decouple_phase2/memory/test-engineer.delta.md
  - Notes:
    # Memory Delta: test-engineer
    
    - Run ID: run_2026_03_04_backend_decouple_phase2
    - Source Agent: test-engineer
    - Recorder: test-engineer
    
    ## 1) Delta Summary
    - 本轮采用 compileall 快速验证后端改动无语法回归。
    
    ## 2) New Stable Preferences
    - 在无外部依赖时，先做编译/导入检查，再补纯 Python 单测。
    
    ## 3) Pitfalls Learned
    - 若立即收口，需在日志中明确“未执行的测试项与后续计划”。
    
    ## 4) Reusable Patterns
    - 对后端抽象改动至少执行 compileall 与工厂函数导入检查。
    
    ## 5) Evidence Links
    - agent_team/runs/run_2026_03_04_backend_decouple_phase2/logs/test-engineer.md

- Run: run_2026_03_04_backend_decouple_phase3
  - Summary: merged delta at 2026-03-04T15:27:19Z
  - Source: runs/run_2026_03_04_backend_decouple_phase3/memory/test-engineer.delta.md
  - Notes:
    # Memory Delta: test-engineer
    
    - Run ID: run_2026_03_04_backend_decouple_phase3
    - Source Agent: test-engineer
    - Recorder: test-engineer
    
    ## 1) Delta Summary
    - 新增插件加载场景单测与示例插件模块，覆盖 runtime 注册路径。
    
    ## 2) New Stable Preferences
    - 测试优先使用仓库内 dummy env，避免引入 MuJoCo/NumPy 等额外依赖。
    
    ## 3) Pitfalls Learned
    - 插件辅助模块的导出函数名需与 `__init__.py` 一致，否则测试导入会直接失败。
    
    ## 4) Reusable Patterns
    - 用 `tests/backend_plugin_example.py` 做 import-time registration 的可复用模板。
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase3/logs/test-engineer.md`

- Run: run_2026_03_04_backend_decouple_phase4
  - Summary: merged delta at 2026-03-04T15:33:07Z
  - Source: runs/run_2026_03_04_backend_decouple_phase4/memory/test-engineer.delta.md
  - Notes:
    # Memory Delta: test-engineer
    
    - Run ID: run_2026_03_04_backend_decouple_phase4
    - Source Agent: test-engineer
    - Recorder: test-engineer
    
    ## 1) Delta Summary
    - 给出 plugin loader 的最小测试集合，重点覆盖幂等与错误处理。
    - 明确无 MuJoCo 依赖的测试实现路径。
    
    ## 2) New Stable Preferences
    - 优先用 `unittest` + dummy env 做机制层回归，避免外部引擎依赖影响 CI 稳定性。
    
    ## 3) Pitfalls Learned
    - “导入即注册”插件在重复加载场景容易出现冲突；测试必须覆盖重复调用路径。
    
    ## 4) Reusable Patterns
    - 用 `backend_plugin_modules` + dummy plugin module 验证运行时注册链路。
    - 用显式 `ImportError` 断言保证错误可观测性。
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase4/logs/test-engineer.md`

- Run: run_2026_03_04_backend_decouple_phase5
  - Summary: merged delta at 2026-03-04T15:40:48Z
  - Source: runs/run_2026_03_04_backend_decouple_phase5/memory/test-engineer.delta.md
  - Notes:
    # Memory Delta: test-engineer
    
    - Run ID: run_2026_03_04_backend_decouple_phase5
    - Source Agent: test-engineer
    - Recorder: test-engineer
    
    ## 1) Delta Summary
    - 输出了三层后端切换测试矩阵（L1 synthetic / L2 real activation / L3 skip policy）。
    - 明确了真实后端测试的激活条件：实现标记 + registry 映射 + 依赖导入 + runner 能力。
    - 明确了 skip 策略与原因模板，避免 CI 假失败。
    
    ## 2) New Stable Preferences
    - 先保持 L1 always-on，L2 按条件自动激活，L3 显式 skip 并带原因。
    - 真实后端 smoke 先从单环境最小 reset/step/render 开始。
    
    ## 3) Pitfalls Learned
    - 仅靠 backend 名称判断不够，必须同时校验 env registry 映射和依赖可用性。
    - 没有 skip 原因会导致“可用性问题”被误判为“功能回归”。
    
    ## 4) Reusable Patterns
    - 使用 backend 元数据 `implemented` 作为激活测试的第一道门槛。
    - 采用 `unittest.skipTest("reason")` 将不可运行场景标准化。
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase5/logs/test-engineer.md`
    - `tests/test_environment_backends.py`
    - `mbrl/environments/backends/physics.py`
    - `mbrl/environments/backends/render.py`

- Run: run_2026_03_04_backend_decouple_phase5
  - Summary: merged delta at 2026-03-04T15:44:36Z
  - Source: runs/run_2026_03_04_backend_decouple_phase5/memory/test-engineer.delta.md
  - Notes:
    # Memory Delta: test-engineer
    
    - Run ID: run_2026_03_04_backend_decouple_phase5
    - Source Agent: test-engineer
    - Recorder: test-engineer
    
    ## 1) Delta Summary
    - 输出了三层后端切换测试矩阵（L1 synthetic / L2 real activation / L3 skip policy）。
    - 明确了真实后端测试的激活条件：实现标记 + registry 映射 + 依赖导入 + runner 能力。
    - 明确了 skip 策略与原因模板，避免 CI 假失败。
    
    ## 2) New Stable Preferences
    - 先保持 L1 always-on，L2 按条件自动激活，L3 显式 skip 并带原因。
    - 真实后端 smoke 先从单环境最小 reset/step/render 开始。
    
    ## 3) Pitfalls Learned
    - 仅靠 backend 名称判断不够，必须同时校验 env registry 映射和依赖可用性。
    - 没有 skip 原因会导致“可用性问题”被误判为“功能回归”。
    
    ## 4) Reusable Patterns
    - 使用 backend 元数据 `implemented` 作为激活测试的第一道门槛。
    - 采用 `unittest.skipTest("reason")` 将不可运行场景标准化。
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase5/logs/test-engineer.md`
    - `tests/test_environment_backends.py`
    - `mbrl/environments/backends/physics.py`
    - `mbrl/environments/backends/render.py`

- Run: run_2026_03_04_backend_decouple_phase6
  - Summary: merged delta at 2026-03-04T15:48:33Z
  - Source: runs/run_2026_03_04_backend_decouple_phase6/memory/test-engineer.delta.md
  - Notes:
    # Memory Delta: test-engineer
    
    - Run ID: run_2026_03_04_backend_decouple_phase6
    - Source Agent: test-engineer
    - Recorder: test-engineer
    
    ## 1) Delta Summary
    - Defined minimal backend-readiness tests with Genesis focus, fully independent of MuJoCo runtime.
    - Confirmed two key assertions: (1) genesis switch path works with `skip_dependency_check`; (2) default genesis path fails fast with clear `ImportError` when dependency missing.
    
    ## 2) New Stable Preferences
    - Prefer deterministic synthetic backend mappings for readiness tests over environment-dependent integration tests.
    - Keep real-backend tests gated behind explicit env flags and skip messages.
    
    ## 3) Pitfalls Learned
    - Dependency-sensitive assertions can become flaky across heterogeneous runners if missing-package assumptions are hardcoded.
    - Exception-message assertions should stay minimal and stable (e.g., include backend name only).
    
    ## 4) Reusable Patterns
    - Register dummy env mapping with target backend, then validate constructor behavior under different backend options.
    - Use gated test (`ENABLE_REAL_BACKEND_TESTS`) for real switch smoke while keeping base suite always-on.
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase6/logs/test-engineer.md`
    - `tests/test_environment_backends.py`
    - `mbrl/environments/backends/physics.py`

- Run: run_2026_03_04_backend_decouple_phase7
  - Summary: merged delta at 2026-03-04T15:58:52Z
  - Source: runs/run_2026_03_04_backend_decouple_phase7/memory/test-engineer.delta.md
  - Notes:
    # Memory Delta: test-engineer
    
    - Run ID: run_2026_03_04_backend_decouple_phase7
    - Source Agent: test-engineer
    - Recorder: test-engineer
    
    ## 1) Delta Summary
    - Updated backend tests so real-switch skip reasons include readiness context (`backend`, `error_type`, `reason`).
    
    ## 2) New Stable Preferences
    - For optional runtime dependencies, prefer deterministic mock-based readiness tests over environment-coupled assertions.
    
    ## 3) Pitfalls Learned
    - Generic skip reasons hide actionable failures; include concrete readiness diagnostics in skip details.
    
    ## 4) Reusable Patterns
    - Centralize candidate selection in helper methods for gated tests to keep branch logic and messaging consistent.
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase7/logs/test-engineer.md`
    - Subagent commit `49053a0c1701e8aa319095837b59c72a86cf81c1`

- Run: run_2026_03_05_backend_decouple_phase8
  - Summary: merged delta at 2026-03-04T16:10:02Z
  - Source: runs/run_2026_03_05_backend_decouple_phase8/memory/test-engineer.delta.md
  - Notes:
    # Memory Delta: test-engineer
    
    - Run ID: run_2026_03_05_backend_decouple_phase8
    - Source Agent: test-engineer
    - Recorder: test-engineer
    
    ## 1) Delta Summary
    - Added deterministic tests for unified switch readiness diagnostics across Genesis-absent and synthetic-ready scenarios.
    
    ## 2) New Stable Preferences
    - For optional backends, prioritize deterministic mock/registry-based tests over host-dependent runtime checks.
    
    ## 3) Pitfalls Learned
    - Test-only branch may fail before implementation symbols are merged; this is acceptable if failure reason is explicit and expected.
    
    ## 4) Reusable Patterns
    - For diagnostics tests, isolate global registry state using setup/teardown snapshots to avoid cross-test contamination.
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_05_backend_decouple_phase8/logs/test-engineer.md`
    - subagent commit `3a975500be3dc37b1d21b061bb011b5aa3de9288`
