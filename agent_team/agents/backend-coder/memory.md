# Agent Memory: backend-coder

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
  - Source: runs/run_2026_03_04_backend_decouple_phase2/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_04_backend_decouple_phase2
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
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

- Run: run_2026_03_04_backend_decouple_phase2
  - Summary: merged delta at 2026-03-04T15:11:00Z
  - Source: runs/run_2026_03_04_backend_decouple_phase2/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_04_backend_decouple_phase2
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
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

- Run: run_2026_03_04_backend_decouple_phase3
  - Summary: merged delta at 2026-03-04T15:27:19Z
  - Source: runs/run_2026_03_04_backend_decouple_phase3/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_04_backend_decouple_phase3
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
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

- Run: run_2026_03_04_backend_decouple_phase4
  - Summary: merged delta at 2026-03-04T15:33:07Z
  - Source: runs/run_2026_03_04_backend_decouple_phase4/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_04_backend_decouple_phase4
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
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
  - Source: runs/run_2026_03_04_backend_decouple_phase5/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_04_backend_decouple_phase5
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
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
  - Source: runs/run_2026_03_04_backend_decouple_phase5/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_04_backend_decouple_phase5
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
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
  - Source: runs/run_2026_03_04_backend_decouple_phase6/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_04_backend_decouple_phase6
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
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
  - Source: runs/run_2026_03_04_backend_decouple_phase7/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_04_backend_decouple_phase7
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
    ## 1) Delta Summary
    - Added a backend diagnostics API/CLI that explicitly reports physics readiness and real backend switch test skip reasons.
    
    ## 2) New Stable Preferences
    - For optional backends (e.g., Genesis), provide machine-readable diagnostics before attempting runtime switching.
    
    ## 3) Pitfalls Learned
    - Sub-agent logs must follow repository-required section headers exactly; otherwise `check_run_logs.sh` fails.
    
    ## 4) Reusable Patterns
    - Mirror test gate order in diagnostics output so operator feedback and test behavior stay consistent.
    - Use lazy wrappers in package `__init__.py` for module entrypoints executed via `python -m`.
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase7/logs/backend-coder.md`
    - Subagent commit `3469dc34541b1ccdf070a1403eb2adaa354a545f`

- Run: run_2026_03_05_backend_decouple_phase8
  - Summary: merged delta at 2026-03-04T16:10:02Z
  - Source: runs/run_2026_03_05_backend_decouple_phase8/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_05_backend_decouple_phase8
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
    ## 1) Delta Summary
    - Added unified tuple-level readiness diagnostics (`env + physics + render`) and CLI parameters to query this tuple directly.
    
    ## 2) New Stable Preferences
    - Extend diagnostics APIs additively to keep prior phase JSON keys backward compatible.
    
    ## 3) Pitfalls Learned
    - In this workspace, `pytest` may be absent; keep fallback validation path using `python3 -m unittest` and CLI smoke checks.
    
    ## 4) Reusable Patterns
    - Tuple readiness schema pattern:
      - requested tuple
      - resolved tuple
      - per-dimension readiness (physics/render/mapping)
      - overall `ready`
      - actionable `next_actions`
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_05_backend_decouple_phase8/logs/backend-coder.md`
    - subagent commit `95e4a4a14023c91a75889973920e4872aeff5ee0`
