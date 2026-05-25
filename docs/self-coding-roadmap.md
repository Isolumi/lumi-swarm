# Lumi Swarm Self-Coding Roadmap

This project is meant to be coded by the human developer. Assistants should guide, explain, review, and help debug, but should not implement code unless explicitly asked.

Use this file as the handoff anchor for any new chat.

## New Chat Instructions

When starting a new chat, say:

```text
Read /home/isolumi/CS/lumi-swarm/docs/self-coding-roadmap.md and guide me through Lumi Swarm. Do not code for me unless I explicitly ask. Help me choose the next step, explain concepts, review my code, and suggest verification commands.
```

Relevant project docs:

- Design spec: `/home/isolumi/CS/lumi-swarm/docs/superpowers/specs/2026-05-25-cuda-agent-swarm-design.md`
- Detailed implementation reference: `/home/isolumi/CS/lumi-swarm/docs/superpowers/plans/2026-05-25-cuda-agent-swarm-first-milestone.md`
- Human-guided roadmap: this file

## Assistant Rules

Assistants should:

- Ask which phase the developer is on before giving detailed guidance.
- Explain one step at a time.
- Prefer concept explanations, API sketches, pseudocode, diagrams, review notes, and test suggestions.
- Let the developer write the actual source files.
- Inspect code only when the developer asks for review or debugging help.
- Give exact commands for building, testing, profiling, and debugging.
- Treat CUDA correctness and measurement as first-class goals.

Assistants should not:

- Write implementation files for the developer.
- Replace the developer's design choices without discussion.
- Expand scope into a generic agent framework before the CUDA serving core works.
- Skip verification claims. Every "done" phase needs commands and observed results.

## Project Direction

Lumi Swarm is a CUDA-first systems project. The agent swarm is the workload that proves the GPU serving design.

The core idea:

- Many specialist coding agents submit generation/retrieval work.
- Agents are logical: planner, implementer, reviewer, verifier.
- GPU resources are shared.
- The CUDA core focuses on batching, KV-cache management, sampling, retrieval, and metrics.

## Phase 0: Orientation And Constraints

Goal: Make sure the developer understands the project shape before writing code.

Key concepts:

- Logical agents vs separate model processes.
- CUDA serving layer vs agent orchestration layer.
- Continuous batching.
- KV-cache blocks/pages.
- CPU reference implementation before CUDA kernels.
- Benchmarks as product features, not afterthoughts.

Done when:

- The developer can explain what the CUDA core owns.
- The developer can explain what the agent runtime owns.
- The developer can explain why this starts with a deterministic backend.

Good assistant prompts:

- "Explain continuous batching using a coding-agent workload."
- "Explain paged KV cache like I am about to implement it."
- "What should the CUDA core own versus the orchestrator?"

## Phase 1: Build Skeleton

Goal: Create the minimal CMake/CUDA/C++ repo shape and prove it builds.

Deliverables:

- Root `CMakeLists.txt`.
- `include/`, `src/`, `src/cuda/`, `tests/`, and `apps/` directories.
- A tiny CTest smoke test.
- `.gitignore`.
- Basic build instructions.

Done when:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

passes.

Good assistant prompts:

- "Review my CMake skeleton."
- "Why is my CUDA target not compiling?"
- "What should the first smoke test prove?"

## Phase 2: Core Host API And Deterministic Backend

Goal: Define the stable API that future real model backends will use.

Deliverables:

- Core request/result types.
- Backend interface.
- Deterministic backend that emits predictable token IDs.
- Tests proving request IDs, token budgets, and deterministic output.

Done when:

- A generation request can be submitted to the deterministic backend.
- The same request produces the same token sequence every time.
- Token budget behavior is tested.

Good assistant prompts:

- "Review my request/result types before I build the scheduler."
- "What fields should a generation request carry for future KV-cache reuse?"
- "Help me design deterministic backend tests."

## Phase 3: Continuous Batching Scheduler

Goal: Implement the first real serving-system component.

Deliverables:

- Request queue.
- Active batch.
- Scheduler step function.
- Completion and cancellation paths.
- Metrics for submitted, queued, active, completed, cancelled, emitted tokens, and last batch size.

Done when:

- 1, 4, and 16 synthetic requests can be served through the scheduler.
- The scheduler respects max batch size.
- Cancellation works for queued and active requests.
- Tests prove queue-to-active-to-complete transitions.

Good assistant prompts:

- "Review my scheduler state machine."
- "Help me find edge cases in request cancellation."
- "What metrics should I expose before writing benchmarks?"

## Phase 4: Paged KV-Cache Accounting

Goal: Build cache ownership and block accounting before integrating real attention.

Deliverables:

- Cache block allocator.
- Cache handle type.
- Allocate/free behavior.
- Fragmentation metric.
- Scheduler integration that allocates cache on activation and frees on completion/cancel.

Done when:

- Cache allocation and free are covered by tests.
- Failed allocation is handled explicitly.
- Scheduler tests prove cache blocks are released after request completion.
- Metrics report total, used, free, largest free run, and fragmentation.

Good assistant prompts:

- "Review my KV-cache allocator for fragmentation bugs."
- "What is a good first paged-cache data structure?"
- "How should scheduler and cache ownership interact?"

## Phase 5: CUDA Sampling Kernels

Goal: Write the first serving-relevant CUDA kernel with CPU reference coverage.

Deliverables:

- CPU reference for top-k sampling.
- CUDA kernel for batched top-k token selection.
- Test comparing CPU and CUDA output.
- Basic benchmark timing.

Done when:

- CPU and CUDA outputs match for deterministic test cases.
- Invalid config inputs are rejected.
- Kernel launches are checked for CUDA errors.
- The benchmark can vary batch size and vocab size.

Good assistant prompts:

- "Review my CPU reference before I write the CUDA kernel."
- "Explain a simple top-k CUDA strategy and its limits."
- "Help debug this CUDA launch or memory-copy error."

## Phase 6: CUDA Retrieval Scoring

Goal: Add a second CUDA workload that maps directly to repo-memory search.

Deliverables:

- CPU dot-product scoring reference.
- CUDA dot-product scoring kernel.
- Test comparing CPU and CUDA scores.
- Benchmark for query count, document count, and embedding dimension.

Done when:

- CPU and CUDA scores match within a stated tolerance.
- Retrieval benchmark reports timing and throughput.
- The retrieval API can be used independently of the agent harness.

Good assistant prompts:

- "Review my retrieval memory layout."
- "How should I compare floating point CPU and GPU results?"
- "What benchmark dimensions are realistic for repo retrieval?"

## Phase 7: Thin Coding-Agent Harness

Goal: Create the smallest real coding-agent workload without letting orchestration dominate the project.

Deliverables:

- Planner, implementer, reviewer, verifier roles.
- Scripted or deterministic first agent flow.
- Controlled fixture repo.
- Harness that applies a simple patch and runs a verification command.
- Run trace showing agent steps and scheduler metrics.

Done when:

- A fixture repo starts failing.
- The scripted swarm proposes or applies a bounded fix.
- Verification passes after the fix.
- All generation work still routes through the scheduler API.

Good assistant prompts:

- "Help me keep the harness thin."
- "Review whether this agent layer is becoming too large."
- "What should the run trace record?"

## Phase 8: Benchmarks And First Report

Goal: Prove the project as a CUDA serving system with repeatable measurements.

Deliverables:

- Synthetic swarm benchmark for 1, 4, 16, and 64 concurrent requests.
- Sampling benchmark.
- Retrieval benchmark.
- First milestone verification report.

Done when:

- Tests pass in Debug and Release.
- Demo exits successfully.
- Benchmarks produce stable, repeatable output.
- Report includes tokens/sec, scheduler steps, cache usage, and kernel timings where available.

Good assistant prompts:

- "Review my benchmark methodology."
- "What metrics are missing for this phase?"
- "Help me interpret these benchmark results."

## Phase 9: Real Model Backend Decision

Goal: Decide how to connect the serving core to real local inference.

Do not start this until Phases 1-8 are working.

Options:

- Integrate a narrow decoder-only backend using existing GEMM/library support.
- Bind to an existing inference engine while keeping Lumi's scheduler/cache/metrics layer.
- Implement selected transformer serving kernels only after profiling shows why.

Done when:

- One target model family is chosen.
- Weight format and tokenizer path are chosen.
- The boundary between external library code and Lumi-owned CUDA code is written down.

Good assistant prompts:

- "Help me choose the first model backend target."
- "What should I avoid implementing from scratch yet?"
- "How do I preserve my scheduler API while adding a real model?"

## Current Recommended Next Step

Start with Phase 1.

Before writing code, ask an assistant:

```text
I am starting Phase 1 of Lumi Swarm. Read docs/self-coding-roadmap.md. Guide me through the build skeleton without writing the code for me.
```
