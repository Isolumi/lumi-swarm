# CUDA Agent Swarm Design

## Overview

Lumi Swarm is a CUDA-first systems project for serving many specialist coding agents through a shared local inference layer. The product shape is an autonomous coding swarm, but the engineering center of gravity is GPU serving: continuous batching, KV-cache management, sampling kernels, retrieval acceleration, and profiling under realistic multi-agent workloads.

The first working system should demonstrate planner, implementer, reviewer, and verifier agents collaborating on a coding task while exercising a CUDA-owned serving core. Agents are logical entities with distinct prompts, state, memory, permissions, and task roles. GPU resources are shared across agents instead of loading a separate model instance per agent.

## Goals

- Build a CUDA/C++ serving core for many concurrent local LLM requests.
- Optimize the serving path for specialist coding-agent workloads.
- Support autonomous coding loops: plan, edit, review, test, revise, and produce a final patch.
- Measure the impact of batching, KV-cache reuse, sampling kernels, and retrieval acceleration.
- Keep the agent orchestrator thin enough that the project remains primarily a CUDA project.

## Non-Goals For The First Milestone

- Full custom transformer inference from scratch.
- Distributed multi-GPU serving.
- Fine-tuning or training.
- A web UI.
- Complex sandboxing or untrusted code execution isolation.
- Supporting many model families at once.

## Architecture

The system has three layers:

1. CUDA serving core
   - Owns GPU memory, model-serving state, request batching, decode scheduling, KV-cache layout, sampling kernels, profiling, and benchmark instrumentation.
   - Exposes a narrow host API for submitting generation requests, stepping or running the scheduler, streaming tokens, managing cache handles, and reading metrics.
   - Uses cuBLAS, CUTLASS, or a narrow backend adapter where appropriate for transformer math while custom CUDA work focuses on serving bottlenecks.

2. Agent runtime
   - Defines specialist coding agents: planners, implementers, reviewers, and verifiers.
   - Gives each agent a role prompt, task state, memory references, retry budget, permissions, and output contract.
   - Sends model and retrieval work to the shared CUDA serving core. Agents do not own separate model processes by default.

3. Coding-work harness
   - Connects the swarm to a local repository.
   - Reads files, builds context bundles, applies candidate patches, runs verification commands, stores logs, and emits the final diff.
   - Is the only layer allowed to touch the filesystem, apply edits, or execute commands.

The central boundary is that agents propose actions, while the harness executes controlled actions. The CUDA serving core is isolated from repository semantics and focuses on throughput, latency, memory behavior, and inference-service correctness.

## Core CUDA Scope

The first serious version focuses on serving many concurrent decode requests efficiently rather than implementing every transformer operation from scratch.

### Continuous Batching

Agent requests arrive at different times with different prompt lengths and token budgets. The scheduler should merge compatible active requests into GPU-friendly prefill and decode steps. It should track queue depth, active batch size, request age, priority, time-to-first-token, decode latency, and tokens per second.

### KV-Cache Management

KV-cache management is a main project pillar. The serving core should use a paged or block-based design so long-running agents, reviewers, retry loops, and shared context prefixes can reuse memory without wasting VRAM.

The cache subsystem should track:

- cache allocation and free operations
- request-to-cache ownership
- reusable context handles
- eviction decisions
- fragmentation
- VRAM usage
- cache hit and reuse rates
- leak detection in tests

### Sampling And Selection Kernels

The project should implement CUDA kernels for serving-relevant logits processing:

- temperature scaling
- top-k or top-p filtering
- repetition penalties
- token selection
- batched random sampling support

These kernels are smaller than full attention kernels but directly relevant to many-agent serving. Each kernel should have a CPU reference implementation and benchmark coverage.

### Embedding And Retrieval Acceleration

Coding agents need fast access to repo memory: file chunks, symbols, previous observations, plan fragments, review notes, and test failures. CUDA should accelerate vector similarity search, batched scoring, clustering, or reranking.

The retrieval layer should be usable independently by the orchestrator and reproducible in benchmarks without running the full coding harness.

### Profiling And Benchmarks

Every CUDA feature should expose metrics suitable for performance work:

- tokens per second
- time-to-first-token
- p50 and p95 latency
- queue depth
- active batch size
- scheduler overhead
- VRAM usage
- cache hit rate
- cache fragmentation
- kernel timings

Benchmarks should compare naive per-agent serving against batched serving under synthetic coding-agent workloads.

## Agent-Swarm Behavior

The first autonomous coding swarm should be small but real.

### Planner Agents

Multiple planner agents inspect the task and repository context, then propose implementation plans. The orchestrator compares or merges their plans into bounded implementation tasks.

### Implementer Agents

Implementers receive bounded tasks from the selected plan. They generate candidate patches from scoped context bundles containing the task, relevant files, constraints, and planner notes. They should not apply changes directly.

### Reviewer Agents

Reviewers inspect proposed diffs from different angles, including correctness, tests, architecture, performance, and safety. Their feedback becomes structured requirements for another implementation pass when needed.

### Verifier Agent

The verifier runs configured commands through the harness, summarizes failures, and decides whether another loop is needed based on verification output and retry limits.

### Orchestrator

The orchestrator owns task state, agent scheduling, conflict resolution, retry limits, and final acceptance. It creates agent jobs, routes outputs, asks the CUDA serving core for generation and retrieval work, and records a run trace.

## Data Flow

A coding task enters through a CLI or API with a target repository, instructions, and verification command. The harness builds a context bundle from relevant files, repository maps, existing tests, and retrieval results. The orchestrator creates specialist agent jobs from that bundle.

Each generation request sent to the CUDA serving core includes:

- role or system prompt
- task context
- priority
- maximum token budget
- cache handle when reusing context
- streaming callback or output buffer

The CUDA core handles batching, prefill and decode scheduling, KV-cache allocation, sampling, and metrics. It returns generated text or structured action proposals to the orchestrator.

The orchestrator routes outputs:

- planner outputs become implementation tasks
- implementer outputs become candidate patches
- reviewer outputs become critique and required changes
- verifier outputs become pass or fail state plus logs

The harness applies controlled patches and runs configured commands. The final output is a patch, a run trace, verification results, and CUDA performance metrics.

## First Milestone

The first milestone should prove the CUDA-serving shape without trying to beat mature inference servers immediately.

Deliverables:

1. Project scaffold
   - CUDA/C++ core
   - thin host API
   - CLI runner
   - benchmark harness
   - small agent-runtime layer

2. Narrow model backend plus deterministic test backend
   - Target one decoder-only local model shape for the first real integration, using existing GEMM/library support where appropriate while the project-owned CUDA work focuses on serving bottlenecks.
   - Provide a deterministic mock backend for scheduler, KV-cache, sampling, and benchmark tests so core behavior is reproducible without depending on model quality.
   - Keep the CUDA interfaces real from the start: generation requests, cache handles, scheduler steps, token outputs, and metrics should use the same host API for both backends.

3. Continuous batching scheduler
   - Support multiple concurrent generation requests from planner, implementer, reviewer, and verifier agents.
   - Track queue depth, active batch size, latency, time-to-first-token, and tokens per second.

4. Paged KV-cache prototype
   - Allocate, reuse, and free cache blocks per request or agent.
   - Report VRAM usage, fragmentation, cache hits, reuse, and eviction decisions.

5. CUDA sampling kernels
   - Implement temperature handling, top-k or top-p filtering, and token selection.
   - Validate against CPU references and benchmark batched workloads.

6. Repo coding demo
   - Run a small autonomous coding loop: plan, patch, review, test, revise once, and output a final diff.

7. Performance report
   - Compare naive per-agent serving with batched serving under a synthetic coding-agent workload.

## Testing Strategy

CUDA correctness should be verified with CPU references and deterministic fixtures wherever possible.

Required tests:

- scheduler state transitions
- queue ordering and priority behavior
- request cancellation
- token budget handling
- sampling kernels against CPU references
- KV-cache allocation, reuse, eviction, fragmentation behavior, and leak detection
- retrieval scoring against CPU references
- CLI smoke tests for the coding harness

Required benchmark scenarios:

- 1 concurrent agent request
- 4 concurrent agent requests
- 16 concurrent agent requests
- 64 concurrent agent requests
- long-context reviewer workload
- retry-heavy implementation workload
- retrieval-heavy repository-analysis workload

## Success Criteria

The first milestone is successful when:

- The CUDA serving core accepts concurrent agent generation requests through a stable host API.
- The scheduler demonstrates continuous batching behavior with visible throughput and latency metrics.
- The KV-cache prototype allocates, reuses, evicts, and reports cache state without leaks in tests.
- CUDA sampling kernels match CPU reference behavior within documented tolerances.
- A small fixture repository can be processed by the autonomous coding loop: plan, patch, review, test, revise if needed, and produce a final diff.
- The benchmark harness can replay swarm-like request patterns without the repo-editing layer.
- A performance report compares naive sequential serving with batched serving and reports tokens per second, time-to-first-token, p50 and p95 latency, VRAM usage, cache hit rate, and scheduler overhead.

The practical definition of done is that Lumi Swarm demonstrates a CUDA-owned serving layer coordinating many specialist coding-agent requests more efficiently than naive sequential generation, with reproducible metrics and a working autonomous coding demo.
