# CUDA Agent Swarm First Milestone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable CUDA-first agent swarm milestone: a C++/CUDA serving core with continuous batching, paged KV-cache accounting, sampling/retrieval kernels, a deterministic backend, a thin coding-agent harness, and repeatable benchmark output.

**Architecture:** Start with a deterministic local backend so scheduler, cache, CUDA kernels, and agent flow can be tested without depending on model quality. Keep agent behavior thin and scripted for the first demo while all requests pass through the same host API that a real decoder-only model backend will use.

**Tech Stack:** CMake 3.31, CUDA 13.0, C++20, CTest, standard library only for the first milestone.

---

## File Structure

- `CMakeLists.txt`: root build definition, C++/CUDA standards, libraries, apps, tests.
- `.gitignore`: build artifacts, cache files, generated run output.
- `README.md`: build, test, benchmark, and demo commands.
- `include/lumi/core/types.hpp`: request IDs, token IDs, request config, generated token chunks, request results.
- `include/lumi/core/metrics.hpp`: scheduler, cache, and benchmark metrics structs.
- `include/lumi/core/backend.hpp`: model backend interface plus deterministic backend declaration.
- `include/lumi/core/scheduler.hpp`: continuous batching scheduler API.
- `include/lumi/core/kv_cache.hpp`: paged KV-cache allocator API.
- `include/lumi/core/sampling.hpp`: CPU sampling reference and CUDA wrapper API.
- `include/lumi/core/retrieval.hpp`: CPU retrieval reference and CUDA wrapper API.
- `include/lumi/agent/agent.hpp`: coding-agent roles, jobs, outputs, and run trace types.
- `include/lumi/agent/harness.hpp`: repository harness API for controlled file edits and verification commands.
- `src/core/mock_backend.cpp`: deterministic backend implementation.
- `src/core/scheduler.cpp`: queueing, activation, batching, cancellation, completion, metrics.
- `src/core/kv_cache.cpp`: block allocation, free, reuse handles, fragmentation metrics.
- `src/core/sampling.cpp`: CPU sampling reference plus CUDA wrapper dispatch.
- `src/core/retrieval.cpp`: CPU retrieval reference plus CUDA wrapper dispatch.
- `src/cuda/sampling_kernels.cu`: CUDA top-k sampling kernel.
- `src/cuda/retrieval_kernels.cu`: CUDA dot-product retrieval kernel.
- `src/agent/agent.cpp`: thin planner/implementer/reviewer/verifier orchestration.
- `src/agent/harness.cpp`: fixture repo creation, controlled file edits, verification command execution.
- `apps/lumi_swarm_cli.cpp`: `demo` command for a small autonomous coding loop.
- `apps/lumi_swarm_bench.cpp`: synthetic swarm workload benchmark command.
- `tests/test_main.hpp`: tiny assertion helpers.
- `tests/test_build.cpp`: build smoke test.
- `tests/test_mock_backend.cpp`: deterministic backend tests.
- `tests/test_scheduler.cpp`: scheduler batching, priority, cancellation, token-budget tests.
- `tests/test_kv_cache.cpp`: cache allocation, reuse, eviction, fragmentation tests.
- `tests/test_sampling.cpp`: CPU/CUDA sampling correctness tests.
- `tests/test_retrieval.cpp`: CPU/CUDA retrieval correctness tests.
- `tests/test_agent_demo.cpp`: fixture coding loop smoke test.

---

### Task 1: Build Scaffold And Smoke Test

**Files:**
- Create: `.gitignore`
- Create: `CMakeLists.txt`
- Create: `README.md`
- Create: `tests/test_main.hpp`
- Create: `tests/test_build.cpp`

- [ ] **Step 1: Write the root CMake file**

Create `CMakeLists.txt` with:

```cmake
cmake_minimum_required(VERSION 3.25)

project(lumi_swarm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

add_library(lumi_core INTERFACE)
target_include_directories(lumi_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(lumi_build_smoke tests/test_build.cpp)
target_link_libraries(lumi_build_smoke PRIVATE lumi_core)
add_test(NAME lumi_build_smoke COMMAND lumi_build_smoke)
```

- [ ] **Step 2: Add build artifact ignores**

Create `.gitignore` with:

```gitignore
build/
cmake-build-*/
.cache/
compile_commands.json
run-output/
*.log
```

- [ ] **Step 3: Add tiny test helpers**

Create `tests/test_main.hpp` with:

```cpp
#pragma once

#include <cstdlib>
#include <iostream>
#include <string_view>

#define LUMI_CHECK(condition)                                                     \
  do {                                                                            \
    if (!(condition)) {                                                           \
      std::cerr << "CHECK failed: " #condition << " at " << __FILE__ << ":"      \
                << __LINE__ << "\n";                                             \
      return EXIT_FAILURE;                                                        \
    }                                                                             \
  } while (false)

inline int lumi_test_pass(std::string_view name) {
  std::cout << "[PASS] " << name << "\n";
  return EXIT_SUCCESS;
}
```

- [ ] **Step 4: Add the smoke test**

Create `tests/test_build.cpp` with:

```cpp
#include "test_main.hpp"

int main() {
  LUMI_CHECK(1 + 1 == 2);
  return lumi_test_pass("build smoke");
}
```

- [ ] **Step 5: Add initial README commands**

Create `README.md` with:

````markdown
# Lumi Swarm

Lumi Swarm is a CUDA-first local agent-swarm project for serving many specialist coding agents through a shared GPU-backed inference layer.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```
````

- [ ] **Step 6: Configure, build, and test**

Run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: CMake configures with CUDA enabled, the smoke test builds, and `100% tests passed`.

- [ ] **Step 7: Commit**

```bash
git add .gitignore CMakeLists.txt README.md tests/test_main.hpp tests/test_build.cpp
git commit -m "build: add CMake scaffold"
```

---

### Task 2: Core Types And Deterministic Backend

**Files:**
- Create: `include/lumi/core/types.hpp`
- Create: `include/lumi/core/metrics.hpp`
- Create: `include/lumi/core/backend.hpp`
- Create: `src/core/mock_backend.cpp`
- Create: `tests/test_mock_backend.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing backend test**

Create `tests/test_mock_backend.cpp` with:

```cpp
#include "lumi/core/backend.hpp"
#include "test_main.hpp"

#include <vector>

int main() {
  lumi::core::DeterministicBackend backend(/*vocab_size=*/128);

  lumi::core::GenerationRequest request;
  request.id = 7;
  request.prompt = "planner: inspect repo";
  request.max_new_tokens = 3;

  auto first = backend.prefill(request);
  LUMI_CHECK(first.tokens.size() == 1);
  LUMI_CHECK(first.finished == false);

  auto second = backend.decode(request, first.tokens);
  auto third = backend.decode(request, second.tokens);

  LUMI_CHECK(second.tokens.size() == 2);
  LUMI_CHECK(third.tokens.size() == 3);
  LUMI_CHECK(third.finished == true);
  LUMI_CHECK(first.tokens[0] != second.tokens[1]);

  return lumi_test_pass("deterministic backend");
}
```

- [ ] **Step 2: Run test to verify it fails before implementation**

Run:

```bash
cmake --build build -j
```

Expected: build fails because `lumi/core/backend.hpp` does not exist.

- [ ] **Step 3: Add core request/result types**

Create `include/lumi/core/types.hpp` with:

```cpp
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lumi::core {

using RequestId = std::uint64_t;
using CacheHandle = std::uint64_t;
using TokenId = std::int32_t;

enum class RequestPriority : std::uint8_t {
  Low = 0,
  Normal = 1,
  High = 2,
};

enum class RequestStatus : std::uint8_t {
  Queued,
  Prefill,
  Decoding,
  Completed,
  Cancelled,
  Failed,
};

struct GenerationRequest {
  RequestId id = 0;
  std::string role;
  std::string prompt;
  RequestPriority priority = RequestPriority::Normal;
  std::uint32_t max_new_tokens = 128;
  std::optional<CacheHandle> cache_handle;
};

struct BackendStep {
  std::vector<TokenId> tokens;
  bool finished = false;
};

struct RequestResult {
  RequestId id = 0;
  RequestStatus status = RequestStatus::Queued;
  std::vector<TokenId> tokens;
  std::string text;
};

}  // namespace lumi::core
```

- [ ] **Step 4: Add metrics types**

Create `include/lumi/core/metrics.hpp` with:

```cpp
#pragma once

#include <cstdint>

namespace lumi::core {

struct SchedulerMetrics {
  std::uint64_t submitted_requests = 0;
  std::uint64_t completed_requests = 0;
  std::uint64_t cancelled_requests = 0;
  std::uint64_t scheduler_steps = 0;
  std::uint32_t queued_requests = 0;
  std::uint32_t active_requests = 0;
  std::uint32_t last_batch_size = 0;
  std::uint64_t emitted_tokens = 0;
};

struct CacheMetrics {
  std::uint32_t total_blocks = 0;
  std::uint32_t used_blocks = 0;
  std::uint32_t free_blocks = 0;
  std::uint32_t largest_free_run = 0;
  double fragmentation = 0.0;
  std::uint64_t allocations = 0;
  std::uint64_t frees = 0;
  std::uint64_t evictions = 0;
};

struct BenchmarkMetrics {
  std::uint32_t concurrency = 0;
  std::uint64_t total_tokens = 0;
  double elapsed_ms = 0.0;
  double tokens_per_second = 0.0;
  SchedulerMetrics scheduler;
  CacheMetrics cache;
};

}  // namespace lumi::core
```

- [ ] **Step 5: Add backend interface**

Create `include/lumi/core/backend.hpp` with:

```cpp
#pragma once

#include "lumi/core/types.hpp"

#include <cstdint>

namespace lumi::core {

class IModelBackend {
 public:
  virtual ~IModelBackend() = default;
  virtual BackendStep prefill(const GenerationRequest& request) = 0;
  virtual BackendStep decode(const GenerationRequest& request,
                             const std::vector<TokenId>& existing_tokens) = 0;
};

class DeterministicBackend final : public IModelBackend {
 public:
  explicit DeterministicBackend(std::int32_t vocab_size);

  BackendStep prefill(const GenerationRequest& request) override;
  BackendStep decode(const GenerationRequest& request,
                     const std::vector<TokenId>& existing_tokens) override;

 private:
  TokenId token_for(const GenerationRequest& request, std::uint32_t position) const;

  std::int32_t vocab_size_;
};

}  // namespace lumi::core
```

- [ ] **Step 6: Implement deterministic backend**

Create `src/core/mock_backend.cpp` with:

```cpp
#include "lumi/core/backend.hpp"

#include <algorithm>
#include <stdexcept>

namespace lumi::core {

DeterministicBackend::DeterministicBackend(std::int32_t vocab_size)
    : vocab_size_(vocab_size) {
  if (vocab_size_ <= 1) {
    throw std::invalid_argument("vocab_size must be greater than 1");
  }
}

BackendStep DeterministicBackend::prefill(const GenerationRequest& request) {
  BackendStep step;
  if (request.max_new_tokens == 0) {
    step.finished = true;
    return step;
  }
  step.tokens.push_back(token_for(request, 0));
  step.finished = request.max_new_tokens == 1;
  return step;
}

BackendStep DeterministicBackend::decode(
    const GenerationRequest& request,
    const std::vector<TokenId>& existing_tokens) {
  BackendStep step;
  step.tokens = existing_tokens;
  if (step.tokens.size() < request.max_new_tokens) {
    step.tokens.push_back(token_for(request, static_cast<std::uint32_t>(step.tokens.size())));
  }
  step.finished = step.tokens.size() >= request.max_new_tokens;
  return step;
}

TokenId DeterministicBackend::token_for(const GenerationRequest& request,
                                        std::uint32_t position) const {
  const auto seed = static_cast<std::uint64_t>(request.id * 9973u) +
                    static_cast<std::uint64_t>(position * 37u) +
                    static_cast<std::uint64_t>(request.prompt.size());
  return static_cast<TokenId>(seed % static_cast<std::uint64_t>(vocab_size_));
}

}  // namespace lumi::core
```

- [ ] **Step 7: Wire core library and backend test**

Modify `CMakeLists.txt` so the full file is:

```cmake
cmake_minimum_required(VERSION 3.25)

project(lumi_swarm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

add_library(lumi_core STATIC
  src/core/mock_backend.cpp
)
target_include_directories(lumi_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(lumi_build_smoke tests/test_build.cpp)
target_link_libraries(lumi_build_smoke PRIVATE lumi_core)
add_test(NAME lumi_build_smoke COMMAND lumi_build_smoke)

add_executable(lumi_mock_backend_test tests/test_mock_backend.cpp)
target_link_libraries(lumi_mock_backend_test PRIVATE lumi_core)
add_test(NAME lumi_mock_backend_test COMMAND lumi_mock_backend_test)
```

- [ ] **Step 8: Run tests**

Run:

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: smoke and mock backend tests pass.

- [ ] **Step 9: Commit**

```bash
git add CMakeLists.txt include/lumi/core src/core tests/test_mock_backend.cpp
git commit -m "feat: add deterministic backend"
```

---

### Task 3: Continuous Batching Scheduler

**Files:**
- Create: `include/lumi/core/scheduler.hpp`
- Create: `src/core/scheduler.cpp`
- Create: `tests/test_scheduler.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing scheduler tests**

Create `tests/test_scheduler.cpp` with:

```cpp
#include "lumi/core/backend.hpp"
#include "lumi/core/scheduler.hpp"
#include "test_main.hpp"

int main() {
  lumi::core::DeterministicBackend backend(256);
  lumi::core::Scheduler scheduler(backend, {.max_batch_size = 2});

  scheduler.submit({.id = 1, .role = "planner", .prompt = "plan", .max_new_tokens = 2});
  scheduler.submit({.id = 2, .role = "reviewer", .prompt = "review", .max_new_tokens = 2});
  scheduler.submit({.id = 3, .role = "implementer", .prompt = "patch", .max_new_tokens = 1});

  scheduler.step();
  auto metrics = scheduler.metrics();
  LUMI_CHECK(metrics.submitted_requests == 3);
  LUMI_CHECK(metrics.active_requests == 2);
  LUMI_CHECK(metrics.queued_requests == 1);
  LUMI_CHECK(metrics.last_batch_size == 2);

  scheduler.step();
  scheduler.step();

  LUMI_CHECK(scheduler.is_idle());
  LUMI_CHECK(scheduler.result(1).status == lumi::core::RequestStatus::Completed);
  LUMI_CHECK(scheduler.result(2).tokens.size() == 2);
  LUMI_CHECK(scheduler.result(3).tokens.size() == 1);
  LUMI_CHECK(scheduler.metrics().completed_requests == 3);

  scheduler.submit({.id = 4, .role = "planner", .prompt = "cancel", .max_new_tokens = 3});
  LUMI_CHECK(scheduler.cancel(4));
  LUMI_CHECK(scheduler.result(4).status == lumi::core::RequestStatus::Cancelled);

  return lumi_test_pass("scheduler batching");
}
```

- [ ] **Step 2: Run test to verify it fails before implementation**

Run:

```bash
cmake --build build -j
```

Expected: build fails because `lumi/core/scheduler.hpp` does not exist.

- [ ] **Step 3: Add scheduler API**

Create `include/lumi/core/scheduler.hpp` with:

```cpp
#pragma once

#include "lumi/core/backend.hpp"
#include "lumi/core/metrics.hpp"
#include "lumi/core/types.hpp"

#include <cstddef>
#include <deque>
#include <unordered_map>
#include <vector>

namespace lumi::core {

struct SchedulerConfig {
  std::size_t max_batch_size = 8;
};

class Scheduler {
 public:
  Scheduler(IModelBackend& backend, SchedulerConfig config);

  void submit(GenerationRequest request);
  bool cancel(RequestId id);
  void step();
  bool is_idle() const;

  RequestResult result(RequestId id) const;
  SchedulerMetrics metrics() const;

 private:
  struct ActiveRequest {
    GenerationRequest request;
    RequestStatus status = RequestStatus::Queued;
    std::vector<TokenId> tokens;
  };

  static bool has_higher_priority(const GenerationRequest& lhs,
                                  const GenerationRequest& rhs);

  void activate_requests();
  void complete_request(const ActiveRequest& active);
  ActiveRequest* find_active(RequestId id);

  IModelBackend& backend_;
  SchedulerConfig config_;
  std::deque<GenerationRequest> queued_;
  std::vector<ActiveRequest> active_;
  std::unordered_map<RequestId, RequestResult> results_;
  SchedulerMetrics metrics_;
};

}  // namespace lumi::core
```

- [ ] **Step 4: Implement scheduler**

Create `src/core/scheduler.cpp` with:

```cpp
#include "lumi/core/scheduler.hpp"

#include <algorithm>
#include <stdexcept>

namespace lumi::core {

Scheduler::Scheduler(IModelBackend& backend, SchedulerConfig config)
    : backend_(backend), config_(config) {
  if (config_.max_batch_size == 0) {
    throw std::invalid_argument("max_batch_size must be greater than zero");
  }
}

void Scheduler::submit(GenerationRequest request) {
  if (request.id == 0) {
    throw std::invalid_argument("request id must be non-zero");
  }
  if (results_.contains(request.id) || find_active(request.id) != nullptr) {
    throw std::invalid_argument("duplicate request id");
  }
  queued_.push_back(std::move(request));
  std::stable_sort(queued_.begin(), queued_.end(), has_higher_priority);
  metrics_.submitted_requests += 1;
  metrics_.queued_requests = static_cast<std::uint32_t>(queued_.size());
}

bool Scheduler::cancel(RequestId id) {
  auto queued_it = std::find_if(queued_.begin(), queued_.end(),
                               [id](const GenerationRequest& request) {
                                 return request.id == id;
                               });
  if (queued_it != queued_.end()) {
    results_[id] = {.id = id, .status = RequestStatus::Cancelled};
    queued_.erase(queued_it);
    metrics_.cancelled_requests += 1;
    metrics_.queued_requests = static_cast<std::uint32_t>(queued_.size());
    return true;
  }

  auto active_it = std::find_if(active_.begin(), active_.end(),
                               [id](const ActiveRequest& active) {
                                 return active.request.id == id;
                               });
  if (active_it != active_.end()) {
    results_[id] = {.id = id, .status = RequestStatus::Cancelled,
                    .tokens = active_it->tokens};
    active_.erase(active_it);
    metrics_.cancelled_requests += 1;
    metrics_.active_requests = static_cast<std::uint32_t>(active_.size());
    return true;
  }

  return false;
}

void Scheduler::step() {
  activate_requests();
  metrics_.scheduler_steps += 1;
  metrics_.last_batch_size = static_cast<std::uint32_t>(active_.size());

  std::vector<RequestId> completed;
  for (auto& active : active_) {
    BackendStep backend_step =
        active.tokens.empty() ? backend_.prefill(active.request)
                              : backend_.decode(active.request, active.tokens);
    const auto before = active.tokens.size();
    active.tokens = std::move(backend_step.tokens);
    metrics_.emitted_tokens += active.tokens.size() - before;
    active.status = backend_step.finished ? RequestStatus::Completed
                                          : RequestStatus::Decoding;
    if (backend_step.finished) {
      completed.push_back(active.request.id);
    }
  }

  for (const auto id : completed) {
    auto it = std::find_if(active_.begin(), active_.end(),
                           [id](const ActiveRequest& active) {
                             return active.request.id == id;
                           });
    if (it != active_.end()) {
      complete_request(*it);
      active_.erase(it);
    }
  }

  activate_requests();
  metrics_.queued_requests = static_cast<std::uint32_t>(queued_.size());
  metrics_.active_requests = static_cast<std::uint32_t>(active_.size());
}

bool Scheduler::is_idle() const {
  return queued_.empty() && active_.empty();
}

RequestResult Scheduler::result(RequestId id) const {
  const auto it = results_.find(id);
  if (it == results_.end()) {
    return {.id = id, .status = RequestStatus::Queued};
  }
  return it->second;
}

SchedulerMetrics Scheduler::metrics() const {
  return metrics_;
}

bool Scheduler::has_higher_priority(const GenerationRequest& lhs,
                                    const GenerationRequest& rhs) {
  return static_cast<int>(lhs.priority) > static_cast<int>(rhs.priority);
}

void Scheduler::activate_requests() {
  while (!queued_.empty() && active_.size() < config_.max_batch_size) {
    auto request = std::move(queued_.front());
    queued_.pop_front();
    active_.push_back({.request = std::move(request),
                       .status = RequestStatus::Prefill});
  }
  metrics_.queued_requests = static_cast<std::uint32_t>(queued_.size());
  metrics_.active_requests = static_cast<std::uint32_t>(active_.size());
}

void Scheduler::complete_request(const ActiveRequest& active) {
  results_[active.request.id] = {.id = active.request.id,
                                 .status = RequestStatus::Completed,
                                 .tokens = active.tokens};
  metrics_.completed_requests += 1;
}

Scheduler::ActiveRequest* Scheduler::find_active(RequestId id) {
  auto it = std::find_if(active_.begin(), active_.end(),
                         [id](const ActiveRequest& active) {
                           return active.request.id == id;
                         });
  return it == active_.end() ? nullptr : &(*it);
}

}  // namespace lumi::core
```

- [ ] **Step 5: Wire scheduler test**

Modify `CMakeLists.txt`:

```cmake
add_library(lumi_core STATIC
  src/core/mock_backend.cpp
  src/core/scheduler.cpp
)

add_executable(lumi_scheduler_test tests/test_scheduler.cpp)
target_link_libraries(lumi_scheduler_test PRIVATE lumi_core)
add_test(NAME lumi_scheduler_test COMMAND lumi_scheduler_test)
```

Keep the existing targets unchanged.

- [ ] **Step 6: Run tests**

Run:

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: scheduler test passes and all earlier tests continue passing.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt include/lumi/core/scheduler.hpp src/core/scheduler.cpp tests/test_scheduler.cpp
git commit -m "feat: add continuous batching scheduler"
```

---

### Task 4: Paged KV-Cache Prototype

**Files:**
- Create: `include/lumi/core/kv_cache.hpp`
- Create: `src/core/kv_cache.cpp`
- Create: `tests/test_kv_cache.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing KV-cache tests**

Create `tests/test_kv_cache.cpp` with:

```cpp
#include "lumi/core/kv_cache.hpp"
#include "test_main.hpp"

int main() {
  lumi::core::PagedKvCache cache({.total_blocks = 4, .tokens_per_block = 16});

  auto first = cache.allocate(10);
  LUMI_CHECK(first.has_value());
  LUMI_CHECK(first->blocks.size() == 1);

  auto second = cache.allocate(40);
  LUMI_CHECK(second.has_value());
  LUMI_CHECK(second->blocks.size() == 3);
  LUMI_CHECK(cache.metrics().used_blocks == 4);

  auto third = cache.allocate(1);
  LUMI_CHECK(!third.has_value());
  LUMI_CHECK(cache.metrics().evictions == 0);

  cache.free(first->handle);
  LUMI_CHECK(cache.metrics().free_blocks == 1);
  LUMI_CHECK(cache.contains(second->handle));
  LUMI_CHECK(!cache.contains(first->handle));

  cache.free(second->handle);
  LUMI_CHECK(cache.metrics().used_blocks == 0);
  LUMI_CHECK(cache.metrics().free_blocks == 4);

  return lumi_test_pass("paged kv cache");
}
```

- [ ] **Step 2: Run test to verify it fails before implementation**

Run:

```bash
cmake --build build -j
```

Expected: build fails because `lumi/core/kv_cache.hpp` does not exist.

- [ ] **Step 3: Add KV-cache API**

Create `include/lumi/core/kv_cache.hpp` with:

```cpp
#pragma once

#include "lumi/core/metrics.hpp"
#include "lumi/core/types.hpp"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lumi::core {

struct KvCacheConfig {
  std::uint32_t total_blocks = 0;
  std::uint32_t tokens_per_block = 0;
};

struct KvCacheAllocation {
  CacheHandle handle = 0;
  std::vector<std::uint32_t> blocks;
};

class PagedKvCache {
 public:
  explicit PagedKvCache(KvCacheConfig config);

  std::optional<KvCacheAllocation> allocate(std::uint32_t tokens);
  void free(CacheHandle handle);
  bool contains(CacheHandle handle) const;
  CacheMetrics metrics() const;

 private:
  std::uint32_t blocks_required(std::uint32_t tokens) const;
  void refresh_metrics();

  KvCacheConfig config_;
  CacheHandle next_handle_ = 1;
  std::vector<bool> used_;
  std::unordered_map<CacheHandle, std::vector<std::uint32_t>> allocations_;
  CacheMetrics metrics_;
};

}  // namespace lumi::core
```

- [ ] **Step 4: Implement KV-cache allocator**

Create `src/core/kv_cache.cpp` with:

```cpp
#include "lumi/core/kv_cache.hpp"

#include <algorithm>
#include <stdexcept>

namespace lumi::core {

PagedKvCache::PagedKvCache(KvCacheConfig config)
    : config_(config), used_(config.total_blocks, false) {
  if (config_.total_blocks == 0 || config_.tokens_per_block == 0) {
    throw std::invalid_argument("kv cache requires non-zero blocks and block size");
  }
  refresh_metrics();
}

std::optional<KvCacheAllocation> PagedKvCache::allocate(std::uint32_t tokens) {
  const auto needed = blocks_required(tokens);
  std::vector<std::uint32_t> blocks;
  blocks.reserve(needed);

  for (std::uint32_t i = 0; i < used_.size() && blocks.size() < needed; ++i) {
    if (!used_[i]) {
      blocks.push_back(i);
    }
  }

  if (blocks.size() != needed) {
    refresh_metrics();
    return std::nullopt;
  }

  for (const auto block : blocks) {
    used_[block] = true;
  }

  const auto handle = next_handle_++;
  allocations_[handle] = blocks;
  metrics_.allocations += 1;
  refresh_metrics();
  return KvCacheAllocation{.handle = handle, .blocks = std::move(blocks)};
}

void PagedKvCache::free(CacheHandle handle) {
  auto it = allocations_.find(handle);
  if (it == allocations_.end()) {
    return;
  }
  for (const auto block : it->second) {
    used_[block] = false;
  }
  allocations_.erase(it);
  metrics_.frees += 1;
  refresh_metrics();
}

bool PagedKvCache::contains(CacheHandle handle) const {
  return allocations_.contains(handle);
}

CacheMetrics PagedKvCache::metrics() const {
  return metrics_;
}

std::uint32_t PagedKvCache::blocks_required(std::uint32_t tokens) const {
  const auto safe_tokens = std::max<std::uint32_t>(tokens, 1);
  return (safe_tokens + config_.tokens_per_block - 1) / config_.tokens_per_block;
}

void PagedKvCache::refresh_metrics() {
  metrics_.total_blocks = config_.total_blocks;
  metrics_.used_blocks = static_cast<std::uint32_t>(
      std::count(used_.begin(), used_.end(), true));
  metrics_.free_blocks = metrics_.total_blocks - metrics_.used_blocks;

  std::uint32_t current_run = 0;
  std::uint32_t largest_run = 0;
  for (const bool used : used_) {
    if (!used) {
      current_run += 1;
      largest_run = std::max(largest_run, current_run);
    } else {
      current_run = 0;
    }
  }
  metrics_.largest_free_run = largest_run;
  metrics_.fragmentation =
      metrics_.free_blocks == 0
          ? 0.0
          : 1.0 - (static_cast<double>(largest_run) /
                   static_cast<double>(metrics_.free_blocks));
}

}  // namespace lumi::core
```

- [ ] **Step 5: Wire KV-cache test**

Modify `CMakeLists.txt`:

```cmake
add_library(lumi_core STATIC
  src/core/mock_backend.cpp
  src/core/scheduler.cpp
  src/core/kv_cache.cpp
)

add_executable(lumi_kv_cache_test tests/test_kv_cache.cpp)
target_link_libraries(lumi_kv_cache_test PRIVATE lumi_core)
add_test(NAME lumi_kv_cache_test COMMAND lumi_kv_cache_test)
```

Keep existing targets unchanged.

- [ ] **Step 6: Run tests**

Run:

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: KV-cache test passes and all earlier tests continue passing.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt include/lumi/core/kv_cache.hpp src/core/kv_cache.cpp tests/test_kv_cache.cpp
git commit -m "feat: add paged kv cache accounting"
```

---

### Task 5: Scheduler And KV-Cache Integration

**Files:**
- Modify: `include/lumi/core/scheduler.hpp`
- Modify: `src/core/scheduler.cpp`
- Modify: `tests/test_scheduler.cpp`

- [ ] **Step 1: Extend scheduler test with cache assertions**

Append this block to `tests/test_scheduler.cpp` before the final `return`:

```cpp
  lumi::core::DeterministicBackend cached_backend(256);
  lumi::core::PagedKvCache cache({.total_blocks = 8, .tokens_per_block = 16});
  lumi::core::Scheduler cached_scheduler(
      cached_backend, {.max_batch_size = 4}, &cache);

  cached_scheduler.submit({.id = 10,
                           .role = "reviewer",
                           .prompt = "long review context",
                           .max_new_tokens = 4});
  cached_scheduler.step();
  LUMI_CHECK(cache.metrics().used_blocks > 0);
  while (!cached_scheduler.is_idle()) {
    cached_scheduler.step();
  }
  LUMI_CHECK(cached_scheduler.result(10).status ==
             lumi::core::RequestStatus::Completed);
  LUMI_CHECK(cache.metrics().used_blocks == 0);
```

- [ ] **Step 2: Run test to verify it fails before integration**

Run:

```bash
cmake --build build -j
```

Expected: build fails because `Scheduler` has no constructor accepting `PagedKvCache*`.

- [ ] **Step 3: Update scheduler header**

Modify `include/lumi/core/scheduler.hpp`:

```cpp
#include "lumi/core/kv_cache.hpp"
```

Change the constructor declaration to:

```cpp
Scheduler(IModelBackend& backend, SchedulerConfig config, PagedKvCache* cache = nullptr);
```

Add fields to `ActiveRequest`:

```cpp
std::optional<CacheHandle> owned_cache;
```

Add a private field:

```cpp
PagedKvCache* cache_ = nullptr;
```

- [ ] **Step 4: Update scheduler implementation**

Modify the constructor in `src/core/scheduler.cpp`:

```cpp
Scheduler::Scheduler(IModelBackend& backend, SchedulerConfig config, PagedKvCache* cache)
    : backend_(backend), config_(config), cache_(cache) {
  if (config_.max_batch_size == 0) {
    throw std::invalid_argument("max_batch_size must be greater than zero");
  }
}
```

Inside `activate_requests()`, replace the `active_.push_back(...)` call with:

```cpp
    ActiveRequest active{.request = std::move(request),
                         .status = RequestStatus::Prefill};
    if (cache_ != nullptr) {
      const auto prompt_tokens =
          static_cast<std::uint32_t>((active.request.prompt.size() / 4) +
                                     active.request.max_new_tokens + 1);
      auto allocation = cache_->allocate(prompt_tokens);
      if (allocation.has_value()) {
        active.owned_cache = allocation->handle;
        active.request.cache_handle = allocation->handle;
      } else {
        results_[active.request.id] = {.id = active.request.id,
                                       .status = RequestStatus::Failed,
                                       .text = "kv cache allocation failed"};
        continue;
      }
    }
    active_.push_back(std::move(active));
```

At the top of `complete_request(const ActiveRequest& active)`, add:

```cpp
  if (cache_ != nullptr && active.owned_cache.has_value()) {
    cache_->free(*active.owned_cache);
  }
```

In the active cancellation branch before `active_.erase(active_it);`, add:

```cpp
    if (cache_ != nullptr && active_it->owned_cache.has_value()) {
      cache_->free(*active_it->owned_cache);
    }
```

- [ ] **Step 5: Run tests**

Run:

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: scheduler cache integration passes and all earlier tests continue passing.

- [ ] **Step 6: Commit**

```bash
git add include/lumi/core/scheduler.hpp src/core/scheduler.cpp tests/test_scheduler.cpp
git commit -m "feat: connect scheduler to kv cache"
```

---

### Task 6: CPU And CUDA Sampling

**Files:**
- Create: `include/lumi/core/sampling.hpp`
- Create: `src/core/sampling.cpp`
- Create: `src/cuda/sampling_kernels.cu`
- Create: `tests/test_sampling.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing sampling test**

Create `tests/test_sampling.cpp` with:

```cpp
#include "lumi/core/sampling.hpp"
#include "test_main.hpp"

#include <vector>

int main() {
  const std::vector<float> logits = {
      0.1F, 2.0F, 0.3F, 1.0F,
      4.0F, 0.2F, 3.0F, 0.1F,
  };

  lumi::core::SamplingConfig config;
  config.batch_size = 2;
  config.vocab_size = 4;
  config.temperature = 1.0F;
  config.top_k = 2;
  config.uniforms = {0.0F, 0.99F};

  const auto cpu = lumi::core::sample_top_k_cpu(logits, config);
  LUMI_CHECK(cpu.size() == 2);
  LUMI_CHECK(cpu[0] == 1);
  LUMI_CHECK(cpu[1] == 2);

  const auto gpu = lumi::core::sample_top_k_cuda(logits, config);
  LUMI_CHECK(gpu == cpu);

  return lumi_test_pass("sampling");
}
```

- [ ] **Step 2: Run test to verify it fails before implementation**

Run:

```bash
cmake --build build -j
```

Expected: build fails because `lumi/core/sampling.hpp` does not exist.

- [ ] **Step 3: Add sampling API**

Create `include/lumi/core/sampling.hpp` with:

```cpp
#pragma once

#include "lumi/core/types.hpp"

#include <cstdint>
#include <vector>

namespace lumi::core {

struct SamplingConfig {
  std::int32_t batch_size = 0;
  std::int32_t vocab_size = 0;
  float temperature = 1.0F;
  std::int32_t top_k = 1;
  std::vector<float> uniforms;
};

std::vector<TokenId> sample_top_k_cpu(const std::vector<float>& logits,
                                      const SamplingConfig& config);

std::vector<TokenId> sample_top_k_cuda(const std::vector<float>& logits,
                                       const SamplingConfig& config);

}  // namespace lumi::core
```

- [ ] **Step 4: Implement CPU reference and CUDA wrapper**

Create `src/core/sampling.cpp` with:

```cpp
#include "lumi/core/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace lumi::core {
namespace {

void validate_sampling_inputs(const std::vector<float>& logits,
                              const SamplingConfig& config) {
  if (config.batch_size <= 0 || config.vocab_size <= 0) {
    throw std::invalid_argument("batch_size and vocab_size must be positive");
  }
  if (config.temperature <= 0.0F) {
    throw std::invalid_argument("temperature must be positive");
  }
  if (config.top_k <= 0 || config.top_k > config.vocab_size) {
    throw std::invalid_argument("top_k must be in [1, vocab_size]");
  }
  if (logits.size() !=
      static_cast<std::size_t>(config.batch_size * config.vocab_size)) {
    throw std::invalid_argument("logits size does not match batch_size * vocab_size");
  }
  if (config.uniforms.size() != static_cast<std::size_t>(config.batch_size)) {
    throw std::invalid_argument("uniforms size must match batch_size");
  }
}

}  // namespace

std::vector<TokenId> sample_top_k_cpu(const std::vector<float>& logits,
                                      const SamplingConfig& config) {
  validate_sampling_inputs(logits, config);
  std::vector<TokenId> output(static_cast<std::size_t>(config.batch_size));

  for (std::int32_t row = 0; row < config.batch_size; ++row) {
    std::vector<std::pair<float, TokenId>> scored;
    scored.reserve(static_cast<std::size_t>(config.vocab_size));
    for (std::int32_t col = 0; col < config.vocab_size; ++col) {
      const auto value = logits[static_cast<std::size_t>(row * config.vocab_size + col)] /
                         config.temperature;
      scored.push_back({value, col});
    }
    std::partial_sort(scored.begin(), scored.begin() + config.top_k, scored.end(),
                      [](const auto& lhs, const auto& rhs) {
                        return lhs.first > rhs.first;
                      });
    scored.resize(static_cast<std::size_t>(config.top_k));

    const auto uniform = std::clamp(config.uniforms[static_cast<std::size_t>(row)],
                                    0.0F, 0.999999F);
    const auto index = std::min<std::int32_t>(
        static_cast<std::int32_t>(uniform * static_cast<float>(config.top_k)),
        config.top_k - 1);
    output[static_cast<std::size_t>(row)] = scored[static_cast<std::size_t>(index)].second;
  }

  return output;
}

std::vector<TokenId> sample_top_k_cuda(const std::vector<float>& logits,
                                       const SamplingConfig& config);

}  // namespace lumi::core
```

- [ ] **Step 5: Implement CUDA sampling kernel**

Create `src/cuda/sampling_kernels.cu` with:

```cuda
#include "lumi/core/sampling.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace lumi::core {
namespace {

constexpr int kMaxTopK = 64;

__global__ void sample_top_k_kernel(const float* logits,
                                    const float* uniforms,
                                    int batch_size,
                                    int vocab_size,
                                    float temperature,
                                    int top_k,
                                    int* output) {
  const int row = blockIdx.x;
  if (row >= batch_size || threadIdx.x != 0) {
    return;
  }

  float best_values[kMaxTopK];
  int best_tokens[kMaxTopK];
  for (int i = 0; i < top_k; ++i) {
    best_values[i] = -3.402823466e+38F;
    best_tokens[i] = 0;
  }

  for (int col = 0; col < vocab_size; ++col) {
    const float value = logits[row * vocab_size + col] / temperature;
    for (int slot = 0; slot < top_k; ++slot) {
      if (value > best_values[slot]) {
        for (int shift = top_k - 1; shift > slot; --shift) {
          best_values[shift] = best_values[shift - 1];
          best_tokens[shift] = best_tokens[shift - 1];
        }
        best_values[slot] = value;
        best_tokens[slot] = col;
        break;
      }
    }
  }

  float uniform = uniforms[row];
  uniform = uniform < 0.0F ? 0.0F : uniform;
  uniform = uniform >= 1.0F ? 0.999999F : uniform;
  int selected = static_cast<int>(uniform * static_cast<float>(top_k));
  selected = selected >= top_k ? top_k - 1 : selected;
  output[row] = best_tokens[selected];
}

void check_cuda(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " +
                             cudaGetErrorString(status));
  }
}

}  // namespace

std::vector<TokenId> sample_top_k_cuda(const std::vector<float>& logits,
                                       const SamplingConfig& config) {
  if (config.top_k > kMaxTopK) {
    throw std::invalid_argument("CUDA sampling top_k must be <= 64");
  }
  if (config.batch_size <= 0 || config.vocab_size <= 0) {
    throw std::invalid_argument("batch_size and vocab_size must be positive");
  }
  if (logits.size() !=
      static_cast<std::size_t>(config.batch_size * config.vocab_size)) {
    throw std::invalid_argument("logits size does not match batch_size * vocab_size");
  }
  if (config.uniforms.size() != static_cast<std::size_t>(config.batch_size)) {
    throw std::invalid_argument("uniforms size must match batch_size");
  }

  float* device_logits = nullptr;
  float* device_uniforms = nullptr;
  int* device_output = nullptr;
  const auto logits_bytes = logits.size() * sizeof(float);
  const auto uniforms_bytes = config.uniforms.size() * sizeof(float);
  const auto output_bytes = static_cast<std::size_t>(config.batch_size) * sizeof(int);

  check_cuda(cudaMalloc(&device_logits, logits_bytes), "cudaMalloc logits");
  check_cuda(cudaMalloc(&device_uniforms, uniforms_bytes), "cudaMalloc uniforms");
  check_cuda(cudaMalloc(&device_output, output_bytes), "cudaMalloc output");

  check_cuda(cudaMemcpy(device_logits, logits.data(), logits_bytes,
                        cudaMemcpyHostToDevice),
             "copy logits");
  check_cuda(cudaMemcpy(device_uniforms, config.uniforms.data(), uniforms_bytes,
                        cudaMemcpyHostToDevice),
             "copy uniforms");

  sample_top_k_kernel<<<config.batch_size, 1>>>(device_logits, device_uniforms,
                                                config.batch_size, config.vocab_size,
                                                config.temperature, config.top_k,
                                                device_output);
  check_cuda(cudaGetLastError(), "sample_top_k_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "sample_top_k_kernel sync");

  std::vector<int> raw_output(static_cast<std::size_t>(config.batch_size));
  check_cuda(cudaMemcpy(raw_output.data(), device_output, output_bytes,
                        cudaMemcpyDeviceToHost),
             "copy output");

  cudaFree(device_output);
  cudaFree(device_uniforms);
  cudaFree(device_logits);

  std::vector<TokenId> output(raw_output.begin(), raw_output.end());
  return output;
}

}  // namespace lumi::core
```

- [ ] **Step 6: Wire CUDA sampling into build**

Modify `CMakeLists.txt`:

```cmake
add_library(lumi_core STATIC
  src/core/mock_backend.cpp
  src/core/scheduler.cpp
  src/core/kv_cache.cpp
  src/core/sampling.cpp
  src/cuda/sampling_kernels.cu
)
set_target_properties(lumi_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(lumi_sampling_test tests/test_sampling.cpp)
target_link_libraries(lumi_sampling_test PRIVATE lumi_core)
add_test(NAME lumi_sampling_test COMMAND lumi_sampling_test)
```

Keep existing targets unchanged.

- [ ] **Step 7: Run tests**

Run:

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: sampling test passes on CPU and CUDA paths.

- [ ] **Step 8: Commit**

```bash
git add CMakeLists.txt include/lumi/core/sampling.hpp src/core/sampling.cpp src/cuda/sampling_kernels.cu tests/test_sampling.cpp
git commit -m "feat: add CUDA top-k sampling"
```

---

### Task 7: Retrieval Scoring CPU And CUDA

**Files:**
- Create: `include/lumi/core/retrieval.hpp`
- Create: `src/core/retrieval.cpp`
- Create: `src/cuda/retrieval_kernels.cu`
- Create: `tests/test_retrieval.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing retrieval test**

Create `tests/test_retrieval.cpp` with:

```cpp
#include "lumi/core/retrieval.hpp"
#include "test_main.hpp"

#include <vector>

int main() {
  lumi::core::RetrievalConfig config;
  config.query_count = 2;
  config.document_count = 3;
  config.dimensions = 2;

  const std::vector<float> queries = {
      1.0F, 0.0F,
      0.0F, 1.0F,
  };
  const std::vector<float> documents = {
      1.0F, 0.0F,
      0.0F, 1.0F,
      0.5F, 0.5F,
  };

  const auto cpu = lumi::core::score_dot_cpu(queries, documents, config);
  const auto gpu = lumi::core::score_dot_cuda(queries, documents, config);

  LUMI_CHECK(cpu.size() == 6);
  LUMI_CHECK(cpu == gpu);
  LUMI_CHECK(cpu[0] == 1.0F);
  LUMI_CHECK(cpu[1] == 0.0F);
  LUMI_CHECK(cpu[5] == 0.5F);

  return lumi_test_pass("retrieval");
}
```

- [ ] **Step 2: Run test to verify it fails before implementation**

Run:

```bash
cmake --build build -j
```

Expected: build fails because `lumi/core/retrieval.hpp` does not exist.

- [ ] **Step 3: Add retrieval API**

Create `include/lumi/core/retrieval.hpp` with:

```cpp
#pragma once

#include <cstdint>
#include <vector>

namespace lumi::core {

struct RetrievalConfig {
  std::int32_t query_count = 0;
  std::int32_t document_count = 0;
  std::int32_t dimensions = 0;
};

std::vector<float> score_dot_cpu(const std::vector<float>& queries,
                                 const std::vector<float>& documents,
                                 const RetrievalConfig& config);

std::vector<float> score_dot_cuda(const std::vector<float>& queries,
                                  const std::vector<float>& documents,
                                  const RetrievalConfig& config);

}  // namespace lumi::core
```

- [ ] **Step 4: Implement CPU retrieval reference**

Create `src/core/retrieval.cpp` with:

```cpp
#include "lumi/core/retrieval.hpp"

#include <stdexcept>

namespace lumi::core {
namespace {

void validate_retrieval_inputs(const std::vector<float>& queries,
                               const std::vector<float>& documents,
                               const RetrievalConfig& config) {
  if (config.query_count <= 0 || config.document_count <= 0 ||
      config.dimensions <= 0) {
    throw std::invalid_argument("retrieval dimensions must be positive");
  }
  if (queries.size() !=
      static_cast<std::size_t>(config.query_count * config.dimensions)) {
    throw std::invalid_argument("query size mismatch");
  }
  if (documents.size() !=
      static_cast<std::size_t>(config.document_count * config.dimensions)) {
    throw std::invalid_argument("document size mismatch");
  }
}

}  // namespace

std::vector<float> score_dot_cpu(const std::vector<float>& queries,
                                 const std::vector<float>& documents,
                                 const RetrievalConfig& config) {
  validate_retrieval_inputs(queries, documents, config);
  std::vector<float> scores(
      static_cast<std::size_t>(config.query_count * config.document_count), 0.0F);

  for (std::int32_t query = 0; query < config.query_count; ++query) {
    for (std::int32_t doc = 0; doc < config.document_count; ++doc) {
      float score = 0.0F;
      for (std::int32_t dim = 0; dim < config.dimensions; ++dim) {
        score += queries[static_cast<std::size_t>(query * config.dimensions + dim)] *
                 documents[static_cast<std::size_t>(doc * config.dimensions + dim)];
      }
      scores[static_cast<std::size_t>(query * config.document_count + doc)] = score;
    }
  }

  return scores;
}

std::vector<float> score_dot_cuda(const std::vector<float>& queries,
                                  const std::vector<float>& documents,
                                  const RetrievalConfig& config);

}  // namespace lumi::core
```

- [ ] **Step 5: Implement CUDA retrieval kernel**

Create `src/cuda/retrieval_kernels.cu` with:

```cuda
#include "lumi/core/retrieval.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace lumi::core {
namespace {

__global__ void dot_kernel(const float* queries,
                           const float* documents,
                           int query_count,
                           int document_count,
                           int dimensions,
                           float* scores) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = query_count * document_count;
  if (index >= total) {
    return;
  }

  const int query = index / document_count;
  const int document = index % document_count;
  float score = 0.0F;
  for (int dim = 0; dim < dimensions; ++dim) {
    score += queries[query * dimensions + dim] *
             documents[document * dimensions + dim];
  }
  scores[index] = score;
}

void check_cuda(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " +
                             cudaGetErrorString(status));
  }
}

}  // namespace

std::vector<float> score_dot_cuda(const std::vector<float>& queries,
                                  const std::vector<float>& documents,
                                  const RetrievalConfig& config) {
  if (config.query_count <= 0 || config.document_count <= 0 ||
      config.dimensions <= 0) {
    throw std::invalid_argument("retrieval dimensions must be positive");
  }

  float* device_queries = nullptr;
  float* device_documents = nullptr;
  float* device_scores = nullptr;
  const auto query_bytes = queries.size() * sizeof(float);
  const auto document_bytes = documents.size() * sizeof(float);
  const auto score_count =
      static_cast<std::size_t>(config.query_count * config.document_count);
  const auto score_bytes = score_count * sizeof(float);

  check_cuda(cudaMalloc(&device_queries, query_bytes), "cudaMalloc queries");
  check_cuda(cudaMalloc(&device_documents, document_bytes), "cudaMalloc documents");
  check_cuda(cudaMalloc(&device_scores, score_bytes), "cudaMalloc scores");

  check_cuda(cudaMemcpy(device_queries, queries.data(), query_bytes,
                        cudaMemcpyHostToDevice),
             "copy queries");
  check_cuda(cudaMemcpy(device_documents, documents.data(), document_bytes,
                        cudaMemcpyHostToDevice),
             "copy documents");

  const int threads = 128;
  const int blocks = static_cast<int>((score_count + threads - 1) / threads);
  dot_kernel<<<blocks, threads>>>(device_queries, device_documents,
                                  config.query_count, config.document_count,
                                  config.dimensions, device_scores);
  check_cuda(cudaGetLastError(), "dot_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "dot_kernel sync");

  std::vector<float> scores(score_count);
  check_cuda(cudaMemcpy(scores.data(), device_scores, score_bytes,
                        cudaMemcpyDeviceToHost),
             "copy scores");

  cudaFree(device_scores);
  cudaFree(device_documents);
  cudaFree(device_queries);

  return scores;
}

}  // namespace lumi::core
```

- [ ] **Step 6: Wire retrieval into build**

Modify `CMakeLists.txt`:

```cmake
add_library(lumi_core STATIC
  src/core/mock_backend.cpp
  src/core/scheduler.cpp
  src/core/kv_cache.cpp
  src/core/sampling.cpp
  src/core/retrieval.cpp
  src/cuda/sampling_kernels.cu
  src/cuda/retrieval_kernels.cu
)

add_executable(lumi_retrieval_test tests/test_retrieval.cpp)
target_link_libraries(lumi_retrieval_test PRIVATE lumi_core)
add_test(NAME lumi_retrieval_test COMMAND lumi_retrieval_test)
```

Keep existing targets unchanged.

- [ ] **Step 7: Run tests**

Run:

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: retrieval test passes on CPU and CUDA paths.

- [ ] **Step 8: Commit**

```bash
git add CMakeLists.txt include/lumi/core/retrieval.hpp src/core/retrieval.cpp src/cuda/retrieval_kernels.cu tests/test_retrieval.cpp
git commit -m "feat: add CUDA retrieval scoring"
```

---

### Task 8: Thin Agent Runtime

**Files:**
- Create: `include/lumi/agent/agent.hpp`
- Create: `src/agent/agent.cpp`
- Create: `tests/test_agent_demo.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing agent-runtime test**

Create `tests/test_agent_demo.cpp` with:

```cpp
#include "lumi/agent/agent.hpp"
#include "lumi/core/backend.hpp"
#include "lumi/core/kv_cache.hpp"
#include "lumi/core/scheduler.hpp"
#include "test_main.hpp"

int main() {
  lumi::core::DeterministicBackend backend(512);
  lumi::core::PagedKvCache cache({.total_blocks = 32, .tokens_per_block = 16});
  lumi::core::Scheduler scheduler(backend, {.max_batch_size = 4}, &cache);

  lumi::agent::SwarmOrchestrator orchestrator(scheduler);
  const auto trace = orchestrator.run_scripted_coding_task(
      {.task = "fix add function", .verification_command = "python3 -m unittest"});

  LUMI_CHECK(trace.steps.size() == 4);
  LUMI_CHECK(trace.steps[0].role == lumi::agent::AgentRole::Planner);
  LUMI_CHECK(trace.steps[1].role == lumi::agent::AgentRole::Implementer);
  LUMI_CHECK(trace.steps[2].role == lumi::agent::AgentRole::Reviewer);
  LUMI_CHECK(trace.steps[3].role == lumi::agent::AgentRole::Verifier);
  LUMI_CHECK(scheduler.metrics().completed_requests == 4);

  return lumi_test_pass("agent runtime");
}
```

- [ ] **Step 2: Run test to verify it fails before implementation**

Run:

```bash
cmake --build build -j
```

Expected: build fails because `lumi/agent/agent.hpp` does not exist.

- [ ] **Step 3: Add agent API**

Create `include/lumi/agent/agent.hpp` with:

```cpp
#pragma once

#include "lumi/core/scheduler.hpp"

#include <string>
#include <vector>

namespace lumi::agent {

enum class AgentRole {
  Planner,
  Implementer,
  Reviewer,
  Verifier,
};

struct CodingTask {
  std::string task;
  std::string verification_command;
};

struct AgentStep {
  AgentRole role;
  std::string prompt;
  std::string output;
};

struct SwarmTrace {
  std::vector<AgentStep> steps;
};

class SwarmOrchestrator {
 public:
  explicit SwarmOrchestrator(lumi::core::Scheduler& scheduler);

  SwarmTrace run_scripted_coding_task(const CodingTask& task);

 private:
  std::string run_agent(AgentRole role, const std::string& prompt);

  lumi::core::Scheduler& scheduler_;
  lumi::core::RequestId next_request_id_ = 1;
};

}  // namespace lumi::agent
```

- [ ] **Step 4: Implement scripted orchestrator**

Create `src/agent/agent.cpp` with:

```cpp
#include "lumi/agent/agent.hpp"

#include <stdexcept>

namespace lumi::agent {
namespace {

std::string role_name(AgentRole role) {
  switch (role) {
    case AgentRole::Planner:
      return "planner";
    case AgentRole::Implementer:
      return "implementer";
    case AgentRole::Reviewer:
      return "reviewer";
    case AgentRole::Verifier:
      return "verifier";
  }
  throw std::invalid_argument("unknown agent role");
}

std::string scripted_output(AgentRole role) {
  switch (role) {
    case AgentRole::Planner:
      return "Plan: inspect calculator.py and update add(a, b).";
    case AgentRole::Implementer:
      return "Patch: replace subtraction with addition.";
    case AgentRole::Reviewer:
      return "Review: patch is scoped and should be verified with unittest.";
    case AgentRole::Verifier:
      return "Verify: run configured command and accept only on success.";
  }
  throw std::invalid_argument("unknown agent role");
}

}  // namespace

SwarmOrchestrator::SwarmOrchestrator(lumi::core::Scheduler& scheduler)
    : scheduler_(scheduler) {}

SwarmTrace SwarmOrchestrator::run_scripted_coding_task(const CodingTask& task) {
  SwarmTrace trace;
  const std::vector<AgentRole> roles = {
      AgentRole::Planner,
      AgentRole::Implementer,
      AgentRole::Reviewer,
      AgentRole::Verifier,
  };

  for (const auto role : roles) {
    const auto prompt = role_name(role) + ": " + task.task + " using " +
                        task.verification_command;
    const auto output = run_agent(role, prompt);
    trace.steps.push_back({.role = role, .prompt = prompt, .output = output});
  }

  return trace;
}

std::string SwarmOrchestrator::run_agent(AgentRole role, const std::string& prompt) {
  const auto id = next_request_id_++;
  scheduler_.submit({.id = id,
                     .role = role_name(role),
                     .prompt = prompt,
                     .priority = lumi::core::RequestPriority::Normal,
                     .max_new_tokens = 4});
  while (scheduler_.result(id).status != lumi::core::RequestStatus::Completed) {
    scheduler_.step();
  }
  return scripted_output(role);
}

}  // namespace lumi::agent
```

- [ ] **Step 5: Wire agent library and test**

Modify `CMakeLists.txt`:

```cmake
add_library(lumi_agent STATIC
  src/agent/agent.cpp
)
target_include_directories(lumi_agent PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lumi_agent PUBLIC lumi_core)

add_executable(lumi_agent_demo_test tests/test_agent_demo.cpp)
target_link_libraries(lumi_agent_demo_test PRIVATE lumi_agent lumi_core)
add_test(NAME lumi_agent_demo_test COMMAND lumi_agent_demo_test)
```

Keep existing targets unchanged.

- [ ] **Step 6: Run tests**

Run:

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: agent-runtime test passes and all earlier tests continue passing.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt include/lumi/agent/agent.hpp src/agent/agent.cpp tests/test_agent_demo.cpp
git commit -m "feat: add scripted coding agents"
```

---

### Task 9: Controlled Coding Harness

**Files:**
- Create: `include/lumi/agent/harness.hpp`
- Create: `src/agent/harness.cpp`
- Modify: `tests/test_agent_demo.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Extend agent demo test with fixture repo verification**

Replace `tests/test_agent_demo.cpp` with:

```cpp
#include "lumi/agent/agent.hpp"
#include "lumi/agent/harness.hpp"
#include "lumi/core/backend.hpp"
#include "lumi/core/kv_cache.hpp"
#include "lumi/core/scheduler.hpp"
#include "test_main.hpp"

#include <filesystem>

int main() {
  const auto repo = std::filesystem::temp_directory_path() / "lumi_swarm_fixture_repo";
  lumi::agent::CodingHarness harness(repo);
  harness.create_calculator_fixture();
  LUMI_CHECK(!harness.run_verification("python3 -m unittest").passed);

  lumi::core::DeterministicBackend backend(512);
  lumi::core::PagedKvCache cache({.total_blocks = 32, .tokens_per_block = 16});
  lumi::core::Scheduler scheduler(backend, {.max_batch_size = 4}, &cache);

  lumi::agent::SwarmOrchestrator orchestrator(scheduler);
  const auto trace = orchestrator.run_scripted_coding_task(
      {.task = "fix add function", .verification_command = "python3 -m unittest"});

  LUMI_CHECK(trace.steps.size() == 4);
  harness.apply_edit({"calculator.py", "def add(a, b):\n    return a + b\n"});
  const auto verification = harness.run_verification("python3 -m unittest");
  LUMI_CHECK(verification.passed);
  LUMI_CHECK(verification.exit_code == 0);

  std::filesystem::remove_all(repo);
  return lumi_test_pass("agent demo with harness");
}
```

- [ ] **Step 2: Run test to verify it fails before harness implementation**

Run:

```bash
cmake --build build -j
```

Expected: build fails because `lumi/agent/harness.hpp` does not exist.

- [ ] **Step 3: Add harness API**

Create `include/lumi/agent/harness.hpp` with:

```cpp
#pragma once

#include <filesystem>
#include <string>

namespace lumi::agent {

struct FileEdit {
  std::filesystem::path relative_path;
  std::string contents;
};

struct VerificationResult {
  bool passed = false;
  int exit_code = 1;
};

class CodingHarness {
 public:
  explicit CodingHarness(std::filesystem::path repo_root);

  void create_calculator_fixture();
  void apply_edit(const FileEdit& edit);
  VerificationResult run_verification(const std::string& command) const;

 private:
  std::filesystem::path repo_root_;
};

}  // namespace lumi::agent
```

- [ ] **Step 4: Implement harness**

Create `src/agent/harness.cpp` with:

```cpp
#include "lumi/agent/harness.hpp"

#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace lumi::agent {

CodingHarness::CodingHarness(std::filesystem::path repo_root)
    : repo_root_(std::move(repo_root)) {}

void CodingHarness::create_calculator_fixture() {
  std::filesystem::remove_all(repo_root_);
  std::filesystem::create_directories(repo_root_);

  {
    std::ofstream source(repo_root_ / "calculator.py");
    source << "def add(a, b):\n"
           << "    return a - b\n";
  }

  {
    std::ofstream test(repo_root_ / "test_calculator.py");
    test << "import unittest\n"
         << "from calculator import add\n\n"
         << "class CalculatorTest(unittest.TestCase):\n"
         << "    def test_add(self):\n"
         << "        self.assertEqual(add(2, 3), 5)\n\n"
         << "if __name__ == '__main__':\n"
         << "    unittest.main()\n";
  }
}

void CodingHarness::apply_edit(const FileEdit& edit) {
  const auto target = repo_root_ / edit.relative_path;
  if (target.lexically_normal().string().find(repo_root_.lexically_normal().string()) != 0) {
    throw std::invalid_argument("edit path escapes repo root");
  }
  std::filesystem::create_directories(target.parent_path());
  std::ofstream output(target);
  output << edit.contents;
}

VerificationResult CodingHarness::run_verification(const std::string& command) const {
  const auto full_command = "cd '" + repo_root_.string() + "' && " + command;
  const int code = std::system(full_command.c_str());
  return {.passed = code == 0, .exit_code = code};
}

}  // namespace lumi::agent
```

- [ ] **Step 5: Wire harness into agent library**

Modify `CMakeLists.txt`:

```cmake
add_library(lumi_agent STATIC
  src/agent/agent.cpp
  src/agent/harness.cpp
)
```

Keep existing target settings unchanged.

- [ ] **Step 6: Run tests**

Run:

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: harness test passes and all earlier tests continue passing.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt include/lumi/agent/harness.hpp src/agent/harness.cpp tests/test_agent_demo.cpp
git commit -m "feat: add controlled coding harness"
```

---

### Task 10: CLI Demo And Benchmark App

**Files:**
- Create: `apps/lumi_swarm_cli.cpp`
- Create: `apps/lumi_swarm_bench.cpp`
- Modify: `README.md`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add CLI demo app**

Create `apps/lumi_swarm_cli.cpp` with:

```cpp
#include "lumi/agent/agent.hpp"
#include "lumi/agent/harness.hpp"
#include "lumi/core/backend.hpp"
#include "lumi/core/kv_cache.hpp"
#include "lumi/core/scheduler.hpp"

#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2 || std::string(argv[1]) != "demo") {
    std::cerr << "usage: lumi_swarm_cli demo\n";
    return 2;
  }

  const auto repo = std::filesystem::temp_directory_path() / "lumi_swarm_cli_demo";
  lumi::agent::CodingHarness harness(repo);
  harness.create_calculator_fixture();

  lumi::core::DeterministicBackend backend(1024);
  lumi::core::PagedKvCache cache({.total_blocks = 64, .tokens_per_block = 16});
  lumi::core::Scheduler scheduler(backend, {.max_batch_size = 4}, &cache);
  lumi::agent::SwarmOrchestrator orchestrator(scheduler);

  const auto before = harness.run_verification("python3 -m unittest");
  const auto trace = orchestrator.run_scripted_coding_task(
      {.task = "fix add function", .verification_command = "python3 -m unittest"});
  harness.apply_edit({"calculator.py", "def add(a, b):\n    return a + b\n"});
  const auto after = harness.run_verification("python3 -m unittest");

  std::cout << "before_passed=" << before.passed << "\n";
  std::cout << "after_passed=" << after.passed << "\n";
  std::cout << "agent_steps=" << trace.steps.size() << "\n";
  std::cout << "completed_requests=" << scheduler.metrics().completed_requests << "\n";
  std::cout << "emitted_tokens=" << scheduler.metrics().emitted_tokens << "\n";

  std::filesystem::remove_all(repo);
  return after.passed ? 0 : 1;
}
```

- [ ] **Step 2: Add benchmark app**

Create `apps/lumi_swarm_bench.cpp` with:

```cpp
#include "lumi/core/backend.hpp"
#include "lumi/core/kv_cache.hpp"
#include "lumi/core/scheduler.hpp"

#include <chrono>
#include <iostream>
#include <vector>

int main() {
  const std::vector<int> concurrencies = {1, 4, 16, 64};

  for (const int concurrency : concurrencies) {
    lumi::core::DeterministicBackend backend(4096);
    lumi::core::PagedKvCache cache({.total_blocks = 4096, .tokens_per_block = 16});
    lumi::core::Scheduler scheduler(backend,
                                    {.max_batch_size = static_cast<std::size_t>(concurrency)},
                                    &cache);

    for (int i = 0; i < concurrency; ++i) {
      scheduler.submit({.id = static_cast<lumi::core::RequestId>(i + 1),
                        .role = "bench-agent",
                        .prompt = "synthetic coding request with repo context",
                        .max_new_tokens = 64});
    }

    const auto start = std::chrono::steady_clock::now();
    while (!scheduler.is_idle()) {
      scheduler.step();
    }
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    const auto metrics = scheduler.metrics();
    const auto tokens_per_second =
        elapsed_ms == 0.0 ? 0.0
                          : (static_cast<double>(metrics.emitted_tokens) /
                             (elapsed_ms / 1000.0));

    std::cout << "{"
              << "\"concurrency\":" << concurrency << ","
              << "\"elapsed_ms\":" << elapsed_ms << ","
              << "\"tokens\":" << metrics.emitted_tokens << ","
              << "\"tokens_per_second\":" << tokens_per_second << ","
              << "\"scheduler_steps\":" << metrics.scheduler_steps << ","
              << "\"cache_used_blocks\":" << cache.metrics().used_blocks
              << "}\n";
  }

  return 0;
}
```

- [ ] **Step 3: Wire apps into build**

Add to `CMakeLists.txt`:

```cmake
add_executable(lumi_swarm_cli apps/lumi_swarm_cli.cpp)
target_link_libraries(lumi_swarm_cli PRIVATE lumi_agent lumi_core)

add_executable(lumi_swarm_bench apps/lumi_swarm_bench.cpp)
target_link_libraries(lumi_swarm_bench PRIVATE lumi_core)
```

- [ ] **Step 4: Update README**

Append to `README.md`:

````markdown

## Demo

```bash
./build/lumi_swarm_cli demo
```

Expected output includes `before_passed=0`, `after_passed=1`, and `completed_requests=4`.

## Benchmark

```bash
./build/lumi_swarm_bench
```

The benchmark prints one JSON-like line each for 1, 4, 16, and 64 concurrent synthetic coding-agent requests.
````

- [ ] **Step 5: Build and run apps**

Run:

```bash
cmake --build build -j
./build/lumi_swarm_cli demo
./build/lumi_swarm_bench
ctest --test-dir build --output-on-failure
```

Expected: demo exits 0, benchmark prints four lines, and tests pass.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt README.md apps/lumi_swarm_cli.cpp apps/lumi_swarm_bench.cpp
git commit -m "feat: add demo and benchmark apps"
```

---

### Task 11: Milestone Verification Report

**Files:**
- Create: `docs/reports/first-milestone-verification.md`
- Modify: `README.md`

- [ ] **Step 1: Run full verification**

Run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/lumi_swarm_cli demo
./build/lumi_swarm_bench
```

Expected: tests pass, demo exits 0, benchmark prints four concurrency lines.

- [ ] **Step 2: Write verification report**

Create `docs/reports/first-milestone-verification.md` with:

````markdown
# First Milestone Verification

## Build

Command:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Result: build completed successfully.

## Tests

Command:

```bash
ctest --test-dir build --output-on-failure
```

Result: all CTest tests passed.

## Demo

Command:

```bash
./build/lumi_swarm_cli demo
```

Expected signal:

```text
before_passed=0
after_passed=1
agent_steps=4
completed_requests=4
```

Result: demo produced the expected signal.

## Benchmark

Command:

```bash
./build/lumi_swarm_bench
```

Expected signal: one metrics line for concurrency 1, 4, 16, and 64.

Result: benchmark produced the expected concurrency lines.

## Scope Covered

- CUDA/C++ serving core scaffold
- deterministic backend
- continuous batching scheduler
- paged KV-cache accounting
- CUDA top-k sampling
- CUDA dot-product retrieval scoring
- thin coding-agent runtime
- controlled coding harness
- CLI demo
- synthetic benchmark
````

- [ ] **Step 3: Link report from README**

Append to `README.md`:

````markdown

## Verification Report

The first milestone verification report lives at `docs/reports/first-milestone-verification.md`.
````

- [ ] **Step 4: Commit**

```bash
git add README.md docs/reports/first-milestone-verification.md
git commit -m "docs: add first milestone verification report"
```

---

## Self-Review Checklist

- Spec coverage: Tasks 1-2 cover scaffold and deterministic backend; Tasks 3 and 5 cover continuous batching and host API; Tasks 4-5 cover KV-cache accounting; Task 6 covers CUDA sampling; Task 7 covers CUDA retrieval scoring; Tasks 8-10 cover the thin coding-agent demo and benchmark; Task 11 covers the performance and verification report.
- Type consistency: `GenerationRequest`, `Scheduler`, `PagedKvCache`, `SamplingConfig`, `RetrievalConfig`, `SwarmOrchestrator`, and `CodingHarness` names are introduced before use and reused consistently.
- Test path: every implementation task starts with a failing test or failing build step, then implementation, then verification.
- Scope discipline: full custom transformer kernels, multi-GPU support, training, web UI, and broad model-family support remain outside this milestone.
