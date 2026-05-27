#pragma once

#include <cstdint>

namespace lumi::core {

struct SchedulerMetrics {
    std::uint64_t submitted_requests = 0;
    std::uint64_t completed_requests = 0;
    std::uint64_t cancelled_requests = 0;
    std::uint64_t scheduler_steps = 0;
    std::uint64_t emitted_tokens = 0;

    std::uint32_t queued_requests = 0;
    std::uint32_t active_requests = 0;
    std::uint32_t last_batch_size = 0;
};

} // namespace lumi::core