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
    Scheduler(IModelBackend &backend, SchedulerConfig config);

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

    // Higher-priority requests should enter the active batch first.
    static bool has_higher_priority(const GenerationRequest &lhs,
                                    const GenerationRequest &rhs);

    void activate_requests();
    void complete_request(const ActiveRequest &active);
    ActiveRequest *find_active(RequestId id);

    IModelBackend &backend_;
    SchedulerConfig config_;

    std::deque<GenerationRequest> queued_;
    std::vector<ActiveRequest> active_;
    std::unordered_map<RequestId, RequestResult> results_;
    SchedulerMetrics metrics_;
};

} // namespace lumi::core