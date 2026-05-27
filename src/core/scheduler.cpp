#include "lumi/core/scheduler.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace lumi::core {

Scheduler::Scheduler(IModelBackend &backend, SchedulerConfig config)
    : backend_(backend), config_(config) {
    if (config_.max_batch_size == 0) {
        throw std::invalid_argument("max_batch_size must be greater than zero");
    }
}

void Scheduler::submit(GenerationRequest request) {
    if (request.id == 0) {
        throw std::invalid_argument("request id must be non-zero");
    }

    const bool duplicate_queued = std::any_of(
        queued_.begin(), queued_.end(), [&](const GenerationRequest &queued) {
            return queued.id == request.id;
        });

    const bool duplicate_result = results_.find(request.id) != results_.end();

    if (duplicate_queued || find_active(request.id) != nullptr ||
        duplicate_result) {
        throw std::invalid_argument("duplicate request id");
    }

    queued_.push_back(std::move(request));
    std::stable_sort(queued_.begin(), queued_.end(), has_higher_priority);

    metrics_.submitted_requests += 1;
    metrics_.queued_requests = static_cast<std::uint32_t>(queued_.size());
}

bool Scheduler::cancel(RequestId id) {
    auto queued_it = std::find_if(
        queued_.begin(), queued_.end(),
        [id](const GenerationRequest &request) { return request.id == id; });

    if (queued_it != queued_.end()) {
        results_[id] = {.id = id, .status = RequestStatus::Cancelled};
        queued_.erase(queued_it);

        metrics_.cancelled_requests += 1;
        metrics_.queued_requests = static_cast<std::uint32_t>(queued_.size());
        return true;
    }

    auto active_it = std::find_if(
        active_.begin(), active_.end(),
        [id](const ActiveRequest &active) { return active.request.id == id; });

    if (active_it != active_.end()) {
        results_[id] = {
            .id = id,
            .status = RequestStatus::Cancelled,
            .tokens = active_it->tokens,
        };

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

    for (auto &active : active_) {
        // Prefill starts a request; decode appends one token step.
        BackendStep backend_step =
            active.tokens.empty()
                ? backend_.prefill(active.request)
                : backend_.decode(active.request, active.tokens);

        const auto previous_size = active.tokens.size();
        active.tokens = std::move(backend_step.tokens);

        metrics_.emitted_tokens += active.tokens.size() - previous_size;
        active.status = backend_step.finished ? RequestStatus::Completed
                                              : RequestStatus::Decoding;

        if (backend_step.finished) {
            completed.push_back(active.request.id);
        }
    }

    for (const auto id : completed) {
        auto it = std::find_if(active_.begin(), active_.end(),
                               [id](const ActiveRequest &active) {
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
    if (const auto it = results_.find(id); it != results_.end()) {
        return it->second;
    }

    for (const auto &active : active_) {
        if (active.request.id == id) {
            return {
                .id = id,
                .status = active.status,
                .tokens = active.tokens,
            };
        }
    }

    for (const auto &queued : queued_) {
        if (queued.id == id) {
            return {.id = id, .status = RequestStatus::Queued};
        }
    }

    return {.id = id,
            .status = RequestStatus::Failed,
            .text = "unknown request id"};
}

SchedulerMetrics Scheduler::metrics() const {
    return metrics_;
}

bool Scheduler::has_higher_priority(const GenerationRequest &lhs,
                                    const GenerationRequest &rhs) {
    return static_cast<int>(lhs.priority) > static_cast<int>(rhs.priority);
}

void Scheduler::activate_requests() {
    while (!queued_.empty() && active_.size() < config_.max_batch_size) {
        auto request = std::move(queued_.front());
        queued_.pop_front();

        active_.push_back({
            .request = std::move(request),
            .status = RequestStatus::Prefill,
        });
    }

    metrics_.queued_requests = static_cast<std::uint32_t>(queued_.size());
    metrics_.active_requests = static_cast<std::uint32_t>(active_.size());
}

void Scheduler::complete_request(const ActiveRequest &active) {
    results_[active.request.id] = {
        .id = active.request.id,
        .status = RequestStatus::Completed,
        .tokens = active.tokens,
    };

    metrics_.completed_requests += 1;
}

Scheduler::ActiveRequest *Scheduler::find_active(RequestId id) {
    auto it = std::find_if(
        active_.begin(), active_.end(),
        [id](const ActiveRequest &active) { return active.request.id == id; });

    return it == active_.end() ? nullptr : &(*it);
}

} // namespace lumi::core
