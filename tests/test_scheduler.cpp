#include "lumi/core/backend.hpp"
#include "lumi/core/scheduler.hpp"

#include <cstdlib>
#include <iostream>

#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::cerr << "CHECK failed: " #condition << " at " << __FILE__     \
                      << ":" << __LINE__ << "\n";                              \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (false)

int main() {
    lumi::core::DeterministicBackend backend(256);
    lumi::core::Scheduler scheduler(backend, {.max_batch_size = 2});

    scheduler.submit(
        {.id = 1, .role = "planner", .prompt = "plan", .max_new_tokens = 2});
    scheduler.submit(
        {.id = 2, .role = "reviewer", .prompt = "review", .max_new_tokens = 2});
    scheduler.submit({.id = 3,
                      .role = "implementer",
                      .prompt = "patch",
                      .max_new_tokens = 1});

    scheduler.step();

    CHECK(scheduler.metrics().submitted_requests == 3);
    CHECK(scheduler.metrics().active_requests == 2);
    CHECK(scheduler.metrics().queued_requests == 1);
    CHECK(scheduler.metrics().last_batch_size == 2);

    while (!scheduler.is_idle()) {
        scheduler.step();
    }

    CHECK(scheduler.result(1).status == lumi::core::RequestStatus::Completed);
    CHECK(scheduler.result(2).status == lumi::core::RequestStatus::Completed);
    CHECK(scheduler.result(3).status == lumi::core::RequestStatus::Completed);

    CHECK(scheduler.result(1).tokens.size() == 2);
    CHECK(scheduler.result(2).tokens.size() == 2);
    CHECK(scheduler.result(3).tokens.size() == 1);

    scheduler.submit(
        {.id = 4, .role = "planner", .prompt = "cancel", .max_new_tokens = 3});
    CHECK(scheduler.cancel(4));
    CHECK(scheduler.result(4).status == lumi::core::RequestStatus::Cancelled);

    std::cout << "[PASS] scheduler batching\n";
    return EXIT_SUCCESS;
}