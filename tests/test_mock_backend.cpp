#include "lumi/core/backend.hpp"

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
    lumi::core::DeterministicBackend backend(128);

    lumi::core::GenerationRequest request;
    request.id = 7;
    request.role = "planner";
    request.prompt = "inspect repo";
    request.max_new_tokens = 3;

    const auto first = backend.prefill(request);
    CHECK(first.tokens.size() == 1);
    CHECK(!first.finished);

    const auto second = backend.decode(request, first.tokens);
    CHECK(second.tokens.size() == 2);
    CHECK(!second.finished);

    const auto third = backend.decode(request, second.tokens);
    CHECK(third.tokens.size() == 3);
    CHECK(third.finished);

    const auto repeated = backend.prefill(request);
    CHECK(repeated.tokens == first.tokens);

    std::cout << "[PASS] deterministic backend\n";
    return EXIT_SUCCESS;
}