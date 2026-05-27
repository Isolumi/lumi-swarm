#include "lumi/core/backend.hpp"

#include <stdexcept>

namespace lumi::core {

DeterministicBackend::DeterministicBackend(std::int32_t vocab_size)
    : vocab_size_(vocab_size) {
    if (vocab_size_ <= 1) {
        throw std::invalid_argument("vocab_size must be greater than 1");
    }
}

BackendStep DeterministicBackend::prefill(const GenerationRequest &request) {
    BackendStep step;

    if (request.max_new_tokens == 0) {
        step.finished = true;
        return step;
    }

    step.tokens.push_back(token_for(request, 0));
    step.finished = request.max_new_tokens == 1;
    return step;
}

BackendStep
DeterministicBackend::decode(const GenerationRequest &request,
                             const std::vector<TokenId> &existing_tokens) {
    BackendStep step;
    step.tokens = existing_tokens;

    if (step.tokens.size() < request.max_new_tokens) {
        step.tokens.push_back(
            token_for(request, static_cast<std::uint32_t>(step.tokens.size())));
    }

    step.finished = step.tokens.size() >= request.max_new_tokens;
    return step;
}

TokenId DeterministicBackend::token_for(const GenerationRequest &request,
                                        std::uint32_t position) const {
    const auto seed = static_cast<std::uint64_t>(request.id * 9973U) +
                      static_cast<std::uint64_t>(position * 37U) +
                      static_cast<std::uint64_t>(request.prompt.size());

    return static_cast<TokenId>(seed % static_cast<std::uint64_t>(vocab_size_));
}

} // namespace lumi::core