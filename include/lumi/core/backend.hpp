#pragma once

#include "lumi/core/types.hpp"

#include <cstdint>
#include <vector>

namespace lumi::core {

class IModelBackend {
  public:
    virtual ~IModelBackend() = default;

    virtual BackendStep prefill(const GenerationRequest &request) = 0;

    virtual BackendStep decode(const GenerationRequest &request,
                               const std::vector<TokenId> &existing_tokens) = 0;
};

class DeterministicBackend final : public IModelBackend {
  public:
    explicit DeterministicBackend(std::int32_t vocab_size);

    BackendStep prefill(const GenerationRequest &request) override;

    BackendStep decode(const GenerationRequest &request,
                       const std::vector<TokenId> &existing_tokens) override;

  private:
    TokenId token_for(const GenerationRequest &request,
                      std::uint32_t position) const;

    std::int32_t vocab_size_;
};

} // namespace lumi::core