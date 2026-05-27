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

} // namespace lumi::core