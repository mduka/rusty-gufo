#pragma once

// The extension's q35/ns.hpp with its `strycast` namespace renamed to
// gufo::models::qwen35moe: the copy of the Qwen kernels lives in its own
// namespace, so it links beside the dense Qwen kernels (gufo::hip) without
// duplicate symbols. Each namespace that mirrors a gufo one also sees gufo's
// names, so the core headers' types (core::GgmlType, hip::detail telemetry)
// resolve as in the extension build. The extension's model_config.hpp, a host
// type no kernel source uses, is left out.

namespace gufo {}
namespace gufo::cache {}
namespace gufo::core {}
namespace gufo::crypto {}
namespace gufo::ext {}
namespace gufo::hip {}
namespace gufo::hip::detail {}
namespace gufo::json {}
namespace gufo::platform {}
namespace gufo::quant {}
namespace gufo::sampling {}
namespace gufo::server {}
namespace gufo::speculative {}

namespace gufo::models::qwen35moe { using namespace ::gufo; }
namespace gufo::models::qwen35moe::cache { using namespace ::gufo::cache; }
namespace gufo::models::qwen35moe::core { using namespace ::gufo::core; }
namespace gufo::models::qwen35moe::crypto { using namespace ::gufo::crypto; }
namespace gufo::models::qwen35moe::ext { using namespace ::gufo::ext; }
namespace gufo::models::qwen35moe::hip { using namespace ::gufo::hip; }
namespace gufo::models::qwen35moe::hip::detail { using namespace ::gufo::hip::detail; }
namespace gufo::models::qwen35moe::json { using namespace ::gufo::json; }
namespace gufo::models::qwen35moe::platform { using namespace ::gufo::platform; }
namespace gufo::models::qwen35moe::quant { using namespace ::gufo::quant; }
namespace gufo::models::qwen35moe::sampling { using namespace ::gufo::sampling; }
namespace gufo::models::qwen35moe::server { using namespace ::gufo::server; }
namespace gufo::models::qwen35moe::speculative { using namespace ::gufo::speculative; }
