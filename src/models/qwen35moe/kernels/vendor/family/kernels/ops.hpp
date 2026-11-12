#ifndef GUFO_MODELS_QWEN_HIP_OPS_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_HPP_

// Compatibility umbrella. New call sites should include the narrow operation
// family they use.
#include "src/models/qwen35moe/kernels/vendor/family/kernels/ops/attention.hpp"
#include "src/models/qwen35moe/kernels/vendor/family/kernels/ops/gemm.hpp"
#include "src/models/qwen35moe/kernels/vendor/family/kernels/ops/norm_residual.hpp"
#include "src/models/qwen35moe/kernels/vendor/family/kernels/ops/ssm.hpp"
#include "src/models/qwen35moe/kernels/vendor/family/kernels/ops/swiglu.hpp"
#include "src/models/qwen35moe/kernels/vendor/family/kernels/ops/token.hpp"

#endif  // GUFO_MODELS_QWEN_HIP_OPS_HPP_
