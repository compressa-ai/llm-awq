#include <torch/extension.h>

torch::Tensor gemm_forward_cuda(torch::Tensor _in_feats, torch::Tensor _kernel,
    torch::Tensor _scaling_factors, torch::Tensor _zeros, int split_k_iters);

torch::Tensor dequantize_weights_cuda(torch::Tensor _kernel,
    torch::Tensor _scaling_factors, torch::Tensor _zeros, int split_k_iters, int thx, int thy, bool dbg);
