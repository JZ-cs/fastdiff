#include <torch/extension.h>
at::Tensor ffc_forward_cuda(
    at::Tensor x,
    at::Tensor W,
    std::string &fcid
);
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor ffc_forward(
    at::Tensor x,
    at::Tensor W,
    std::string fcid
){
    CHECK_INPUT(x);
    CHECK_INPUT(W);
    return ffc_forward_cuda(x, W, fcid);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ffc_forward, "fast linear");
}