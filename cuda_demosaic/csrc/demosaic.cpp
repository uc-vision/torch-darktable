#include <torch/extension.h>

torch::Tensor ppg_demosaic_cuda(torch::Tensor input, uint32_t filters, float median_threshold);
torch::Tensor rcd_demosaic_cuda(torch::Tensor input, uint32_t filters, float input_scale, float output_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ppg_demosaic", &ppg_demosaic_cuda, "PPG Demosaic (CUDA)",
          py::arg("input"), py::arg("filters"), py::arg("median_threshold") = 0.0f);
    m.def("rcd_demosaic", &rcd_demosaic_cuda, "RCD Demosaic (CUDA)",
          py::arg("input"), py::arg("filters"), py::arg("input_scale") = 1.0f, py::arg("output_scale") = 1.0f);
}