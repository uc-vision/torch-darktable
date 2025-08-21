#include <torch/extension.h>

torch::Tensor ppg_demosaic_cuda(torch::Tensor input, uint32_t filters, float median_threshold);
torch::Tensor rcd_demosaic_cuda(torch::Tensor input, uint32_t filters, float input_scale, float output_scale);
torch::Tensor postprocess_demosaic_cuda(torch::Tensor input, uint32_t filters, int color_smoothing_passes, 
                                       bool green_eq_local, bool green_eq_global, float green_eq_threshold);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ppg_demosaic", &ppg_demosaic_cuda, "PPG Demosaic (CUDA)",
          py::arg("input"), py::arg("filters"), py::arg("median_threshold") = 0.0f);
    m.def("rcd_demosaic", &rcd_demosaic_cuda, "RCD Demosaic (CUDA)",
          py::arg("input"), py::arg("filters"), py::arg("input_scale") = 1.0f, py::arg("output_scale") = 1.0f);
    m.def("postprocess_demosaic", &postprocess_demosaic_cuda, "Post-process Demosaic (CUDA)",
          py::arg("input"), py::arg("filters"), py::arg("color_smoothing_passes") = 0,
          py::arg("green_eq_local") = false, py::arg("green_eq_global") = false, 
          py::arg("green_eq_threshold") = 0.0001f);
}