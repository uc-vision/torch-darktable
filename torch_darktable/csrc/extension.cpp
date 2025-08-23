#include "demosaic.h"
#include "laplacian.h"
#include "color_conversions.h"

#include <ATen/ATen.h>


// Forward declarations for implementations in kernel files
std::shared_ptr<PPG> create_ppg(torch::Device device, int width, int height, 
  uint32_t filters, float median_threshold = 0.0f);

std::shared_ptr<RCD> create_rcd(torch::Device device, int width, int height, 
  uint32_t filters, float input_scale = 1.0f, float output_scale = 1.0f);

std::shared_ptr<PostProcess> create_postprocess(torch::Device device,
  int width, int height, uint32_t filters,
  int color_smoothing_passes,
  bool green_eq_local, bool green_eq_global,
  float green_eq_threshold);

std::shared_ptr<Laplacian> create_laplacian(torch::Device device, 
  int width, int height,
  int num_gamma, 
  float sigma, float shadows, float highlights, float clarity);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {



    // Algorithm classes for direct use
    py::class_<PPG, std::shared_ptr<PPG>>(m, "PPG")
        .def(py::init(&create_ppg), "Create PPG demosaic",
             py::arg("device"), py::arg("width"), py::arg("height"),
             py::arg("filters"), py::arg("median_threshold") = 0.0f)
        .def("process", &PPG::process, "Process image with PPG algorithm",
             py::arg("input"));

    py::class_<RCD, std::shared_ptr<RCD>>(m, "RCD")
        .def(py::init(&create_rcd), "Create RCD demosaic",
             py::arg("device"), py::arg("width"), py::arg("height"),
             py::arg("filters"), py::arg("input_scale") = 1.0f, py::arg("output_scale") = 1.0f)
        .def("process", &RCD::process, "Process image with RCD algorithm",
             py::arg("input"));

    py::class_<PostProcess, std::shared_ptr<PostProcess>>(m, "PostProcess")
        .def(py::init(&create_postprocess), "Create post-process algorithm",
             py::arg("device"), py::arg("width"), py::arg("height"),
             py::arg("filters"), py::arg("color_smoothing_passes") = 0,
             py::arg("green_eq_local") = false, py::arg("green_eq_global") = false,
             py::arg("green_eq_threshold") = 0.04f)
        .def("process", &PostProcess::process, "Process image with post-processing",
             py::arg("input"));



    py::class_<Laplacian, std::shared_ptr<Laplacian>>(m, "Laplacian")
        .def(py::init(&create_laplacian), "Create Laplacian algorithm",
             py::arg("device"), 
             py::arg("width"), py::arg("height"),
             py::arg("num_gamma") = 6, 
             py::arg("sigma") = 0.2f,
             py::arg("shadows") = 0.0f, py::arg("highlights") = 0.0f, py::arg("clarity") = 0.0f)
        .def("process", &Laplacian::process, "Process image with Laplacian filter",
             py::arg("input"))
        .def("get_parameters", &Laplacian::get_parameters, "Get current parameters")
        .def("set_sigma", &Laplacian::set_sigma, "Set sigma parameter")
        .def("set_shadows", &Laplacian::set_shadows, "Set shadows parameter")
        .def("set_highlights", &Laplacian::set_highlights, "Set highlights parameter")
        .def("set_clarity", &Laplacian::set_clarity, "Set clarity parameter");


    // Color conversion functions
    m.def("compute_luminance", &compute_luminance, "Compute luminance from RGB image",
          py::arg("rgb"));
    m.def("modify_luminance", &modify_luminance, "Modify luminance of RGB image",
          py::arg("rgb"), py::arg("new_luminance"));

    m.def("rgb_to_xyz", &rgb_to_xyz, "Convert RGB to XYZ color space",
          py::arg("rgb"));
    m.def("xyz_to_lab", &xyz_to_lab, "Convert XYZ to LAB color space",
          py::arg("xyz"));
    m.def("lab_to_xyz", &lab_to_xyz, "Convert LAB to XYZ color space",
          py::arg("lab"));
    m.def("xyz_to_rgb", &xyz_to_rgb, "Convert XYZ to RGB color space",
          py::arg("xyz"));

    m.def("rgb_to_lab", &rgb_to_lab, "Convert RGB to LAB color space",
          py::arg("rgb"));
    m.def("lab_to_rgb", &lab_to_rgb, "Convert LAB to RGB color space",
          py::arg("lab"));

}