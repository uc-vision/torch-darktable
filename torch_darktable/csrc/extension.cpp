#include "debayer/demosaic.h"
#include "local_contrast/laplacian.h"
#include "local_contrast/bilateral.h"
#include "color_conversions.h"
#include "packed.h"
#include "tonemap/tonemap.h"
#include "white_balance.h"
#include "denoise/denoise.h"

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

 


// Forward declarations for implementations in kernel files
std::shared_ptr<PPG> create_ppg(torch::Device device, int width, int height, 
  BayerPattern pattern, float median_threshold = 0.0f);

std::shared_ptr<RCD> create_rcd(torch::Device device, int width, int height, 
  BayerPattern pattern, float input_scale = 1.0f, float output_scale = 1.0f);

std::shared_ptr<PostProcess> create_postprocess(torch::Device device,
  int width, int height, BayerPattern pattern,
  int color_smoothing_passes,
  bool green_eq_local, bool green_eq_global,
  float green_eq_threshold);

torch::Tensor bilinear5x5_demosaic(const torch::Tensor& input, BayerPattern pattern);

std::shared_ptr<Laplacian> create_laplacian(torch::Device device, 
  int width, int height,
  int num_gamma, 
  float sigma, float shadows, float highlights, float clarity);

std::shared_ptr<Bilateral> create_bilateral(torch::Device device,
  int width, int height, float sigma_s, float sigma_r);

std::shared_ptr<Wiener> create_wiener(torch::Device device, int width, int height,
  int overlap_factor, int tile_size, int channels);


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Minimal helper to reduce boilerplate for bulk setters
 



    // Algorithm classes for direct use
    py::class_<PPG, std::shared_ptr<PPG>>(m, "PPG")
        .def(py::init(&create_ppg), "Create PPG demosaic",
             py::arg("device"), py::arg("width"), py::arg("height"),
             py::arg("pattern"), py::arg("median_threshold") = 0.0f)
        .def("process", &PPG::process, "Process image with PPG algorithm",
             py::arg("input"))
        .def_property("median_threshold", &PPG::get_median_threshold, &PPG::set_median_threshold, "Median threshold parameter");

    py::class_<RCD, std::shared_ptr<RCD>>(m, "RCD")
        .def(py::init(&create_rcd), "Create RCD demosaic",
             py::arg("device"), py::arg("width"), py::arg("height"),
             py::arg("pattern"), py::arg("input_scale") = 1.0f, py::arg("output_scale") = 1.0f)
        .def("process", &RCD::process, "Process image with RCD algorithm",
             py::arg("input"))
        .def_property("input_scale", &RCD::get_input_scale, &RCD::set_input_scale, "Input scaling")
        .def_property("output_scale", &RCD::get_output_scale, &RCD::set_output_scale, "Output scaling");

    py::class_<PostProcess, std::shared_ptr<PostProcess>>(m, "PostProcess")
        .def(py::init(&create_postprocess), "Create post-process algorithm",
             py::arg("device"), py::arg("width"), py::arg("height"),
             py::arg("pattern"), py::arg("color_smoothing_passes") = 0,
             py::arg("green_eq_local") = false, py::arg("green_eq_global") = false,
             py::arg("green_eq_threshold") = 0.04f)
        .def("process", &PostProcess::process, "Process image with post-processing",
             py::arg("input"))
        .def_property("color_smoothing_passes", &PostProcess::get_color_smoothing_passes, &PostProcess::set_color_smoothing_passes, "Number of color smoothing passes")
        .def_property("green_eq_local", &PostProcess::get_green_eq_local, &PostProcess::set_green_eq_local, "Enable local green equilibration")
        .def_property("green_eq_global", &PostProcess::get_green_eq_global, &PostProcess::set_green_eq_global, "Enable global green equilibration")
        .def_property("green_eq_threshold", &PostProcess::get_green_eq_threshold, &PostProcess::set_green_eq_threshold, "Green equilibration threshold");



    py::class_<Laplacian, std::shared_ptr<Laplacian>>(m, "Laplacian")
        .def(py::init(&create_laplacian), "Create Laplacian algorithm",
             py::arg("device"), 
             py::arg("width"), py::arg("height"),
             py::arg("num_gamma") = 6, 
             py::arg("sigma") = 0.2f,
             py::arg("shadows") = 1.0f, py::arg("highlights") = 1.0f, py::arg("clarity") = 0.0f)
        .def("process", &Laplacian::process, "Process image with Laplacian filter",
             py::arg("input"))
        .def_property("sigma", &Laplacian::get_sigma, &Laplacian::set_sigma, "Sigma parameter")
        .def_property("shadows", &Laplacian::get_shadows, &Laplacian::set_shadows, "Shadows parameter")
        .def_property("highlights", &Laplacian::get_highlights, &Laplacian::set_highlights, "Highlights parameter")
        .def_property("clarity", &Laplacian::get_clarity, &Laplacian::set_clarity, "Clarity parameter");


    py::class_<Bilateral, std::shared_ptr<Bilateral>>(m, "Bilateral")
        .def(py::init(&create_bilateral), "Create Bilateral grid algorithm",
             py::arg("device"),
             py::arg("width"), py::arg("height"),
             py::arg("sigma_s") = 8.0f, py::arg("sigma_r") = 0.1f)
        .def("process", &Bilateral::process, "Local contrast on luminance with bilateral grid",
             py::arg("luminance"), py::arg("detail"))
        .def_property("sigma_s", &Bilateral::get_sigma_s, &Bilateral::set_sigma_s, "Spatial sigma")
        .def_property("sigma_r", &Bilateral::get_sigma_r, &Bilateral::set_sigma_r, "Range sigma");

    // Note: Bilateral should be used as a class object from Python


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

    m.def("color_transform_3x3", &color_transform_3x3, "Apply 3x3 color matrix transform with clamping",
          py::arg("input"), py::arg("matrix_3x3"));

    // Packed 12-bit encoding/decoding functions
    m.def("encode12_u16", &encode12_u16, "Encode uint16 values to packed 12-bit",
          py::arg("input"), py::arg("ids_format") = false);
    m.def("encode12_float", &encode12_float, "Encode float values to packed 12-bit", 
          py::arg("input"), py::arg("ids_format") = false, py::arg("scaled") = true);
    
    m.def("decode12_float", &decode12_float, "Decode packed 12-bit to float",
          py::arg("input"), py::arg("ids_format") = false, py::arg("scaled") = true);
    m.def("decode12_half", &decode12_half, "Decode packed 12-bit to half precision",
          py::arg("input"), py::arg("ids_format") = false, py::arg("scaled") = true);
    m.def("decode12_u16", &decode12_u16, "Decode packed 12-bit to uint16",
          py::arg("input"), py::arg("ids_format") = false);

    // Tone mapping functions
    m.def("compute_image_bounds", &compute_image_bounds, "Compute min/max bounds of image",
          py::arg("image"), py::arg("stride") = 8);
    m.def("compute_image_metrics", &compute_image_metrics, "Compute 9-vector image metrics for tone mapping",
          py::arg("images"), py::arg("stride") = 8, py::arg("min_gray") = 1e-4f);
    m.def("reinhard_tonemap", &reinhard_tonemap, "Apply Reinhard tone mapping",
          py::arg("image"), py::arg("metrics"), 
          py::arg("gamma") = 1.0f, py::arg("intensity") = 1.0f, 
          py::arg("light_adapt") = 0.8f);
    m.def("aces_tonemap", &aces_tonemap, "Apply ACES tone mapping",
          py::arg("image"), py::arg("gamma") = 2.2f);

    // Expose BayerPattern enum
    py::enum_<BayerPattern>(m, "BayerPattern")
        .value("RGGB", BayerPattern::RGGB)
        .value("BGGR", BayerPattern::BGGR)
        .value("GRBG", BayerPattern::GRBG)
        .value("GBRG", BayerPattern::GBRG);

    // Bilinear 5x5 demosaic function
    m.def("bilinear5x5_demosaic", &bilinear5x5_demosaic, "Apply 5x5 bilinear demosaic",
          py::arg("input"), py::arg("pattern"));

    // White balance functions
    m.def("apply_white_balance", &apply_white_balance, "Apply white balance gains to Bayer image",
          py::arg("bayer_image"), py::arg("gains"), py::arg("pattern"));
    m.def("estimate_white_balance", &estimate_white_balance, "Estimate white balance from Bayer images",
          py::arg("bayer_images"), py::arg("pattern"), py::arg("quantile") = 0.95f, py::arg("stride") = 8);

    // Wiener denoiser
    py::class_<Wiener, std::shared_ptr<Wiener>>(m, "Wiener")
        .def(py::init(&create_wiener), "Create Wiener denoiser",
             py::arg("device"), py::arg("width"), py::arg("height"),
             py::arg("overlap_factor") = 4, py::arg("tile_size") = 32, py::arg("channels") = 3)
        .def("process", &Wiener::process, "Process image with Wiener filter",
             py::arg("input"), py::arg("noise_sigmas"))
        .def_property_readonly("overlap_factor", &Wiener::get_overlap_factor, "Overlap factor (read-only)");


}

#pragma GCC diagnostic pop