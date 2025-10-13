#include <vector>
#include <cassert>
#include <memory>
#include <iostream>
#include <sstream>

#include <ATen/cuda/CUDAContext.h>
#include <nvjpeg.h>

#include "jpeg_encoder.h"

nvjpegImage_t interleavedImage(torch::Tensor const& image) {
  TORCH_CHECK(image.dim() == 3 && image.size(2) == 3, 
    "for interleaved (BGRI, RGBI) expected 3D tensor (H, W, C)");

  nvjpegImage_t img; 

  for(int i = 0; i < NVJPEG_MAX_COMPONENT; i++){
      img.channel[i] = nullptr;
      img.pitch[i] = 0;
  }

  img.pitch[0] = (unsigned int)at::stride(image, 0);
  img.channel[0] = (unsigned char*)image.data_ptr();

  return img;
}


nvjpegImage_t planarImage(torch::Tensor const& image) {
  TORCH_CHECK(image.dim() == 3 && image.size(0) == 3, 
    "for planar (BGR, RGB) expected 3D tensor (C, H, W)");

  nvjpegImage_t img; 

  for(int i = 0; i < NVJPEG_MAX_COMPONENT; i++){
      img.channel[i] = nullptr;
      img.pitch[i] = 0;
  }

  size_t plane_stride = at::stride(image, 0);

  for(int i = 0; i < 3; i++) {
    img.pitch[i] = (unsigned int)at::stride(image, 1);
    img.channel[i] = (unsigned char*)image.data_ptr() + plane_stride * i;
  }
  
  
  return img;
}


inline const char* error_string(nvjpegStatus_t code) {
  switch(code) {
    case NVJPEG_STATUS_SUCCESS: return "success";
    case NVJPEG_STATUS_NOT_INITIALIZED: return "not initialized";
    case NVJPEG_STATUS_INVALID_PARAMETER: return "invalid parameter";
    case NVJPEG_STATUS_BAD_JPEG: return "bad jpeg";
    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED: return "not supported";
    case NVJPEG_STATUS_ALLOCATOR_FAILURE: return "allocation failed";
    case NVJPEG_STATUS_EXECUTION_FAILED: return "execution failed";
    case NVJPEG_STATUS_ARCH_MISMATCH: return "arch mismatch";
    case NVJPEG_STATUS_INTERNAL_ERROR: return "internal error";
    default: return "unknown";
  }
}


static nvjpegInputFormat_t to_nvjpeg_format(JpegInputFormat format) {
  switch(format) {
    case JpegInputFormat::BGR: return NVJPEG_INPUT_BGR;
    case JpegInputFormat::RGB: return NVJPEG_INPUT_RGB;
    case JpegInputFormat::BGRI: return NVJPEG_INPUT_BGRI;
    case JpegInputFormat::RGBI: return NVJPEG_INPUT_RGBI;
  }
  throw std::runtime_error("Invalid input format");
}

static nvjpegChromaSubsampling_t to_nvjpeg_subsampling(JpegSubsampling subsampling) {
  switch(subsampling) {
    case JpegSubsampling::CSS_444: return NVJPEG_CSS_444;
    case JpegSubsampling::CSS_422: return NVJPEG_CSS_422;
    case JpegSubsampling::CSS_GRAY: return NVJPEG_CSS_GRAY;
  }
  throw std::runtime_error("Invalid subsampling");
}

JpegException::JpegException(std::string const& _context, int _code) :
  code(_code), context(_context)
{ }

const char* JpegException::what() const throw() {
  std::stringstream ss;
  ss << context << ", nvjpeg error " << code << ": " << error_string((nvjpegStatus_t)code);
  return ss.str().c_str();
}

inline void check_nvjpeg(std::string const &message, nvjpegStatus_t code) {
  if (NVJPEG_STATUS_SUCCESS != code){
      throw JpegException(message, (int)code);
  }
}

struct JpegCoderImpl : public JpegCoder {
  JpegCoderImpl() {
    nvjpegCreateSimple(&nv_handle);
    nvjpegJpegStateCreate(nv_handle, &nv_statue);
    nvjpegEncoderStateCreate(nv_handle, &enc_state, NULL);
  }

  ~JpegCoderImpl() {
    nvjpegJpegStateDestroy(nv_statue);
    nvjpegEncoderStateDestroy(enc_state);
    nvjpegDestroy(nv_handle);
  }

  inline nvjpegEncoderParams_t createParams(int quality, nvjpegChromaSubsampling_t subsampling, bool progressive, cudaStream_t stream = nullptr) {
    nvjpegEncoderParams_t params;

    nvjpegEncoderParamsCreate(nv_handle, &params, stream);

    nvjpegEncoderParamsSetQuality(params, quality, stream);
    nvjpegEncoderParamsSetOptimizedHuffman(params, 1, stream);
    nvjpegEncoderParamsSetSamplingFactors(params, subsampling, stream);
    
    nvjpegJpegEncoding_t jpeg_encoding = progressive ? NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN : NVJPEG_ENCODING_BASELINE_DCT;
    nvjpegEncoderParamsSetEncoding(params, jpeg_encoding, stream);

    return params;
  }

  nvjpegImage_t createImage(torch::Tensor const& data, nvjpegInputFormat_t input_format, size_t &width, size_t &height) const {
    TORCH_CHECK(data.is_cuda(), "Input image should be on CUDA device");
    TORCH_CHECK(data.dtype() == torch::kU8, "Input image should be uint8");
    TORCH_CHECK(data.is_contiguous(), "Input data should be contiguous");

    bool interleaved = input_format == NVJPEG_INPUT_BGRI || input_format == NVJPEG_INPUT_RGBI;

    if(interleaved) {
      width = data.size(1);
      height = data.size(0);
      return interleavedImage(data);
    } else {
      width = data.size(2);
      height = data.size(1);
      return planarImage(data);
    }
  }


  torch::Tensor encode(torch::Tensor const& data, int quality, JpegInputFormat input_format, JpegSubsampling subsampling, bool progressive) override {
    nvjpegInputFormat_t nv_format = to_nvjpeg_format(input_format);
    nvjpegChromaSubsampling_t nv_subsampling = to_nvjpeg_subsampling(subsampling);
    py::gil_scoped_release release;
    size_t width, height;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    nvjpegEncoderParams_t params = createParams(quality, nv_subsampling, progressive, stream);
    nvjpegImage_t image = createImage(data, nv_format, width, height);

    check_nvjpeg("nvjpegEncodeImage", 
      nvjpegEncodeImage(nv_handle, enc_state, params, 
        &image, nv_format, width, height, stream));

    size_t length;
    nvjpegEncodeRetrieveBitstream(nv_handle, enc_state, NULL, &length, stream);
    auto buffer = torch::empty({ int(length) }, torch::TensorOptions().dtype(torch::kUInt8));

    nvjpegEncodeRetrieveBitstream(nv_handle, enc_state, (unsigned char*)buffer.data_ptr(), &length, stream);
    nvjpegEncoderParamsDestroy(params);

    return buffer;
  }


  nvjpegHandle_t nv_handle;
  nvjpegJpegState_t nv_statue;
  nvjpegEncoderState_t enc_state;
};

std::shared_ptr<JpegCoder> create_jpeg_coder() {
  return std::make_shared<JpegCoderImpl>();
}



