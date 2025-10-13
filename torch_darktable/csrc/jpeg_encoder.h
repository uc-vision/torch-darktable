#pragma once

#include <torch/torch.h>
#include <torch/extension.h>

enum class JpegInputFormat {
  BGR,
  RGB,
  BGRI,
  RGBI
};

enum class JpegSubsampling {
  CSS_444,
  CSS_422,
  CSS_GRAY
};

class JpegException : public std::exception {
  int code;
  std::string context;

public:
  JpegException(std::string const& _context, int _code);
  const char* what() const throw();
};

struct JpegCoder {
  virtual ~JpegCoder() = default;
  
  virtual torch::Tensor encode(torch::Tensor const& data, int quality, 
                                JpegInputFormat input_format, 
                                JpegSubsampling subsampling, 
                                bool progressive) = 0;
};

std::shared_ptr<JpegCoder> create_jpeg_coder();
