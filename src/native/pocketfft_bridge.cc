#include <complex>
#include <cstddef>

#include "pocketfft_hdronly.h"

extern "C" int pocketfft_r2c_f32(std::size_t len, const float *input, float *output_interleaved) {
    using namespace pocketfft;

    try {
        shape_t shape_in{len};
        stride_t stride_in{static_cast<std::ptrdiff_t>(sizeof(float))};
        stride_t stride_out{static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};
        auto *output = reinterpret_cast<std::complex<float> *>(output_interleaved);

        r2c(shape_in, stride_in, stride_out, 0, true, input, output, 1.0f, 1);
        return 0;
    } catch (...) {
        return 1;
    }
}
