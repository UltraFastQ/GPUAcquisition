#ifndef __GPUINFO_HPP__
#define __GPUINFO_HPP__

namespace gpu {
    /* Displays information about all CUDA devices plugged into the system, as well as information about the CUDA API. Throws std::runtime_error in case of failure. */
    void display_cuda_info();

    /* Displays information about the CUDA API. Throws std::runtime_error in case of failure. */
    void display_cuda_api_version();

    /* Displays information about a CUDA device. Throws std::runtime_error in case of failure. */
    void display_gpu_info(int device);
}

#endif /* __GPUINFO_H__ */
