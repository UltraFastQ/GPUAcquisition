#include "Utils.hpp"
#include "GPUInfo.hpp"

void gpu::display_cuda_info() {
    gpu::display_cuda_api_version();

    int num_devices = 0;
    auto rc = cudaGetDeviceCount(&num_devices);
    utils::cuda_err_handle(rc, "cudaGetDeviceCount failed", __FILE__, __LINE__);
    
    if (num_devices == 0) {
        utils::log_warning("No CUDA enabled devices found", __FILE__, __LINE__);
    }
    else {
        utils::log_message(std::string("Number of CUDA enabled devices found: ") + std::to_string(num_devices));

        // Display information for each device
        for (int device = 0; device < num_devices; ++device) {
            gpu::display_gpu_info(device);
        }
    }
}

void gpu::display_cuda_api_version() {
    int cuda_version_support = 0;
    auto rc = cudaDriverGetVersion(&cuda_version_support);
    utils::cuda_err_handle(rc, "cudaDriverGetVersion failed", __FILE__, __LINE__);
    utils::log_message(std::string("Latest CUDA supported by driver: ") + std::to_string(cuda_version_support/1000) + "." + std::to_string((cuda_version_support % 100)/10));

    int cuda_rt_version = 0;
    rc = cudaRuntimeGetVersion(&cuda_rt_version);
    utils::cuda_err_handle(rc, "cudaRuntimeGetVersion failed", __FILE__, __LINE__);
    utils::log_message(std::string("CUDA RT version: ") + std::to_string(cuda_rt_version/1000) + "." + std::to_string((cuda_rt_version % 100)/10));
}

void gpu::display_gpu_info(int device) {
    auto rc = cudaSetDevice(device);
    utils::cuda_err_handle(rc, "cudaSetDevice failed", __FILE__, __LINE__);

    cudaDeviceProp props;
    rc = cudaGetDeviceProperties(&props, device);
    utils::cuda_err_handle(rc, "cudaGetDeviceProperties failed", __FILE__, __LINE__);

    utils::log_message(std::string("Device ") + std::to_string(device) + ": \"" + props.name + "\"");
    utils::log_message(std::string("Compute capability: ") + std::to_string(props.major) + "." + std::to_string(props.minor));
    utils::log_message(std::string("Total global memory: ") + std::to_string(props.totalGlobalMem / (1 << 20)) + " MiB");
    utils::log_message(std::string("Number of SMs: ") + std::to_string(props.multiProcessorCount));
    utils::log_message(std::string("Max size of grid (x, y, z): (") + std::to_string(props.maxGridSize[0]) + ", " + std::to_string(props.maxGridSize[1]) + ", " + std::to_string(props.maxGridSize[2]) + ")");
    utils::log_message(std::string("Max size of a thread block (x, y, z): (") + std::to_string(props.maxThreadsDim[0]) + ", " + std::to_string(props.maxThreadsDim[1]) + ", " + std::to_string(props.maxThreadsDim[2]) + ")");
    utils::log_message(std::string("Max threads per block: ") + std::to_string(props.maxThreadsPerBlock));
}
