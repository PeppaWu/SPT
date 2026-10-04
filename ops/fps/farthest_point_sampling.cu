#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

// Use squared XYZ distances and lowest-index tie breaking. Each
// block handles one cloud, with all sampling rounds inside the same kernel.
__global__ void farthest_point_sampling_kernel(
        const float* xyz, float* distance, const int64_t* initial,
        int64_t* centroids, int N, int npoint) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float best_distance[256];
    __shared__ int best_index[256];
    int farthest = initial[b];

    for (int round = 0; round < npoint; ++round) {
        if (tid == 0) {
            centroids[b * npoint + round] = farthest;
        }
        float max_dist = -1e10;
        int max_idx = 0;
        for (int n = tid; n < N; n += blockDim.x) {
            float d = distance[b * N + n];
            if (n == farthest || d == -1e10) {
                d = -1e10;
            } else {
                float dist = 0.0;
                // The caller supplies contiguous packed XYZ coordinates.
                for (int i = 0; i < 3; ++i) {
                    float diff = xyz[b * N * 3 + n * 3 + i]
                               - xyz[b * N * 3 + farthest * 3 + i];
                    dist += diff * diff;
                }
                d = fminf(d, dist);
            }
            distance[b * N + n] = d;
            if (d > max_dist) {
                max_dist = d;
                max_idx = n;
            }
        }
        best_distance[tid] = max_dist;
        best_index[tid] = max_idx;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (tid < offset) {
                const float other = best_distance[tid + offset];
                const int other_idx = best_index[tid + offset];
                if (other > best_distance[tid] ||
                    (other == best_distance[tid] && other_idx < best_index[tid])) {
                    best_distance[tid] = other;
                    best_index[tid] = other_idx;
                }
            }
            __syncthreads();
        }
        farthest = best_index[0];
        // All threads must read the winner before the next round overwrites it.
        __syncthreads();
    }
}

std::vector<torch::Tensor> farthest_point_sample_cuda(torch::Tensor xyz, int npoint) {
    const c10::cuda::CUDAGuard device_guard(xyz.device());
    const int B = xyz.size(0);
    const int N = xyz.size(1);
    auto centroids = torch::zeros({B, npoint}, torch::dtype(torch::kInt64).device(xyz.device()));
    auto distance = torch::full({B, N}, 1e10, torch::dtype(torch::kFloat32).device(xyz.device()));
    // Sample one initial point per cloud.
    auto farthest = torch::randint(0, N, {B}, torch::dtype(torch::kInt64).device(xyz.device()));
    if (B > 0 && npoint > 0) {
        farthest_point_sampling_kernel<<<B, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            xyz.data_ptr<float>(), distance.data_ptr<float>(),
            farthest.data_ptr<int64_t>(), centroids.data_ptr<int64_t>(), N, npoint);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {centroids};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("farthest_point_sample_cuda", &farthest_point_sample_cuda, "Farthest Point Sampling CUDA");
}
