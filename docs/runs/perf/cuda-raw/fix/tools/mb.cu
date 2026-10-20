// Microbenchmark of pocket phase-1 / phase-2 variants (scratch, not for the repo).
#include <cuda_bf16.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstdlib>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("err %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

// --- reference phase 1 per warp (old kernel's loop), one warp per neuron, a block per row (NB warps)
template <int NB>
__global__ void p1_plain(const __nv_bfloat16* __restrict__ y, int d, const float* __restrict__ up, const int* __restrict__ used, float* __restrict__ act) {
    const int lane = threadIdx.x & 31, j = blockIdx.x * NB + (threadIdx.x >> 5);
    if (j >= *used) return;
    const __nv_bfloat16* yr = y + (size_t)blockIdx.y * d;
    const float* uj = up + (size_t)j * d;
    float s = 0.f;
    for (int i = lane; i < d; i += 32) s += uj[i] * __bfloat162float(yr[i]);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) { float z = fmaxf(s, 0.f); act[blockIdx.y * 64 + j] = z * z; }
}

// register batches of B per lane
template <int NB, int B>
__global__ void p1_batch(const __nv_bfloat16* __restrict__ y, int d, const float* __restrict__ up, const int* __restrict__ used, float* __restrict__ act) {
    const int lane = threadIdx.x & 31, j = blockIdx.x * NB + (threadIdx.x >> 5);
    if (j >= *used) return;
    const __nv_bfloat16* yr = y + (size_t)blockIdx.y * d;
    const float* uj = up + (size_t)j * d;
    float s = 0.f;
    for (int i0 = lane; i0 < d; i0 += 32 * B) {
        float u[B], v[B];
#pragma unroll
        for (int b = 0; b < B; ++b) { const int i = i0 + 32 * b; u[b] = i < d ? uj[i] : 0.f; v[b] = i < d ? __bfloat162float(yr[i]) : 0.f; }
#pragma unroll
        for (int b = 0; b < B; ++b) if (i0 + 32 * b < d) s += u[b] * v[b];
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) { float z = fmaxf(s, 0.f); act[blockIdx.y * 64 + j] = z * z; }
}

// smem staged
template <int NB, int R, int C>
__global__ void p1_smem(const __nv_bfloat16* __restrict__ y, int d, const float* __restrict__ up, const int* __restrict__ used, float* __restrict__ act, int rows) {
    __shared__ float us[NB][32 * C];
    __shared__ __nv_bfloat16 ys[R][32 * C];
    const int n = *used;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, j0 = blockIdx.x * NB;
    if (j0 >= n) return;
    const int r0 = blockIdx.y * R, tile = min(R, rows - r0);
    float s[R];
#pragma unroll
    for (int t = 0; t < R; ++t) s[t] = 0.f;
    for (int i0 = 0; i0 < d; i0 += 32 * C) {
#pragma unroll
        for (int m = 0; m < C; ++m) { const int e = threadIdx.x + m * NB * 32, q = e / (32 * C), i = i0 + e % (32 * C); if (j0 + q < n && i < d) us[q][e % (32 * C)] = up[(size_t)(j0 + q) * d + i]; }
#pragma unroll
        for (int m = 0; m < R * C / NB; ++m) { const int e = threadIdx.x + m * NB * 32, t = e / (32 * C), i = i0 + e % (32 * C); if (t < tile && i < d) ys[t][e % (32 * C)] = y[(size_t)(r0 + t) * d + i]; }
        __syncthreads();
        if (j0 + w < n) {
#pragma unroll
            for (int c = 0; c < C; ++c) if (i0 + 32 * c + lane < d) { const float u = us[w][32 * c + lane];
#pragma unroll
                for (int t = 0; t < R; ++t) if (t < tile) s[t] += u * __bfloat162float(ys[t][32 * c + lane]); }
        }
        __syncthreads();
    }
    if (j0 + w >= n) return;
    for (int t = 0; t < R; ++t) if (t < tile) { float v = s[t];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) { float z = fmaxf(v, 0.f); act[(r0 + t) * 64 + j0 + w] = z * z; } }
}

// phase 2: a row per thread (grid.y = rows), batches of B
template <int B>
__global__ void p2_row(const float* __restrict__ act, const float* __restrict__ down, const int* __restrict__ used, int d, float* __restrict__ out) {
    __shared__ float a[64];
    const int n = *used;
    if (threadIdx.x < 64) a[threadIdx.x] = threadIdx.x < n ? act[blockIdx.y * 64 + threadIdx.x] : 0.f;
    __syncthreads();
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= d) return;
    float s = 0.f;
    for (int j0 = 0; j0 < n; j0 += B) {
        float w[B];
#pragma unroll
        for (int b = 0; b < B; ++b) w[b] = j0 + b < n ? down[(size_t)(j0 + b) * d + k] : 0.f;
#pragma unroll
        for (int b = 0; b < B; ++b) if (j0 + b < n) s += a[j0 + b] * w[b];
    }
    out[((size_t)blockIdx.y * 8 + 6) * d + k] += s;
}

// phase 2: 16 rows a thread, t outer per batch
template <int B>
__global__ void p2_tile(const float* __restrict__ act, const float* __restrict__ down, const int* __restrict__ used, int d, float* __restrict__ out, int rows) {
    __shared__ float a[16 * 64];
    const int n = *used, r0 = blockIdx.y * 16, tile = min(16, rows - r0);
    for (int i = threadIdx.x; i < tile * 64; i += blockDim.x) if (i % 64 < n) a[i] = act[(size_t)r0 * 64 + i];
    __syncthreads();
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= d) return;
    float s[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) s[t] = 0.f;
    for (int j0 = 0; j0 < n; j0 += B) {
        float w[B];
#pragma unroll
        for (int b = 0; b < B; ++b) w[b] = j0 + b < n ? down[(size_t)(j0 + b) * d + k] : 0.f;
#pragma unroll
        for (int t = 0; t < 16; ++t) if (t < tile) {
#pragma unroll
            for (int b = 0; b < B; ++b) if (j0 + b < n) s[t] += a[t * 64 + j0 + b] * w[b]; }
    }
    for (int t = 0; t < 16; ++t) if (t < tile) out[((size_t)(r0 + t) * 8 + 6) * d + k] += s[t];
}

// a warp NPW neurons x R rows straight from global (L1 shares a block's rows)
template <int NPW, int R>
__global__ void p1_reg(const __nv_bfloat16* __restrict__ y, int d, const float* __restrict__ up, const int* __restrict__ used, float* __restrict__ act, int rows) {
    const int n = *used;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int jw = (blockIdx.x * (blockDim.x >> 5) + w) * NPW;
    if (jw >= n) return;
    const int r0 = blockIdx.y * R, tile = min(R, rows - r0);
    float s[NPW][R];
#pragma unroll
    for (int q = 0; q < NPW; ++q)
#pragma unroll
        for (int t = 0; t < R; ++t) s[q][t] = 0.f;
    for (int i = lane; i < d; i += 32) {
        float u[NPW], v[R];
#pragma unroll
        for (int q = 0; q < NPW; ++q) u[q] = jw + q < n ? up[(size_t)(jw + q) * d + i] : 0.f;
#pragma unroll
        for (int t = 0; t < R; ++t) v[t] = t < tile ? __bfloat162float(y[(size_t)(r0 + t) * d + i]) : 0.f;
#pragma unroll
        for (int q = 0; q < NPW; ++q)
#pragma unroll
            for (int t = 0; t < R; ++t) s[q][t] += u[q] * v[t];
    }
#pragma unroll
    for (int q = 0; q < NPW; ++q)
#pragma unroll
        for (int t = 0; t < R; ++t) {
            float v = s[q][t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && jw + q < n && t < tile) { float z = fmaxf(v, 0.f); act[(size_t)(r0 + t) * 64 + jw + q] = z * z; }
        }
}

// 16 rows a thread over KC columns a thread
template <int B, int KC>
__global__ void p2_tile2(const float* __restrict__ act, const float* __restrict__ down, const int* __restrict__ used, int d, float* __restrict__ out, int rows) {
    __shared__ float a[16 * 64];
    const int n = *used, r0 = blockIdx.y * 16, tile = min(16, rows - r0);
    for (int i = threadIdx.x; i < tile * 64; i += blockDim.x) if (i % 64 < n) a[i] = act[(size_t)r0 * 64 + i];
    __syncthreads();
    const int k0 = blockIdx.x * blockDim.x * KC + threadIdx.x;
    float s[KC][16];
#pragma unroll
    for (int c = 0; c < KC; ++c)
#pragma unroll
        for (int t = 0; t < 16; ++t) s[c][t] = 0.f;
    for (int j0 = 0; j0 < n; j0 += B) {
        float w[KC][B];
#pragma unroll
        for (int c = 0; c < KC; ++c)
#pragma unroll
            for (int b = 0; b < B; ++b) { const int k = k0 + c * blockDim.x; w[c][b] = j0 + b < n && k < d ? down[(size_t)(j0 + b) * d + k] : 0.f; }
#pragma unroll
        for (int t = 0; t < 16; ++t) if (t < tile) {
#pragma unroll
            for (int b = 0; b < B; ++b) if (j0 + b < n) { const float x = a[t * 64 + j0 + b];
#pragma unroll
                for (int c = 0; c < KC; ++c) s[c][t] += x * w[c][b]; } }
    }
#pragma unroll
    for (int c = 0; c < KC; ++c) { const int k = k0 + c * blockDim.x; if (k < d)
        for (int t = 0; t < 16; ++t) if (t < tile) out[((size_t)(r0 + t) * 8 + 6) * d + k] += s[c][t]; }
}

int main(int argc, char** argv) {
    const int d = 2688, rows = argc > 1 ? atoi(argv[1]) : 1, n = argc > 2 ? atoi(argv[2]) : 44;
    __nv_bfloat16 *y; float *up, *down, *act, *out; int* used; char* flush;
    CK(cudaMalloc(&y, (size_t)2048 * d * 2)); CK(cudaMalloc(&up, 64 * d * 4)); CK(cudaMalloc(&down, 64 * d * 4));
    CK(cudaMalloc(&act, 2048 * 64 * 4)); CK(cudaMalloc(&out, (size_t)2048 * 8 * d * 4)); CK(cudaMalloc(&used, 4)); CK(cudaMalloc(&flush, 128 << 20));
    std::vector<float> h(64 * d); for (auto& v : h) v = (rand() / (float)RAND_MAX - 0.5f) * 0.03f;
    CK(cudaMemcpy(up, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(down, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(used, &n, 4, cudaMemcpyHostToDevice)); CK(cudaMemset(y, 0x3f, (size_t)2048 * d * 2)); CK(cudaMemset(out, 0, (size_t)2048 * 8 * d * 4));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto time = [&](const char* name, auto launch) {
        for (int cold = 1; cold >= 0; --cold) {
            std::vector<float> ts;
            for (int r = 0; r < 42; ++r) {
                if (cold) cudaMemsetAsync(flush, r, 128 << 20);
                cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1); if (r >= 2) ts.push_back(ms * 1000);
            }
            CK(cudaGetLastError());
            std::sort(ts.begin(), ts.end());
            printf("%-28s rows=%4d n=%2d %s %7.1f us\n", name, rows, n, cold ? "cold" : "warm", ts[ts.size() / 2]);
        }
    };
    const int R16 = (rows + 15) / 16;
    if (argc > 3) {
    time("p1_plain NB=4", [&] { p1_plain<4><<<dim3(16, rows), 128>>>(y, d, up, used, act); });
    time("p1_reg 8,8 b256", [&] { p1_reg<8, 8><<<dim3(1, (rows + 7) / 8), 256>>>(y, d, up, used, act, rows); });
    time("p1_reg 4,8 b128", [&] { p1_reg<4, 8><<<dim3(4, (rows + 7) / 8), 128>>>(y, d, up, used, act, rows); });
    time("p1_reg 2,4 b128", [&] { p1_reg<2, 4><<<dim3(8, (rows + 3) / 4), 128>>>(y, d, up, used, act, rows); });
    time("p2_row B=16 t128", [&] { p2_row<16><<<dim3(21, rows), 128>>>(act, down, used, d, out); });
    time("p2_tile B=8 t128", [&] { p2_tile<8><<<dim3(21, R16), 128>>>(act, down, used, d, out, rows); });
    return 0;
    }
    if (rows > 16) {
    time("p1_smem 8,16,8", [&] { p1_smem<8, 16, 8><<<dim3(8, R16), 256>>>(y, d, up, used, act, rows); });
    time("p1_reg 4,8 b256", [&] { p1_reg<4, 8><<<dim3(2, (rows + 7) / 8), 256>>>(y, d, up, used, act, rows); });
    time("p1_reg 2,16 b256", [&] { p1_reg<2, 16><<<dim3(4, R16), 256>>>(y, d, up, used, act, rows); });
    time("p1_reg 4,16 b256", [&] { p1_reg<4, 16><<<dim3(2, R16), 256>>>(y, d, up, used, act, rows); });
    time("p1_reg 2,8 b256", [&] { p1_reg<2, 8><<<dim3(4, (rows + 7) / 8), 256>>>(y, d, up, used, act, rows); });
    time("p1_reg 4,8 b128", [&] { p1_reg<4, 8><<<dim3(4, (rows + 7) / 8), 128>>>(y, d, up, used, act, rows); });
    time("p1_reg 8,8 b256", [&] { p1_reg<8, 8><<<dim3(1, (rows + 7) / 8), 256>>>(y, d, up, used, act, rows); });
    time("p2_tile B=8 t128", [&] { p2_tile<8><<<dim3(21, R16), 128>>>(act, down, used, d, out, rows); });
    time("p2_tile2 B=8 KC=2 t128", [&] { p2_tile2<8, 2><<<dim3(11, R16), 128>>>(act, down, used, d, out, rows); });
    time("p2_tile2 B=8 KC=4 t64", [&] { p2_tile2<8, 4><<<dim3(11, R16), 64>>>(act, down, used, d, out, rows); });
    time("p2_tile2 B=4 KC=4 t64", [&] { p2_tile2<4, 4><<<dim3(11, R16), 64>>>(act, down, used, d, out, rows); });
    time("p2_tile2 B=4 KC=2 t128", [&] { p2_tile2<4, 2><<<dim3(11, R16), 128>>>(act, down, used, d, out, rows); });
    return 0;
    }
    time("p1_plain NB=1", [&] { p1_plain<1><<<dim3(64, rows), 32>>>(y, d, up, used, act); });
    time("p1_plain NB=4", [&] { p1_plain<4><<<dim3(16, rows), 128>>>(y, d, up, used, act); });
    time("p1_batch NB=1 B=12", [&] { p1_batch<1, 12><<<dim3(64, rows), 32>>>(y, d, up, used, act); });
    time("p1_batch NB=1 B=28", [&] { p1_batch<1, 28><<<dim3(64, rows), 32>>>(y, d, up, used, act); });
    time("p1_batch NB=2 B=28", [&] { p1_batch<2, 28><<<dim3(32, rows), 64>>>(y, d, up, used, act); });
    time("p1_batch NB=4 B=28", [&] { p1_batch<4, 28><<<dim3(16, rows), 128>>>(y, d, up, used, act); });
    time("p1_batch NB=1 B=42", [&] { p1_batch<1, 42><<<dim3(64, rows), 32>>>(y, d, up, used, act); });
    time("p1_smem 4,1,32", [&] { p1_smem<4, 1, 32><<<dim3(16, rows), 128>>>(y, d, up, used, act, rows); });
    time("p1_smem 1,1,32", [&] { p1_smem<1, 1, 32><<<dim3(64, rows), 32>>>(y, d, up, used, act, rows); });
    time("p1_smem 2,1,32", [&] { p1_smem<2, 1, 32><<<dim3(32, rows), 64>>>(y, d, up, used, act, rows); });
    time("p1_smem 8,16,8", [&] { p1_smem<8, 16, 8><<<dim3(8, R16), 256>>>(y, d, up, used, act, rows); });
    time("p1_smem 4,16,8", [&] { p1_smem<4, 16, 8><<<dim3(16, R16), 128>>>(y, d, up, used, act, rows); });
    time("p1_smem 8,16,16", [&] { p1_smem<8, 16, 16><<<dim3(8, R16), 256>>>(y, d, up, used, act, rows); });
    time("p2_row B=16 t128", [&] { p2_row<16><<<dim3(21, rows), 128>>>(act, down, used, d, out); });
    time("p2_row B=16 t64", [&] { p2_row<16><<<dim3(42, rows), 64>>>(act, down, used, d, out); });
    time("p2_row B=32 t64", [&] { p2_row<32><<<dim3(42, rows), 64>>>(act, down, used, d, out); });
    time("p2_row B=64 t64", [&] { p2_row<64><<<dim3(42, rows), 64>>>(act, down, used, d, out); });
    time("p2_row B=8 t64", [&] { p2_row<8><<<dim3(42, rows), 64>>>(act, down, used, d, out); });
    time("p2_tile B=16 t128", [&] { p2_tile<16><<<dim3(21, R16), 128>>>(act, down, used, d, out, rows); });
    time("p2_tile B=8 t128", [&] { p2_tile<8><<<dim3(21, R16), 128>>>(act, down, used, d, out, rows); });
    time("p2_tile B=16 t64", [&] { p2_tile<16><<<dim3(42, R16), 64>>>(act, down, used, d, out, rows); });
    return 0;
}
