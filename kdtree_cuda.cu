#include "kdtree_cuda.h"
#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <deque>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace cu {

    static inline void CheckCudaError(const char *msg) {
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::stringstream ss;
            ss << "CUDA error[" << msg << "]: " << cudaGetErrorString(err) << "\n";
            throw std::runtime_error(ss.str());
        }
    }

    template<typename T>
    static void CudaMalloc(T *&ptr, size_t num) {
        cudaMalloc((void **) &ptr, sizeof(T) * num);
    }

    void KDTree::Clear() {
        cudaFree(nodes_);
        cudaFree(pts_);
        pt_num_ = 0;
    }

    static auto BuildKDtree(const std::vector <Point> &pts) {
        using Node = KDTree::Node;
        std::vector <Node> nodes;
        if (pts.empty()) {
            return nodes;
        }
        nodes.reserve(pts.size());
        std::vector<int> indices(pts.size());
        std::iota(indices.begin(), indices.end(), 0);

        struct BuildTask {
            BuildTask(Node &_node, int *_indices, int _pt_num, int _depth)
                    : node(&_node), indices(_indices), pt_num(_pt_num), depth(_depth) {
            }

            Node *node; //!< the node to be built
            int *indices; //!< the beginning pointer to the indices
            int pt_num; //!< num of points belong to the node
            int depth; //!< depth of the node
        };

        nodes.emplace_back(); // the first node is root
        std::deque <BuildTask> tasks;
        tasks.emplace_back(nodes[0], indices.data(), int(pts.size()), 0);
        while (!tasks.empty()) {
            auto crt_task = tasks.front();
            tasks.pop_front();
            const int pt_num = crt_task.pt_num;

            const int axis = crt_task.depth % 2;
            const int mid = pt_num / 2;
            int *ids = crt_task.indices;
            std::nth_element(ids, ids + mid, ids + pt_num, [&](int lhs, int rhs) {
                return pts[lhs][axis] < pts[rhs][axis];
            });
            auto &crt_node = *crt_task.node;
            crt_node.pid = ids[mid];
            crt_node.axis = axis;
            int next_depth = crt_task.depth + 1;
            auto CreateChildNodeTask = [&](int *idx_start, int pt_num) {
                nodes.emplace_back();
                int nid = static_cast<int>(nodes.size() - 1);
                tasks.emplace_back(nodes[nid], idx_start, pt_num, next_depth);
                return nid;
            };
            if (mid > 0) {
                crt_node.next[0] = CreateChildNodeTask(ids, mid);
            }
            int r_num = pt_num - mid - 1;
            if (r_num > 0) {
                crt_node.next[1] = CreateChildNodeTask(ids + mid + 1, r_num);
            }
        }
        return nodes;
    }

    void KDTree::Build(const std::vector <Point> &pts) {
        Clear();
        if (pts.empty()) {
            return;
        }
        const auto pt_num = static_cast<int>(pts.size());
        CudaMalloc(pts_, pt_num);
        CudaMalloc(nodes_, pt_num);
        CheckCudaError("KDTree::Build");
        cudaMemcpyAsync(pts_, pts.data(), sizeof(Point) * pt_num, cudaMemcpyHostToDevice);
        auto nodes = BuildKDtree(pts);
        cudaMemcpy(nodes_, nodes.data(), sizeof(Node) * pt_num, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        pt_num_ = pts.size();
        CheckCudaError("KDTree::Build");
    }

    __device__ float SquaredDistance(const Point &pt1, const Point &pt2) {
        float dx = pt1.coord[0] - pt2.coord[0];
        float dy = pt1.coord[1] - pt2.coord[1];
        return dx * dx + dy * dy;
    }

    struct CheckTask {
        CheckTask() = default;

        __device__ void Set(int _nid, float _diff_square) {
            nid = _nid;
            diff_square = _diff_square;
        }

        int nid{-1}; //!< node index
        float diff_square{0.0};
    };

    __device__ void
    Search(const KDTree::Node *nodes, const Point *pts, const Point &query, int &ret_idx, float &ret_dist) {
        int guess;
        float min_sd = 3e30;
        CheckTask tasks[15]; // 15 buf stack can hold about 30k nodes
        tasks[0].Set(0, 0.0f);
        int stack_size = 1;
        while (stack_size > 0) {
            // pop stack top
            const auto &crt_task = tasks[--stack_size];
            if (crt_task.diff_square >= min_sd) {
                continue;
            }
            const auto &node = nodes[crt_task.nid];
            const auto pid = node.pid;
            const auto &train = pts[pid];
            float sd = SquaredDistance(train, query);
            if (sd < min_sd) {
                guess = pid;
                min_sd = sd;
            }

            const auto axis = node.axis;
            const int dir = query.coord[axis] < train.coord[axis] ? 0 : 1;
            auto diff = query.coord[axis] - train.coord[axis];
            auto PushNode = [&](float diff_square, int nid) {
                if (nid < 0) {
                    return;
                }
                tasks[stack_size++].Set(nid, diff_square);
            };
            PushNode(diff * diff, node.next[!dir]);
            PushNode(0.0f, node.next[dir]);
        }

        ret_idx = guess;
        ret_dist = sqrt(min_sd);
    }

    __global__ void SearchBatch(const KDTree::Node *nodes, const Point *pts, const Point *queries, size_t query_num,
                                int *ret_indices, float *ret_dists) {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= query_num) {
            return;
        }
        Search(nodes, pts, queries[idx], ret_indices[idx], ret_dists[idx]);
    }

    void KDTree::NNSearchBatch(const std::vector <Point> &queries, std::vector<int> &indices,
                               std::vector<float> &min_dists) const {
        if (pt_num_ == 0 || queries.empty()) {
            return;
        }

        const size_t query_num = queries.size();
        const int thd_num = 512;
        const int blk_num = query_num / thd_num + ((query_num % thd_num) ? 1 : 0);

        // copy queries to device
        Point *gpu_queries;
        int *gpu_ret_indices;
        float *gpu_ret_dists;
        CudaMalloc(gpu_queries, query_num);
        CudaMalloc(gpu_ret_indices, query_num);
        CudaMalloc(gpu_ret_dists, query_num);
        CheckCudaError("KDTree::NNSearchBatch");
        cudaMemcpy(gpu_queries, queries.data(), sizeof(Point) * query_num, cudaMemcpyHostToDevice);
        CheckCudaError("KDTree::NNSearchBatch");

        // do NN search
        SearchBatch<<<blk_num, thd_num>>>(nodes_, pts_, gpu_queries, query_num,
                                          gpu_ret_indices, gpu_ret_dists);
        CheckCudaError("KDTree::NNSearchBatch");
        cudaDeviceSynchronize();

        // copy data back to host
        indices.resize(query_num);
        min_dists.resize(query_num);
        cudaMemcpyAsync(indices.data(), gpu_ret_indices, sizeof(int) * query_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(min_dists.data(), gpu_ret_dists, sizeof(float) * query_num, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaFree(gpu_queries);
        cudaFree(gpu_ret_indices);
        cudaFree(gpu_ret_dists);
    }

} // namespace cu