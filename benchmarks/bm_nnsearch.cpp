#include "kdtree.hpp"
#include "kdtree_cuda.h"
#include "kdtree_ref.h"
#include "utils.hpp"
#include <benchmark/benchmark.h>

namespace {

    utils::Sampler sampler({0, 500, 0, 800});
    auto samples = sampler.get_samples(2000);
    auto queries = [] {
        const size_t num = 45000;
        std::vector<Eigen::Vector2f> pts;
        pts.reserve(num);
        for (size_t i = 0; i < num; ++i) {
            pts.emplace_back(sampler.get_point());
        }
        return pts;
    }();

    void BM_nnsearch_ref(benchmark::State &state) {
        ref::KDTree_<float> kdt(samples);
        for (auto _: state) {
            float dist = 0.0f;
            for (const auto &query: queries) {
                auto idx = kdt.nnSearch(query, &dist);
            }
        }
    }

    BENCHMARK(BM_nnsearch_ref);

    void BM_nnsearch(benchmark::State &state) {
        cpu::KDTree kdt(samples);
        for (auto _: state) {
            float dist = 0.0f;
            for (const auto &query: queries) {
                auto idx = kdt.NNSearch({query.x(), query.y()}, &dist);
            }
        }
    }

    BENCHMARK(BM_nnsearch);

    void BM_nnsearch_cuda(benchmark::State &state) {
        cu::KDTree kdt(samples);
        std::vector<cu::Point> pts;
        cu::Copy(queries, pts);
        for (auto _: state) {
            std::vector<int> indices;
            std::vector<float> dists;
            kdt.NNSearchBatch(pts, indices, dists);
        }
    }

    BENCHMARK(BM_nnsearch_cuda);

} // anonymous namespace

BENCHMARK_MAIN();