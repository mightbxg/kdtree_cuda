#include "kdtree.hpp"
#include "kdtree_cuda.h"
#include "kdtree_ref.h"
#include "utils.hpp"
#include <benchmark/benchmark.h>

namespace {

    utils::Sampler sampler({0, 500, 0, 800});
    auto samples = sampler.get_samples(2000);

    void BM_create_kdtree_ref(benchmark::State &state) {
        for (auto _: state) {
            ref::KDTree_<float> kdt(samples);
        }
    }

    BENCHMARK(BM_create_kdtree_ref);

    void BM_create_kdtree(benchmark::State &state) {
        for (auto _: state) {
            cpu::KDTree kdt(samples);
        }
    }

    BENCHMARK(BM_create_kdtree);

    void BM_create_kdtree_cuda(benchmark::State &state) {
        for (auto _: state) {
            cu::KDTree kdt(samples);
        }
    }

    BENCHMARK(BM_create_kdtree_cuda);

} // anonymous namespace

BENCHMARK_MAIN();