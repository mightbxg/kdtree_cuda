#include "kdtree.hpp"
#include "kdtree_ref.h"
#include "utils.hpp"
#include <gtest/gtest.h>

TEST(kdtree, nnsearch) {
    constexpr size_t kTestNum = 500;
    constexpr size_t kQueryNum = 500;
    constexpr size_t kSampleNum = 2000;
    const Eigen::Vector4f ranges{0, 500, 0, 800};
    utils::Sampler sampler(ranges);

    for (size_t test_idx = 0; test_idx < kTestNum; ++test_idx) {
        auto seed = std::random_device()();
        auto samples = sampler.get_samples(kSampleNum, seed);
        ref::KDTree_<float> kdt_ref(samples);
        cpu::KDTree kdt(samples);

        for (size_t query_idx = 0; query_idx < kQueryNum; ++query_idx) {
            auto query = sampler.get_point();
            float dist = utils::kFloatInf, dist_ref = utils::kFloatInf;
            auto idx_ref = kdt_ref.nnSearch(query, &dist_ref);
            auto idx = kdt.NNSearch({query.x(), query.y()}, &dist);
            ASSERT_EQ(idx_ref, idx) << "query[" << query << "] seed[" << seed
                                    << "]\n  ref[" << samples[idx_ref] << " | " << dist_ref
                                    << "]\n  out[" << samples[idx] << " | " << dist << "]";
        }
    }
}