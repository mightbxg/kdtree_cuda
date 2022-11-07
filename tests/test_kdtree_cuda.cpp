#include "kdtree_cuda.h"
#include "kdtree_ref.h"
#include "utils.hpp"
#include <gtest/gtest.h>

TEST(kdtree_cuda, nnsearch) {
    constexpr size_t kTestNum = 500;
    constexpr size_t kQueryNum = 500;
    constexpr size_t kSampleNum = 2000;
    const Eigen::Vector4f ranges{0, 500, 0, 800};
    utils::Sampler sampler(ranges);

    for (size_t test_idx = 0; test_idx < kTestNum; ++test_idx) {
        auto seed = std::random_device()();
        auto samples = sampler.get_samples(kSampleNum, seed);
        ref::KDTree_<float> kdt_ref(samples);
        cu::KDTree kdt(samples);

        // get queries and ref results
        std::vector<cu::Point> queries;
        std::vector<int> indices_ref;
        std::vector<float> dists_ref;
        queries.reserve(kQueryNum);
        indices_ref.reserve(kQueryNum);
        dists_ref.reserve(kQueryNum);
        for (size_t query_idx = 0; query_idx < kQueryNum; ++query_idx) {
            auto query = sampler.get_point();
            queries.emplace_back(query.x(), query.y());
            float dist;
            auto idx = kdt_ref.nnSearch(query, &dist);
            indices_ref.push_back(idx);
            dists_ref.push_back(dist);
        }

        // test kdtree
        std::vector<int> indices;
        std::vector<float> dists;
        kdt.NNSearchBatch(queries, indices, dists);

        // check results
        for (size_t i = 0; i < kQueryNum; ++i) {
            const auto &q = queries[i];
            ASSERT_EQ(indices_ref[i], indices[i]) << "query[" << q[0] << ", " << q[1] << "] seed[" << seed
                                                  << "]\n ref[" << samples[indices_ref[i]] << " | " << dists_ref[i]
                                                  << "]\n out[" << samples[indices[i]] << " | " << dists[i];
            ASSERT_FLOAT_EQ(dists_ref[i], dists[i]) << "query[" << q[0] << ", " << q[1] << "] seed[" << seed
                                                    << "]\n ref[" << samples[indices_ref[i]] << " | " << dists_ref[i]
                                                    << "]\n out[" << samples[indices[i]] << " | " << dists[i];
        }
    }
}