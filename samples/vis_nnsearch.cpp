#include "kdtree.hpp"
#include "kdtree_cuda.h"
#include "kdtree_ref.h"
#include "utils.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

static inline auto to_cv(const Eigen::Vector2f &pt) {
    return cv::Point2f(pt.x(), pt.y());
}

int main(int argc, char **argv) {
    int rows = 500, cols = 800;
    Eigen::Vector4f ranges = {0.0, float(cols), 0.0, float(rows)};
    auto seed = std::random_device()();
    utils::Sampler sampler(ranges);
    auto samples = sampler.get_samples(1000, seed);
    Eigen::Vector2f query = sampler.get_point();
    cout << "seed[" << seed << "] query[" << query << "]\n";

    // ref
    ref::KDTree_<float> kdt_ref(samples);
    float dist_ref = 0.0f;
    auto idx_ref = kdt_ref.nnSearch(query, &dist_ref);
    const auto &pt_ref = samples[idx_ref];
    cout << " ref: idx[" << idx_ref << "] pt[" << pt_ref << "] dist[" << dist_ref << "]\n";

    // cpu kdtree
    cpu::KDTree kdt(samples);
    float dist = 0.0f;
    auto idx = kdt.NNSearch({query.x(), query.y()}, &dist);
    const auto &pt = samples[idx];
    cout << " kdt: idx[" << idx << "] pt[" << pt << "] dist[" << dist << "]\n";

    // cuda kdtree
    cu::KDTree kdt_cuda;
    kdt_cuda.Build(samples);
    int idx_cu;
    float dist_cu;
    {
        std::vector<cu::Point> queries_cu = {{query[0], query[1]}};
        std::vector<int> indices_cu;
        std::vector<float> dists_cu;
        kdt_cuda.NNSearchBatch(queries_cu, indices_cu, dists_cu);
        idx_cu = indices_cu.front();
        dist_cu = dists_cu.front();
    }
    const auto &pt_cu = samples[idx_cu];
    cout << "cuda: idx[" << idx_cu << "] pt[" << pt_cu << "] dist[" << dist_cu << "]\n";

    cv::Mat image = cv::Mat::zeros(rows, cols, CV_8UC3);
    for (const auto &pt: samples) {
        cv::circle(image, to_cv(pt), 1, {255, 255, 255}, cv::FILLED);
    }
    cv::circle(image, to_cv(query), 2, {0, 0, 255}, cv::FILLED);
    cv::circle(image, to_cv(pt_ref), 2, {0, 255, 0}, cv::FILLED);
    cv::circle(image, to_cv(pt), 2, {255, 0, 0}, cv::FILLED);
    cv::imshow("pts", image);
    cv::waitKey(0);
}