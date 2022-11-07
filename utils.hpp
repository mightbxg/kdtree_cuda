#pragma once

#include <Eigen/Core>
#include <iostream>
#include <random>
#include <vector>

namespace utils {

    inline static auto kFloatInf = std::numeric_limits<float>::infinity();

    inline auto rand_seed() {
        return std::random_device()();
    }

    class Sampler {
    public:
        using Point = Eigen::Vector2f;
        using Ranges = Eigen::Vector4f;

        explicit Sampler(const Ranges &_ranges, size_t _seed = rand_seed())
                : ranges(_ranges), seed(_seed), gen_(_seed), dist_x_(ranges[0], ranges[1]),
                  dist_y_(ranges[2], ranges[3]) {
        }

        [[nodiscard]] auto get_samples(size_t num, size_t _seed = rand_seed()) const {
            using namespace std;
            mt19937 gen(_seed);
            uniform_real_distribution<float> dist_x(ranges[0], ranges[1]);
            uniform_real_distribution<float> dist_y(ranges[2], ranges[3]);
            vector<Point> ret;
            ret.reserve(num);
            for (size_t i = 0; i < num; ++i) {
                ret.emplace_back(dist_x(gen), dist_y(gen));
            }
            return ret;
        }

        [[nodiscard]] auto get_point() {
            return Point{dist_x_(gen_), dist_y_(gen_)};
        }

    public:
        const Ranges ranges;
        const size_t seed;

    private:
        std::mt19937 gen_;
        std::uniform_real_distribution<float> dist_x_;
        std::uniform_real_distribution<float> dist_y_;
    };

} // namespace utils

inline std::ostream &operator<<(std::ostream &os, const Eigen::Vector2f &pt) {
    os << pt.x() << ", " << pt.y();
    return os;
}
