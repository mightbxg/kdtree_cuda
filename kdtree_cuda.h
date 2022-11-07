#pragma once

#include <vector>

namespace cu {

    struct Point {
    public:
        Point(float x, float y)
                : coord{x, y} {
        }

        auto &operator[](int idx) {
            return coord[idx];
        }

        auto &operator[](int idx) const {
            return coord[idx];
        }

    public:
        float coord[2];
    };

    template<typename Pt>
    inline void Copy(const std::vector <Pt> &in, std::vector <Point> &out) {
        out.clear();
        out.reserve(in.size());
        for (const auto &pt: in) {
            out.emplace_back(pt.x(), pt.y());
        }
    }

    template<>
    inline void Copy(const std::vector <Point> &in, std::vector <Point> &out) {
        out = in;
    }

    class KDTree {
    public:
        KDTree() = default;

        template<typename Pt>
        explicit KDTree(const std::vector <Pt> &pts) {
            Build(pts);
        }

        ///@brief Build kdtree from points.
        void Build(const std::vector <Point> &pts);

        template<typename Pt>
        void Build(const std::vector <Pt> &pts) {
            std::vector <Point> _pts;
            Copy(pts, _pts);
            Build(_pts);
        }

        ///@brief Clear all nodes.
        void Clear();

        ///@brief Search the nearest neighbor.
        void NNSearchBatch(const std::vector <Point> &queries, std::vector<int> &indices,
                           std::vector<float> &min_dists) const;

    public:
        struct Node {
            int pid{-1}; //!< index to the original point
            int axis{-1}; //!< dimension's axis
            int next[2]{-1, -1}; // index of child nodes
        };

    private:
        Node *nodes_{nullptr}; //!< nodes on gpu
        Point *pts_{nullptr}; //!< points on gpu
        std::size_t pt_num_{0};
    };

} // namespace cu