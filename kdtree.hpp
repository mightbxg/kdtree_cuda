#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <stack>
#include <vector>

namespace cpu {

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

    private:
        float coord[2];
    };

    inline float SquaredDistance(const Point &pt1, const Point &pt2) {
        float dx = pt1[0] - pt2[0];
        float dy = pt1[1] - pt2[1];
        return dx * dx + dy * dy;
    }

    template<typename Pt>
    inline void Copy(const std::vector<Pt> &in, std::vector<Point> &out) {
        out.clear();
        out.reserve(in.size());
        for (const auto &pt: in) {
            out.emplace_back(pt.x(), pt.y());
        }
    }

    template<>
    inline void Copy(const std::vector<Point> &in, std::vector<Point> &out) {
        out = in;
    }

    class KDTree {
    public:
        KDTree() = default;

        template<typename Pt>
        explicit KDTree(const std::vector<Pt> &pts) {
            Build(pts);
        }

        ///@brief Build kdtree from points.
        template<typename Pt>
        void Build(const std::vector<Pt> &pts) {
            if (pts.empty()) {
                return;
            }
            Copy(pts, pts_);
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

            nodes_.clear();
            nodes_.reserve(pts.size());
            nodes_.emplace_back(); // the first node is root
            std::deque<BuildTask> tasks;
            tasks.emplace_back(nodes_[0], indices.data(), int(pts.size()), 0);
            while (!tasks.empty()) {
                auto crt_task = tasks.front();
                tasks.pop_front();
                const int pt_num = crt_task.pt_num;

                const int axis = crt_task.depth % 2;
                const int mid = pt_num / 2;
                int *ids = crt_task.indices;
                std::nth_element(ids, ids + mid, ids + pt_num, [&](int lhs, int rhs) {
                    return pts_[lhs][axis] < pts_[rhs][axis];
                });
                auto &crt_node = *crt_task.node;
                crt_node.pid = ids[mid];
                crt_node.axis = axis;
                int next_depth = crt_task.depth + 1;
                auto CreateChildNodeTask = [&](int *idx_start, int pt_num) {
                    nodes_.emplace_back();
                    int nid = static_cast<int>(nodes_.size() - 1);
                    tasks.emplace_back(nodes_[nid], idx_start, pt_num, next_depth);
                    return nid;
                };
                if (mid > 0) {
                    crt_node.next[0] = CreateChildNodeTask(ids, mid);
                }
                if (int r_num = pt_num - mid - 1; r_num > 0) {
                    crt_node.next[1] = CreateChildNodeTask(ids + mid + 1, r_num);
                }
            }
        }

        [[nodiscard]] bool IsPointIdValid(int id) const {
            return id >= 0 && id < pts_.size();
        }

        ///@brief Validates the kdtree.
        [[nodiscard]] bool Validate() const {
            if (nodes_.size() != pts_.size()) {
                return false;
            }
            if (nodes_.empty()) {
                return true;
            }
            std::deque<int> to_visit = {0};
            while (!to_visit.empty()) {
                auto nid = to_visit.front();
                to_visit.pop_front();
                const auto &node = nodes_[nid];
                if (!IsPointIdValid(node.pid)) {
                    return false;
                }
                const auto axis = node.axis;
                const int nid0 = node.next[0];
                const int nid1 = node.next[1];
                if (nid0 >= 0) {
                    const auto &node0 = nodes_[nid0];
                    if (pts_[node.pid][axis] < pts_[node0.pid][axis]) {
                        return false;
                    }
                    to_visit.push_back(nid0);
                }
                if (nid1 >= 0) {
                    const auto &node1 = nodes_[nid1];
                    if (pts_[node.pid][axis] > pts_[node1.pid][axis]) {
                        return false;
                    }
                    to_visit.push_back(nid1);
                }
            }
            return true;
        }

        ///@brief Search the nearest neighbor.
        [[nodiscard]] int NNSearch(const Point &query, float *min_dist = nullptr) const {
            if (nodes_.empty()) {
                return -1;
            }

            int guess;
            auto min_sd = std::numeric_limits<float>::max();
            struct CheckTask {
                CheckTask(float _diff_square, int _nid)
                        : diff_square(_diff_square), nid(_nid) {
                }

                float diff_square;
                int nid; //!< node index
            };
            std::stack<CheckTask> tasks;
            tasks.emplace(0.0f, 0);
            while (!tasks.empty()) {
                auto crt_task = tasks.top();
                tasks.pop();
                if (crt_task.diff_square >= min_sd) {
                    continue;
                }
                const auto &node = nodes_[crt_task.nid];
                const auto pid = node.pid;
                const auto &train = pts_[pid];
                float sd = SquaredDistance(query, train);
                if (sd < min_sd) {
                    guess = pid;
                    min_sd = sd;
                }

                const auto axis = node.axis;
                const int dir = query[axis] < train[axis] ? 0 : 1;
                auto diff = query[axis] - train[axis];
                auto PushNode = [&tasks](float diff_square, int nid) {
                    if (nid < 0) {
                        return;
                    }
                    tasks.emplace(diff_square, nid);
                };
                PushNode(diff * diff, node.next[!dir]);
                PushNode(0.0f, node.next[dir]);
            }

            if (min_dist) {
                *min_dist = std::sqrt(min_sd);
            }
            return guess;
        }

    private:
        struct Node {
            int pid{-1}; //!< index to the original point
            int axis{-1}; //!< dimension's axis
            int next[2]{-1, -1}; // index of child nodes
        };

    private:
        std::vector<Node> nodes_;
        std::vector<Point> pts_;
    };

} // namespace cpu