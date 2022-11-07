/**
 * @brief Simple k-d tree implementation from
 * https://github.com/gishi523/kd-tree
 */

#pragma once

#include <algorithm>
#include <exception>
#include <functional>
#include <numeric>
#include <vector>

#include <Eigen/Core>

namespace ref {

///@brief k-d tree class.
    template<class Scalar_ = double>
    class KDTree_ {
    public:
        using Scalar = Scalar_;
        using Point = Eigen::Matrix<Scalar, 2, 1>;

        ///@brief The constructors.
        KDTree_()
                : root_(nullptr) {};

        explicit KDTree_(const std::vector <Point> &points)
                : root_(nullptr) {
            build(points);
        }

        ///@brief The destructor.
        ~KDTree_() { clear(); }

        ///@brief Re-builds k-d tree.
        void build(const std::vector <Point> &points) {
            clear();

            points_ = points;

            std::vector<int> indices(points.size());
            std::iota(std::begin(indices), std::end(indices), 0);

            root_ = buildRecursive(indices.data(), (int) points.size(), 0);
        }

        ///@brief Clears k-d tree.
        void clear() {
            clearRecursive(root_);
            root_ = nullptr;
            points_.clear();
        }

        ///@brief Validates k-d tree.
        [[nodiscard]] bool validate() const {
            try {
                validateRecursive(root_, 0);
            } catch (const Exception &) {
                return false;
            }

            return true;
        }

        ///@brief Searches the nearest neighbor.
        int nnSearch(const Point &query, Scalar *minDist = nullptr) const {
            int guess;
            auto _minDist = std::numeric_limits<Scalar>::max();

            nnSearchRecursive(query, root_, &guess, &_minDist);

            if (minDist)
                *minDist = _minDist;

            return guess;
        }

        ///@brief Searches k-nearest neighbors.
        std::vector<int> knnSearch(const Point &query, int k) const {
            KnnQueue queue(k);
            knnSearchRecursive(query, root_, queue, k);

            std::vector<int> indices(queue.size());
            for (size_t i = 0; i < queue.size(); i++)
                indices[i] = queue[i].second;

            return indices;
        }

        ///@brief Searches neighbors within radius.
        std::vector<int> radiusSearch(const Point &query, Scalar radius) const {
            std::vector<int> indices;
            radiusSearchRecursive(query, root_, indices, radius);
            return indices;
        }

    private:
        ///@brief k-d tree node.
        struct Node {
            int idx; //!< index to the original point
            Node *next[2]; //!< pointers to the child nodes
            int axis; //!< dimension's axis

            Node()
                    : idx(-1), axis(-1) {
                next[0] = next[1] = nullptr;
            }
        };

        ///@brief k-d tree exception.
        class Exception : public std::exception {
            using std::exception::exception;
        };

        ///@brief Bounded priority queue.
        template<class T, class Compare = std::less <T>>
        class BoundedPriorityQueue {
        public:
            BoundedPriorityQueue() = delete;

            explicit BoundedPriorityQueue(size_t bound)
                    : bound_(bound) {
                elements_.reserve(bound + 1);
            };

            void push(const T &val) {
                auto it = std::find_if(
                        std::begin(elements_), std::end(elements_),
                        [&](const T &element) { return Compare()(val, element); });
                elements_.insert(it, val);

                if (elements_.size() > bound_)
                    elements_.resize(bound_);
            }

            const T &back() const { return elements_.back(); };

            const T &operator[](size_t index) const { return elements_[index]; }

            [[nodiscard]] size_t size() const { return elements_.size(); }

        private:
            size_t bound_;
            std::vector <T> elements_;
        };

        ///@brief Priority queue of <distance, index> pair.
        using KnnQueue = BoundedPriorityQueue<std::pair < Scalar, int>>;

        ///@brief Builds k-d tree recursively.
        Node *buildRecursive(int *indices, int npoints, int depth) {
            if (npoints <= 0)
                return nullptr;

            const int axis = depth % Point::RowsAtCompileTime;
            const int mid = (npoints - 1) / 2;

            std::nth_element(indices, indices + mid, indices + npoints,
                             [&](int lhs, int rhs) {
                                 return points_[lhs][axis] < points_[rhs][axis];
                             });

            Node *node = new Node();
            node->idx = indices[mid];
            node->axis = axis;

            node->next[0] = buildRecursive(indices, mid, depth + 1);
            node->next[1] = buildRecursive(indices + mid + 1, npoints - mid - 1, depth + 1);

            return node;
        }

        ///@brief Clears k-d tree recursively.
        void clearRecursive(Node *node) {
            if (node == nullptr)
                return;

            if (node->next[0])
                clearRecursive(node->next[0]);

            if (node->next[1])
                clearRecursive(node->next[1]);

            delete node;
        }

        ///@brief Validates k-d tree recursively.
        void validateRecursive(const Node *node, int depth) const {
            if (node == nullptr)
                return;

            const int axis = node->axis;
            const Node *node0 = node->next[0];
            const Node *node1 = node->next[1];

            if (node0 && node1) {
                if (points_[node->idx][axis] < points_[node0->idx][axis])
                    throw Exception();

                if (points_[node->idx][axis] > points_[node1->idx][axis])
                    throw Exception();
            }

            if (node0)
                validateRecursive(node0, depth + 1);

            if (node1)
                validateRecursive(node1, depth + 1);
        }

        static Scalar distance(const Point &p, const Point &q) {
            return (p - q).norm();
        }

        ///@brief Searches the nearest neighbor recursively.
        void nnSearchRecursive(const Point &query, const Node *node, int *guess,
                               Scalar *minDist) const {
            if (node == nullptr)
                return;

            const Point &train = points_[node->idx];

            const Scalar dist = distance(query, train);
            if (dist < *minDist) {
                *minDist = dist;
                *guess = node->idx;
            }

            const int axis = node->axis;
            const int dir = query[axis] < train[axis] ? 0 : 1;
            nnSearchRecursive(query, node->next[dir], guess, minDist);

            const Scalar diff = fabs(query[axis] - train[axis]);
            if (diff < *minDist)
                nnSearchRecursive(query, node->next[!dir], guess, minDist);
        }

        ///@brief Searches k-nearest neighbors recursively.
        void knnSearchRecursive(const Point &query, const Node *node, KnnQueue &queue,
                                int k) const {
            if (node == nullptr)
                return;

            const Point &train = points_[node->idx];

            const Scalar dist = distance(query, train);
            queue.push(std::make_pair(dist, node->idx));

            const int axis = node->axis;
            const int dir = query[axis] < train[axis] ? 0 : 1;
            knnSearchRecursive(query, node->next[dir], queue, k);

            const Scalar diff = fabs(query[axis] - train[axis]);
            if ((int) queue.size() < k || diff < queue.back().first)
                knnSearchRecursive(query, node->next[!dir], queue, k);
        }

        ///@brief Searches neighbors within radius.
        void radiusSearchRecursive(const Point &query, const Node *node,
                                   std::vector<int> &indices, Scalar radius) const {
            if (node == nullptr)
                return;

            const Point &train = points_[node->idx];

            const Scalar dist = distance(query, train);
            if (dist < radius)
                indices.push_back(node->idx);

            const int axis = node->axis;
            const int dir = query[axis] < train[axis] ? 0 : 1;
            radiusSearchRecursive(query, node->next[dir], indices, radius);

            const Scalar diff = fabs(query[axis] - train[axis]);
            if (diff < radius)
                radiusSearchRecursive(query, node->next[!dir], indices, radius);
        }

        Node *root_; //!< root node
        std::vector <Point> points_; //!< points
    };

} // namespace ref