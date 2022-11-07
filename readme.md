# KDtree with CUDA

2-dimensional KDtree with cuda, only NN (Nearest Neighbour) search is implemented. The tree nodes structure is built on CPU and transferred to GPU before used.

## Benchmark
On laptop with Intel Core i7-9750H and NVIDIA GeForce GTX 1650:
```bash
# build kdtree with 2k sample points
----------------------------------------------------------------
Benchmark                      Time             CPU   Iterations
----------------------------------------------------------------
BM_create_kdtree_ref      259041 ns       259010 ns         2696
BM_create_kdtree          222636 ns       222629 ns         3126
BM_create_kdtree_cuda     228376 ns       228338 ns         3067

# NN search with 2k sample points and 45k queries
-----------------------------------------------------------
Benchmark                 Time             CPU   Iterations
-----------------------------------------------------------
BM_nnsearch_ref     9337326 ns      9336433 ns           72
BM_nnsearch         8687874 ns      8687220 ns           79
BM_nnsearch_cuda     375813 ns       375790 ns         1678
```

# Build
- Samples need [OpenCV](https://github.com/opencv/opencv)
- Tests need [Google Test](https://github.com/google/googletest)
- Benchmarks need [Google Benchmark](https://github.com/google/benchmark)

# References
- [kd-tree](https://github.com/gishi523/kd-tree) (also the ref in tests and benchmarks)
- [KD-TREE ON GPU FOR 3D POINT SEARCHING](http://nghiaho.com/?p=437)