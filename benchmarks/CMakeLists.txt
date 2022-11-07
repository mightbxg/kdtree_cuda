find_package(benchmark REQUIRED)

add_executable(bm_create_kdtree bm_create_kdtree.cpp)
target_link_libraries(bm_create_kdtree benchmark::benchmark common_depends kdtree_cuda)

add_executable(bm_nnsearch bm_nnsearch.cpp)
target_link_libraries(bm_nnsearch benchmark::benchmark common_depends kdtree_cuda)
