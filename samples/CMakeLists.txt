find_package(OpenCV REQUIRED)

add_executable(vis_nnsearch vis_nnsearch.cpp)
target_link_libraries(vis_nnsearch ${OpenCV_LIBS} common_depends kdtree_cuda)
