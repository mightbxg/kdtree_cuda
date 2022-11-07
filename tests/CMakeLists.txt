find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(test_kdtree test_kdtree.cpp)
target_link_libraries(test_kdtree GTest::gtest GTest::gtest_main common_depends)
gtest_discover_tests(test_kdtree)

add_executable(test_kdtree_cuda test_kdtree_cuda.cpp)
target_link_libraries(test_kdtree_cuda GTest::gtest GTest::gtest_main common_depends kdtree_cuda)
gtest_discover_tests(test_kdtree_cuda)
