/**
 * @file    2DLinesFittingApp.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include <pcl/io/pcd_io.h>

#include <lines_fitting/lines_fitting.hpp>

namespace
{
using PointCloudType = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointCloudType>;
using PointCloudPtr = PointCloud::Ptr;
using LinesFitting = perception::LinesFitting<PointCloudType, 2>;

basic::utils::Timer timer;
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: [app] [path/to/pcd]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string pclFilePath = argv[1];
    PointCloudPtr inCloud(new PointCloud);
    if (pcl::io::loadPCDFile(pclFilePath, *inCloud) == -1) {
        std::cerr << "Failed to load pcl file" << std::endl;
        return EXIT_FAILURE;
    }

    LinesFitting::Param param = {.minNumPoints = 30, .epsilonDist = 0.006, .p = 0.99, .maxIterations = 1000};
    LinesFitting linesFitter(param);

    timer.start();
    auto lineSegments = linesFitter.run(inCloud);
    std::cout << "\nprocessing time: " << timer.getMs() << "[ms]\n";

    cv::Mat image = perception::utils::draw2DLineSegmentImage<PointCloudType>(inCloud, lineSegments);
    cv::imwrite("image.png", image);

    return EXIT_SUCCESS;
}
