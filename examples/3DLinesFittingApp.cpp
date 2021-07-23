/**
 * @file    2DLinesFittingApp.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <lines_fitting/lines_fitting.hpp>

namespace
{
using PointCloudType = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointCloudType>;
using PointCloudPtr = PointCloud::Ptr;
using LinesFitting = perception::LinesFitting<PointCloudType, 3>;
using LineSegment = perception::LineSegment<PointCloudType>;

inline pcl::visualization::PCLVisualizer::Ptr initializeViewer()
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    pcl::PointXYZ o(0.1, 0, 0);
    viewer->addSphere(o, 0.1, "sphere", 0);
    viewer->setBackgroundColor(0.05, 0.05, 0.05, 0);
    viewer->addCoordinateSystem(0.5);
    viewer->setCameraPosition(-26, 0, 3, 10, -1, 0.5, 0, 0, 1);

    return viewer;
}

inline void addLine(const pcl::visualization::PCLVisualizer::Ptr& viewer, const LineSegment& line,
                    const std::string& lineLabel, const std::array<std::uint8_t, 3>& color = {255, 0, 255})
{
    const auto& projected = line.projected();
    const PointCloudType& closest = projected->points.front();
    const PointCloudType& farthest = projected->points.back();

    viewer->addLine(closest, farthest, color[0] / 255., color[1] / 255., color[2] / 255., lineLabel);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, lineLabel);

    const std::string cloudLabel = "cloud" + lineLabel;
    viewer->addPointCloud<PointCloudType>(line.inliers(), cloudLabel);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color[0] / 255., color[1] / 255.,
                                             color[2] / 255., cloudLabel);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, cloudLabel);
};

auto viewer = initializeViewer();
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

    LinesFitting::Param param = {.minNumPoints = 30, .epsilonDist = 0.05, .p = 0.99, .maxIterations = 10000};
    LinesFitting linesFitter(param);

    timer.start();
    auto lineSegments = linesFitter.run(inCloud);
    std::cout << "\nprocessing time: " << timer.getMs() << "[ms]\n";

    viewer->addPointCloud<PointCloudType>(inCloud, "original_cloud");

    std::cout << "number of detected lines: " << lineSegments.size() << "\n";

    for (std::size_t i = 0; i < lineSegments.size(); ++i) {
        const auto& ls = lineSegments[i];
        std::string lb = "ls" + std::to_string(i);
        ::addLine(viewer, ls, lb);
    }

    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }

    return EXIT_SUCCESS;
}
