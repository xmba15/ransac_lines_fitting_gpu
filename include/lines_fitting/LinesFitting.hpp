/**
 * @file    LinesFitting.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "LineSegment.hpp"

namespace perception
{
template <typename POINT_CLOUD_TYPE, std::size_t DIMENSION> class LinesFitting
{
 public:
    struct Param {
        int minNumPoints = 20;
        float epsilonDist = 0.05;  // [meter]
        float p = 0.99;
        int maxIterations = 10000;
    };

    using PointCloudType = POINT_CLOUD_TYPE;
    using PointCloud = pcl::PointCloud<PointCloudType>;
    using PointCloudPtr = typename PointCloud::Ptr;
    using LineSegment = perception::LineSegment<PointCloudType>;

    explicit LinesFitting(const Param& param);

    ~LinesFitting();

    std::vector<LineSegment> run(const PointCloudPtr& inCloud);

 private:
    std::vector<std::vector<float>> toPointsVector(const PointCloudPtr& inCloud) const;

 private:
    Param m_param;
};
}  // namespace perception
