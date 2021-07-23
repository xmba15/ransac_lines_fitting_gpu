/**
 * @file    LinesFitting.cu
 *
 * @author  btran
 *
 */

#include <algorithm>
#include <vector>

#include <pcl/common/io.h>

#include <lines_fitting/LinesFitting.hpp>

#include "cuda/LinesFittingInterface.hpp"

namespace perception
{
template <typename POINT_CLOUD_TYPE, std::size_t DIMENSION>
LinesFitting<POINT_CLOUD_TYPE, DIMENSION>::LinesFitting(const Param& param)
    : m_param(param)
{
}

template <typename POINT_CLOUD_TYPE, std::size_t DIMENSION> LinesFitting<POINT_CLOUD_TYPE, DIMENSION>::~LinesFitting()
{
}

template <typename POINT_CLOUD_TYPE, std::size_t DIMENSION>
std::vector<std::vector<float>>
LinesFitting<POINT_CLOUD_TYPE, DIMENSION>::toPointsVector(const PointCloudPtr& inCloud) const
{
    std::vector<std::vector<float>> pointsV;
    pointsV.reserve(inCloud->size());

    if (DIMENSION == 2) {
        std::transform(inCloud->points.begin(), inCloud->points.end(), std::back_inserter(pointsV),
                       [](const auto& point) -> std::vector<float> {
                           return {point.x, point.y};
                       });
    } else if (DIMENSION == 3) {
        std::transform(inCloud->points.begin(), inCloud->points.end(), std::back_inserter(pointsV),
                       [](const auto& point) -> std::vector<float> {
                           return {point.x, point.y, point.z};
                       });
    }

    return pointsV;
}

template <typename POINT_CLOUD_TYPE, std::size_t DIMENSION>
std::vector<typename LinesFitting<POINT_CLOUD_TYPE, DIMENSION>::LineSegment>
LinesFitting<POINT_CLOUD_TYPE, DIMENSION>::run(const PointCloudPtr& inCloud)
{
    const auto pointsV = this->toPointsVector(inCloud);
    auto cudaLineSegments = perception::cuda::run<DIMENSION>(pointsV, m_param.minNumPoints, m_param.epsilonDist,
                                                             m_param.p, m_param.maxIterations);

    std::vector<LineSegment> result;
    result.reserve(cudaLineSegments.size());

    for (auto& cudaLineSegment : cudaLineSegments) {
        PointCloudPtr inliers(new PointCloud);
        pcl::copyPointCloud(*inCloud, cudaLineSegment.inlierIndices, *inliers);
        pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
        coeffs->values.reserve(6);
        std::copy(cudaLineSegment.coeffs.p.begin(), cudaLineSegment.coeffs.p.end(), std::back_inserter(coeffs->values));
        std::copy(cudaLineSegment.coeffs.d.begin(), cudaLineSegment.coeffs.d.end(), std::back_inserter(coeffs->values));
        result.emplace_back(inliers, coeffs);
    }

    return result;
}

#undef INSTANTIATE_TEMPLATE
#undef INSTANTIATE_TEMPLATE_DIMENSION
#define INSTANTIATE_TEMPLATE(DATA_TYPE, DIMENSION) template class LinesFitting<DATA_TYPE, DIMENSION>;
#define INSTANTIATE_TEMPLATE_DIMENSION(DIMENSION)                                                                      \
    INSTANTIATE_TEMPLATE(pcl::PointXYZ, DIMENSION);                                                                    \
    INSTANTIATE_TEMPLATE(pcl::PointXYZI, DIMENSION);                                                                   \
    INSTANTIATE_TEMPLATE(pcl::PointXYZINormal, DIMENSION);                                                             \
    INSTANTIATE_TEMPLATE(pcl::PointXYZRGB, DIMENSION);                                                                 \
    INSTANTIATE_TEMPLATE(pcl::PointXYZRGBNormal, DIMENSION);

INSTANTIATE_TEMPLATE_DIMENSION(2);
INSTANTIATE_TEMPLATE_DIMENSION(3);

#undef INSTANTIATE_TEMPLATE_DIMENSION
#undef INSTANTIATE_TEMPLATE
}  // namespace perception
