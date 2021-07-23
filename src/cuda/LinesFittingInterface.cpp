/**
 * @file    LinesFittingInterface.cpp
 *
 * @author  btran
 *
 */

#include <algorithm>

#include "LinesFittingInterface.hpp"
#include "PointTypes.hpp"
#include "SACModel.hpp"

namespace perception
{
namespace cuda
{
using SACModel2D = SACModel<PointXY>;
using SACModel3D = SACModel<PointXYZ>;

template <>
std::vector<perception::cuda::LineSegment> run<2>(const std::vector<std::vector<float>>& pointsV,
                                                  const int minNumPoints, const float epsilonDist, const float p,
                                                  const int maxIterations)
{
    std::vector<perception::cuda::PointXY> customPointsV;
    std::transform(pointsV.begin(), pointsV.end(), std::back_inserter(customPointsV),
                   [](const auto& point) { return perception::cuda::PointXY(point[0], point[1]); });

    cuda::SACModel2D model(minNumPoints, epsilonDist, p, maxIterations);
    model.setInputCloud(customPointsV);
    if (!model.computeModel()) {
        return {};
    }

    std::vector<cuda::SACModel2D::Coefficients> hostCoeffsV;
    std::vector<std::vector<int>> hostInlierIndicesV;
    model.downloadToHost(hostCoeffsV, hostInlierIndicesV);

    std::vector<perception::cuda::LineSegment> result;
    result.reserve(hostCoeffsV.size());
    for (std::size_t i = 0; i < hostCoeffsV.size(); ++i) {
        auto& hostCoeffs = hostCoeffsV[i];
        perception::cuda::LineSegment lineSegment;
        lineSegment.coeffs.p = {hostCoeffs.p.x, hostCoeffs.p.y, 0};
        lineSegment.coeffs.d = {hostCoeffs.d.x, hostCoeffs.d.y, 0};
        lineSegment.inlierIndices = std::move(hostInlierIndicesV[i]);
        result.emplace_back(std::move(lineSegment));
    }

    return result;
}

template <>
std::vector<perception::cuda::LineSegment> run<3>(const std::vector<std::vector<float>>& pointsV,
                                                  const int minNumPoints, const float epsilonDist, const float p,
                                                  const int maxIterations)
{
    std::vector<perception::cuda::PointXYZ> customPointsV;
    std::transform(pointsV.begin(), pointsV.end(), std::back_inserter(customPointsV),
                   [](const auto& point) { return perception::cuda::PointXYZ(point[0], point[1], point[2]); });

    cuda::SACModel3D model(minNumPoints, epsilonDist, p, maxIterations);
    model.setInputCloud(customPointsV);
    if (!model.computeModel()) {
        return {};
    }

    std::vector<cuda::SACModel3D::Coefficients> hostCoeffsV;
    std::vector<std::vector<int>> hostInlierIndicesV;
    model.downloadToHost(hostCoeffsV, hostInlierIndicesV);

    std::vector<perception::cuda::LineSegment> result;
    result.reserve(hostCoeffsV.size());
    for (std::size_t i = 0; i < hostCoeffsV.size(); ++i) {
        auto& hostCoeffs = hostCoeffsV[i];
        perception::cuda::LineSegment lineSegment;
        lineSegment.coeffs.p = {hostCoeffs.p.x, hostCoeffs.p.y, hostCoeffs.p.z};
        lineSegment.coeffs.d = {hostCoeffs.d.x, hostCoeffs.d.y, hostCoeffs.d.z};
        lineSegment.inlierIndices = std::move(hostInlierIndicesV[i]);
        result.emplace_back(std::move(lineSegment));
    }

    return result;
}
}  // namespace cuda
}  // namespace perception
