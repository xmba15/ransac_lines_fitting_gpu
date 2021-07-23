/**
 * @file    LineSegment.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace perception
{
template <typename POINT_CLOUD_TYPE> class LineSegment
{
 public:
    using PointCloudType = POINT_CLOUD_TYPE;
    using PointCloud = pcl::PointCloud<PointCloudType>;
    using PointCloudPtr = typename PointCloud::Ptr;

    LineSegment(const PointCloudPtr& inliers, const pcl::ModelCoefficients::Ptr& coeffs)
        : m_inliers(inliers)
        , m_coeffs(coeffs)
    {
        if (!m_coeffs) {
            throw std::runtime_error("null line model coefficients");
        }

        auto calcSignedDistance = [](const PointCloudType& point, const pcl::ModelCoefficients& coeffs) {
            return point.getVector3fMap().dot(
                Eigen::Map<Eigen::Vector3f>(const_cast<float*>(coeffs.values.data() + 3), 3));
        };
        std::sort(m_inliers->points.begin(), m_inliers->points.end(),
                  [this, &calcSignedDistance](const auto& e1, const auto& e2) {
                      return calcSignedDistance(e1, *m_coeffs) < calcSignedDistance(e2, *m_coeffs);
                  });

        m_projected = projectPoints(m_inliers, coeffs, pcl::SACMODEL_LINE);
    }

    static PointCloudPtr projectPoints(const PointCloudPtr& inCloud, const pcl::ModelCoefficients::Ptr& coeffs,
                                       const int modelType)
    {
        PointCloudPtr projected(new PointCloud);
        pcl::ProjectInliers<POINT_CLOUD_TYPE> projector;
        projector.setModelType(modelType);
        projector.setInputCloud(inCloud);
        projector.setModelCoefficients(coeffs);
        projector.filter(*projected);

        return projected;
    }

    const auto& inliers() const
    {
        return m_inliers;
    }

    const auto& projected() const
    {
        return m_projected;
    }

 private:
    PointCloudPtr m_inliers;
    pcl::ModelCoefficients::Ptr m_coeffs;
    PointCloudPtr m_projected;
};
}  // namespace perception
