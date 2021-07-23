/**
 * @file    SACModel.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <memory>

#include <thrust/device_vector.h>
#include <thrust/random/linear_congruential_engine.h>

#include "BasicUtility.hpp"

namespace perception
{
namespace cuda
{
template <typename POINT_CLOUD_TYPE> class SACModel
{
 public:
    using PointCloudType = POINT_CLOUD_TYPE;
    using PointCloud = thrust::device_vector<PointCloudType>;

    using Indices = thrust::device_vector<int>;
    using IndicesPtr = std::shared_ptr<Indices>;

    struct Coefficients {
        __host__ __device__ Coefficients(const PointCloudType& p = PointCloudType(),
                                         const PointCloudType& d = PointCloudType())
            : p(p)
            , d(d)
        {
        }

        PointCloudType p;
        PointCloudType d;
    };

    using Hypotheses = thrust::device_vector<Coefficients>;

    struct HypothesisCreator;
    struct InlierIndicesGatherer;

    SACModel(const int minNumPoints, const float epsilonDist, const float p, const int maxIterations,
             const int seed = 2021)
        : m_minNumPoints(minNumPoints)
        , m_epsilonDist(epsilonDist)
        , m_p(p)
        , m_maxIterations(maxIterations)
        , m_rng(seed)
        , m_seed(seed)
    {
    }

    void setInputCloud(const std::vector<PointCloudType>& points);

    bool computeModel();

    void downloadToHost(std::vector<Coefficients>& hostCoeffsV,
                        std::vector<std::vector<int>>& hostInlierIndicesV) const;

 private:
    bool computeModelUtils(Indices& targetIndices);

    bool generateModelHypotheses(const PointCloud& points, const Indices& indices, Hypotheses& hypotheses,
                                 const int maxIterations) const;

    int selectWithinDistance(const PointCloud& points, const Indices& targetIndices, const Hypotheses& hypotheses,
                             const int hypothesisIdx, thrust::device_vector<bool>& isInliers,
                             const float epsilonDist) const;

    void clear()
    {
        m_points.clear();
        m_coeffsV.clear();
        m_inliersIndicesV.clear();
    }

 private:
    int m_minNumPoints;
    float m_epsilonDist;
    float m_p;
    int m_maxIterations;
    mutable thrust::minstd_rand m_rng;
    int m_seed;

    PointCloud m_points;
    thrust::device_vector<Coefficients> m_coeffsV;
    std::vector<Indices> m_inliersIndicesV;
};
}  // namespace cuda
}  // namespace perception
