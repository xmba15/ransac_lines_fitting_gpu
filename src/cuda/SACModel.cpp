/**
 * @file    SACModel.cpp
 *
 * @author  btran
 *
 */

#include "SACModel.hpp"
#include "BasicUtility.hpp"
#include "PointTypes.hpp"

namespace perception
{
namespace cuda
{
template <typename POINT_CLOUD_TYPE> struct SACModel<POINT_CLOUD_TYPE>::HypothesisCreator {
    const PointCloudType* input;
    const int* indices;
    int numIndices;
    mutable thrust::minstd_rand rng;

    __host__ __device__ HypothesisCreator(const PointCloudType* input, const int* indices, int numIndices,
                                          const thrust::minstd_rand& rng)
        : input(input)
        , indices(indices)
        , numIndices(numIndices)
        , rng(rng)
    {
    }

    __host__ __device__ Coefficients operator()(const int n)
    {
        Coefficients coeffs;

        int2 samples;
        float trand = numIndices / (RAND_MAX + 1.0f);
        rng.discard(n);
        while (utils::almostEquals<float>(coeffs.d.norm(), 0)) {
            samples.x = indices[(int)(rng() * trand)];
            samples.y = indices[(int)(rng() * trand)];
            coeffs.d = (input[indices[samples.y]] - input[indices[samples.x]]).normalized();
        }
        coeffs.p = input[indices[samples.x]];

        return coeffs;
    }
};

template <typename POINT_CLOUD_TYPE> struct SACModel<POINT_CLOUD_TYPE>::InlierIndicesGatherer {
    const PointCloudType* input;
    const Coefficients* coeffs;
    float epsilonDist;

    __host__ __device__ InlierIndicesGatherer(const PointCloudType* input, const Coefficients* hypotheses,
                                              const int hypothesisIdx, const float epsilonDist)
        : input(input)
        , coeffs(&hypotheses[hypothesisIdx])
        , epsilonDist(epsilonDist)
    {
    }

    template <typename Tuple> __host__ __device__ void operator()(Tuple& tuple)
    {
        thrust::get<1>(tuple) = input[thrust::get<0>(tuple)].distanceToLine(coeffs->p, coeffs->d) < epsilonDist;
    }
};

template <typename POINT_CLOUD_TYPE>
void SACModel<POINT_CLOUD_TYPE>::setInputCloud(const std::vector<PointCloudType>& points)
{
    this->clear();
    m_points.resize(points.size());
    thrust::copy(points.begin(), points.end(), m_points.begin());
}

template <typename POINT_CLOUD_TYPE> bool SACModel<POINT_CLOUD_TYPE>::computeModel()
{
    m_coeffsV.clear();
    m_inliersIndicesV.clear();

    Indices targetIndices(m_points.size());
    thrust::sequence(targetIndices.begin(), targetIndices.end(), 0);

    while (true) {
        if (!computeModelUtils(targetIndices)) {
            break;
        }
    }

    return !m_coeffsV.empty();
}

template <typename POINT_CLOUD_TYPE> bool SACModel<POINT_CLOUD_TYPE>::computeModelUtils(Indices& targetIndices)
{
    if (targetIndices.size() < m_minNumPoints) {
        return false;
    }

    Hypotheses h;
    this->generateModelHypotheses(m_points, targetIndices, h, m_maxIterations);

    int iteration = 0;
    int k = 1;
    int bestNumInliers = -1;
    int bestIteration = -1;

    thrust::device_vector<bool> bestIsInliers(targetIndices.size());
    thrust::device_vector<bool> isInliers(targetIndices.size());

    while (iteration < std::min(k, m_maxIterations)) {
        int numInliers = this->selectWithinDistance(m_points, targetIndices, h, iteration, isInliers, m_epsilonDist);

        if (numInliers > bestNumInliers) {
            bestNumInliers = numInliers;
            bestIteration = iteration;
            bestIsInliers = isInliers;

            float w = static_cast<float>(bestNumInliers) / m_points.size();
            float pNumOutliers = 1.0f - std::pow(w, 2);
            k = std::log(1. - m_p) / std::log(pNumOutliers);
        }

        iteration++;
    }

    if (bestIteration == -1 || bestNumInliers < m_minNumPoints) {
        return false;
    }

    Indices inliers;
    utils::split_remove<int, thrust::device_vector>(targetIndices, bestIsInliers, inliers);
    m_coeffsV.push_back(h[bestIteration]);
    m_inliersIndicesV.emplace_back(std::move(inliers));

    return true;
}

template <typename POINT_CLOUD_TYPE>
void SACModel<POINT_CLOUD_TYPE>::downloadToHost(std::vector<Coefficients>& hostCoeffsV,
                                                std::vector<std::vector<int>>& hostInlierIndicesV) const
{
    if (m_coeffsV.empty()) {
        return;
    }
    hostCoeffsV.resize(m_coeffsV.size());
    thrust::copy(m_coeffsV.begin(), m_coeffsV.end(), hostCoeffsV.begin());
    hostInlierIndicesV.resize(m_coeffsV.size());

    for (std::size_t i = 0; i < m_coeffsV.size(); ++i) {
        auto& hostInlierIndices = hostInlierIndicesV[i];
        auto& deviceInlierIndices = m_inliersIndicesV[i];
        hostInlierIndices.resize(deviceInlierIndices.size());
        thrust::copy(deviceInlierIndices.begin(), deviceInlierIndices.end(), hostInlierIndices.begin());
    }
}

template <typename POINT_CLOUD_TYPE>
bool SACModel<POINT_CLOUD_TYPE>::generateModelHypotheses(const PointCloud& points, const Indices& indices,
                                                         Hypotheses& hypotheses, const int maxIterations) const
{
    hypotheses.resize(maxIterations);
    thrust::device_vector<int> randoms(maxIterations);

    thrust::counting_iterator<int> idxSequence(0);
    thrust::transform(idxSequence, idxSequence + maxIterations, randoms.begin(),
                      utils::ParallelRandomGenerator(m_seed));

    thrust::transform(randoms.begin(), randoms.end(), hypotheses.begin(),
                      HypothesisCreator(thrust::raw_pointer_cast(points.data()),
                                        thrust::raw_pointer_cast(indices.data()), indices.size(), m_rng));

    return true;
}

template <typename POINT_CLOUD_TYPE>
int SACModel<POINT_CLOUD_TYPE>::selectWithinDistance(const PointCloud& points, const Indices& targetIndices,
                                                     const Hypotheses& hypotheses, const int hypothesisIdx,
                                                     thrust::device_vector<bool>& isInliers,
                                                     const float epsilonDist) const
{
    isInliers.resize(targetIndices.size(), false);

    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(targetIndices.begin(), isInliers.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(targetIndices.end(), isInliers.end())),
                     InlierIndicesGatherer(thrust::raw_pointer_cast(points.data()),
                                           thrust::raw_pointer_cast(hypotheses.data()), hypothesisIdx, epsilonDist));

    int numInliers = thrust::count(isInliers.begin(), isInliers.end(), true);

    return numInliers;
}

#undef INSTANTIATE_TEMPLATE
#define INSTANTIATE_TEMPLATE(DATA_TYPE) template class SACModel<DATA_TYPE>;
INSTANTIATE_TEMPLATE(PointXY);
INSTANTIATE_TEMPLATE(PointXYZ);

#undef INSTANTIATE_TEMPLATE
}  // namespace cuda
}  // namespace perception
