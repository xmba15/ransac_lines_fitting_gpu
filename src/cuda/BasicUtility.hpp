/**
 * @file    BasicUtility.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <cuda.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random/linear_congruential_engine.h>

namespace perception
{
namespace cuda
{
namespace utils
{
/**
 *  @brief Ref: https://pytorch.org/docs/stable/generated/torch.isclose.html
 */
template <typename DataType>
inline __host__ __device__ bool almostEquals(const DataType input, const DataType other, DataType rtol = 1e-05,
                                             DataType atol = 1e-08)
{
    return std::abs(input - other) <= atol + rtol * std::abs(other);
}

struct ParallelRandomGenerator {
    __host__ __device__ ParallelRandomGenerator(const int seed)
        : seed(seed)
        , rng(thrust::minstd_rand(seed))
    {
    }

    __host__ __device__ int operator()(const int n) const
    {
        rng.discard(n);
        return rng();
    }

    int seed;
    mutable thrust::minstd_rand rng;
};

template <typename DataType, template <typename...> class Container>
inline void split(const Container<DataType>& v, const Container<bool>& isInliers,
                  Container<DataType>* inliers = nullptr, Container<DataType>* outliers = nullptr)
{
    if (!inliers && !outliers) {
        return;
    }
    int numInliers = thrust::count(isInliers.begin(), isInliers.end(), true);
    int numData = v.size();

    if (inliers) {
        inliers->resize(numInliers);
        Container<bool> inliersStencil(numInliers);

        thrust::copy_if(
            thrust::make_zip_iterator(thrust::make_tuple(v.begin(), isInliers.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(v.end(), isInliers.end())),
            thrust::make_zip_iterator(thrust::make_tuple(inliers->begin(), inliersStencil.begin())),
            [=] __host__ __device__(const thrust::tuple<int, bool>& tuple) { return thrust::get<1>(tuple); });
    }

    if (outliers) {
        outliers->resize(v.size() - numInliers);
        Container<bool> outliersStencil(v.size() - numInliers);

        thrust::copy_if(
            thrust::make_zip_iterator(thrust::make_tuple(v.begin(), isInliers.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(v.end(), isInliers.end())),
            thrust::make_zip_iterator(thrust::make_tuple(outliers->begin(), outliersStencil.begin())),
            [=] __host__ __device__(const thrust::tuple<int, bool>& tuple) { return !thrust::get<1>(tuple); });
    }
}

template <typename DataType, template <typename...> class Container>
inline void split_remove(Container<DataType>& v, Container<bool>& isInliers, Container<DataType>& inliers)

{
    int numInliers = thrust::count(isInliers.begin(), isInliers.end(), true);
    int numData = v.size();

    inliers.resize(numInliers);
    Container<bool> inliersStencil(numInliers);

    thrust::copy_if(thrust::make_zip_iterator(thrust::make_tuple(v.begin(), isInliers.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(v.end(), isInliers.end())),
                    thrust::make_zip_iterator(thrust::make_tuple(inliers.begin(), inliersStencil.begin())),
                    [=] __host__ __device__(const thrust::tuple<int, bool>& tuple) { return thrust::get<1>(tuple); });

    thrust::remove_if(thrust::make_zip_iterator(thrust::make_tuple(v.begin(), isInliers.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(v.end(), isInliers.end())),
                      [=] __host__ __device__(const thrust::tuple<int, bool>& tuple) { return thrust::get<1>(tuple); });

    v.resize(numData - numInliers);
    isInliers.resize(numData - numInliers);
}
}  // namespace utils
}  // namespace cuda
}  // namespace perception
