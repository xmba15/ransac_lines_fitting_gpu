/**
 * @file    LinesFittingInterface.hpp
 *
 * @author  btran
 *
 */

#include <vector>

namespace perception
{
namespace cuda
{
struct LineCoefficients {
    std::vector<float> p;
    std::vector<float> d;
};

struct LineSegment {
    LineCoefficients coeffs;
    std::vector<int> inlierIndices;
};

template <std::size_t DIMENSION>
std::vector<LineSegment> run(const std::vector<std::vector<float>>& pointsV, const int minNumPoints,
                             const float epsilonDist, const float p, const int maxIterations)
{
    return {};
}

template <>
std::vector<LineSegment> run<2>(const std::vector<std::vector<float>>& pointsV, const int minNumPoints,
                                const float epsilonDist, const float p, const int maxIterations);

template <>
std::vector<LineSegment> run<3>(const std::vector<std::vector<float>>& pointsV, const int minNumPoints,
                                const float epsilonDist, const float p, const int maxIterations);
}  // namespace cuda
}  // namespace perception
