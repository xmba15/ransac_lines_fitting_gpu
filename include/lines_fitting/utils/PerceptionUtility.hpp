/**
 * @file    PerceptionUtility.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

#include "../LineSegment.hpp"
#include <pcl/common/common.h>
#include <pcl/point_types.h>

namespace perception
{
namespace utils
{
template <typename PointCloudType>
inline cv::Mat draw2DImage(const typename pcl::PointCloud<PointCloudType>::ConstPtr& inCloud, const float xStep = 0.008,
                           const float yStep = 0.008, const int defaultWidth = 500, const int defaultHeight = 500)
{
    cv::Scalar bgColor = cv::Scalar(167, 167, 167);
    if (inCloud->size() < 2) {
        return cv::Mat(defaultHeight, defaultWidth, CV_8UC3, bgColor);
    }

    PointCloudType minP, maxP;
    pcl::getMinMax3D<PointCloudType>(*inCloud, minP, maxP);

    int height = (maxP.y - minP.y) / yStep;
    int width = (maxP.x - minP.x) / xStep;

    cv::Mat image(height, width, CV_8UC3, bgColor);
    for (const auto& point : inCloud->points) {
        int y = (point.y - minP.y) / yStep;
        int x = (point.x - minP.x) / xStep;
        image.ptr<cv::Vec3b>(y)[x] = cv::Vec3b(0, 0, 255);
        cv::circle(image, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }

    return image;
}

template <typename PointCloudType>
inline cv::Mat draw2DLineSegmentImage(const typename pcl::PointCloud<PointCloudType>::ConstPtr& inCloud,
                                      const std::vector<perception::LineSegment<PointCloudType>>& lineSegments,
                                      const float xStep = 0.008, const float yStep = 0.008,
                                      const int defaultWidth = 500, const int defaultHeight = 500)
{
    cv::Mat image = draw2DImage<PointCloudType>(inCloud, xStep, yStep, defaultWidth, defaultHeight);
    if (inCloud->size() < 2) {
        return image;
    }

    cv::Mat noLineImage = image.clone();

    PointCloudType minP, maxP;
    pcl::getMinMax3D<PointCloudType>(*inCloud, minP, maxP);

    cv::Scalar lineColor = cv::Scalar(255, 0, 0);
    for (const auto& lineSegment : lineSegments) {
        const PointCloudType& start = lineSegment.projected()->points.front();
        const PointCloudType& end = lineSegment.projected()->points.back();

        int startY = (start.y - minP.y) / yStep;
        int startX = (start.x - minP.x) / xStep;

        int endY = (end.y - minP.y) / yStep;
        int endX = (end.x - minP.x) / xStep;

        cv::line(image, cv::Point(startX, startY), cv::Point(endX, endY), lineColor, 1);
    }

    int height = image.rows;
    int width = image.cols;
    int stepSize = 50;
    cv::Scalar gridColor(100, 100, 100);
    for (int i = stepSize; i < height; i += stepSize) {
        cv::line(image, cv::Point(0, i), cv::Point(width, i), gridColor);
    }

    for (int i = stepSize; i < width; i += stepSize) {
        cv::line(image, cv::Point(i, 0), cv::Point(i, height), gridColor);
    }

    cv::hconcat(noLineImage, image, image);

    return image;
}
}  // namespace utils
}  // namespace perception
