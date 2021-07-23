/**
 * @file    Types.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <cuda.h>
#include <thrust/device_vector.h>

#include "BasicUtility.hpp"

namespace perception
{
namespace cuda
{
struct PointXY {
    using DataType = float2;

    union {
        float2 xy;
        struct {
            float x;
            float y;
        };
    };

    __host__ __device__ PointXY(const float x = 0., const float y = 0.)
        : x(x)
        , y(y)
    {
    }

    __host__ __device__ DataType data() const
    {
        return xy;
    }

    __host__ __device__ PointXY& operator-=(const PointXY& rhs)
    {
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }

    __host__ __device__ PointXY& operator*=(const float scalar)
    {
        x *= scalar;
        y *= scalar;
        return *this;
    }

    __host__ __device__ PointXY operator-(const PointXY& rhs) const
    {
        PointXY res = *this;
        res -= rhs;
        return res;
    }

    __host__ __device__ PointXY operator*(const float scalar) const
    {
        PointXY res = *this;
        res *= scalar;
        return res;
    }

    __host__ __device__ float norm() const
    {
        return std::sqrt(x * x + y * y);
    }

    __host__ __device__ float dot(const PointXY& rhs) const
    {
        return x * rhs.x + y * rhs.y;
    }

    __host__ __device__ PointXY& normalize()
    {
        float normV = this->norm();

        if (!utils::almostEquals<float>(normV, 0)) {
            x /= normV;
            y /= normV;
        }
        return *this;
    }

    __host__ __device__ PointXY normalized() const
    {
        PointXY res = *this;
        res.normalize();
        return res;
    }

    __host__ __device__ float distanceToLine(const PointXY& point, const PointXY& direction) const
    {
        PointXY diff = *this;
        diff -= point;
        return (diff - direction * (diff.dot(direction))).norm();
    }
};

struct PointXYZ {
    using DataType = float3;

    union {
        float3 xyz;
        struct {
            float x;
            float y;
            float z;
        };
    };

    __host__ __device__ PointXYZ(const float x = 0., const float y = 0., const float z = 0.)
        : x(x)
        , y(y)
        , z(z)
    {
    }

    __host__ __device__ DataType data() const
    {
        return xyz;
    }

    __host__ __device__ PointXYZ& operator-=(const PointXYZ& rhs)
    {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return (*this);
    }

    __host__ __device__ PointXYZ& operator*=(const float scalar)
    {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    __host__ __device__ PointXYZ operator-(const PointXYZ& rhs) const
    {
        PointXYZ res = *this;
        res -= rhs;
        return res;
    }

    __host__ __device__ PointXYZ operator*(const float scalar) const
    {
        PointXYZ res = *this;
        res *= scalar;
        return res;
    }

    __host__ __device__ float norm() const
    {
        return std::sqrt(x * x + y * y + z * z);
    }

    __host__ __device__ float dot(const PointXYZ& rhs) const
    {
        return x * rhs.x + y * rhs.y + z * rhs.z;
    }

    __host__ __device__ PointXYZ& normalize()
    {
        float normV = this->norm();
        if (!utils::almostEquals<float>(normV, 0)) {
            x /= normV;
            y /= normV;
            z /= normV;
        }
        return *this;
    }

    __host__ __device__ PointXYZ normalized() const
    {
        PointXYZ res = *this;
        res.normalize();
        return res;
    }

    __host__ __device__ PointXYZ cross(const PointXYZ& rhs) const
    {
        PointXYZ res;
        res.x = y * rhs.z - z * rhs.y;
        res.y = -(x * rhs.z - z * rhs.x);
        res.z = x * rhs.y - y * rhs.x;

        return res;
    }

    __host__ __device__ float distanceToLine(const PointXYZ& point, const PointXYZ& direction) const
    {
        PointXYZ diff = *this;
        diff -= point;
        return (diff - direction * (diff.dot(direction))).norm();
    }
};
}  // namespace cuda
}  // namespace perception
