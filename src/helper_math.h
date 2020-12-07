#include "vector_types.h"

inline __host__ __device__ double dot(double2 x1, double2 x2){
  return x1.x*x2.x + x1.y*x2.y;
}
// inline __host__ __device__ double2 cross(double2 v, double2 w){
//   return make_double2(v.y*w.z - w.y*v.z, w.x*v.z - v.x*w.z, v.x*w.y - v.y*w.x);
// }

inline __host__ __device__ double2 operator-(double2 &a)
{
  return make_double2(-a.x, -a.y);
}
inline __host__ __device__ double2 operator+(double2 a, double2 b)
{
  return make_double2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(double2 &a, double2 b)
{
  a.x += b.x;
  a.y += b.y;
}
inline __host__ __device__ double2 operator+(double2 a, double b)
{
  return make_double2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(double2 &a, double b)
{
  a.x += b;
  a.y += b;
}
inline __host__ __device__ double2 operator+(double b, double2 a)
{
  return make_double2(a.x + b, a.y + b);
}
inline __host__ __device__ double2 operator-(double2 a, double2 b)
{
  return make_double2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(double2 &a, double2 b)
{
  a.x -= b.x;
  a.y -= b.y;
}
inline __host__ __device__ double2 operator-(double2 a, double b)
{
  return make_double2(a.x - b, a.y - b);
}
inline __host__ __device__ double2 operator-(double b, double2 a)
{
  return make_double2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(double2 &a, double b)
{
  a.x -= b;
  a.y -= b;
}
inline __host__ __device__ double2 operator*(double2 a, double2 b)
{
  return make_double2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(double2 &a, double2 b)
{
  a.x *= b.x;
  a.y *= b.y;
}
inline __host__ __device__ double2 operator*(double2 a, double b)
{
  return make_double2(a.x * b, a.y * b);
}
inline __host__ __device__ double2 operator*(double b, double2 a)
{
  return make_double2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(double2 &a, double b)
{
  a.x *= b;
  a.y *= b;
}
inline __host__ __device__ double2 operator/(double2 a, double2 b)
{
  return make_double2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(double2 &a, double2 b)
{
  a.x /= b.x;
  a.y /= b.y;
}
inline __host__ __device__ double2 operator/(double2 a, double b)
{
  return make_double2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(double2 &a, double b)
{
  a.x /= b;
  a.y /= b;
}
inline __host__ __device__ double2 operator/(double b, double2 a)
{
  return make_double2(b / a.x, b / a.y);
}
