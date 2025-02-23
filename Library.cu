#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "Library.cuh";
#include <iostream>;

using namespace std;
//using namespace Library;

template<typename T = float>
struct Vector3 {
public:
    T x = T(), y = T(), z = T();
    __device__ __host__ Vector3(T x = T(), T y = T(), T z = T()) {
        this->x = x;
        this->y = y;
        this->z = z;
    }
    __device__ __host__ T Magnitude() {
        return sqrtf(this->x * this->x + this->y * this->y);
    }
    __device__ __host__ Vector3 operator=(Vector3 value) {
        //return Vector3(value.x, value.y, value.z);
        this->x = value.x;
        this->y = value.y;
        this->z = value.z;
        return *this;
    }
    __device__ __host__ Vector3 operator+(Vector3 value) {
        return Vector3(this->x + value.x, this->y + value.y, this->z + value.z);
    }
    __device__ __host__ Vector3 operator+=(Vector3 value) {
        this->x += value.x;
        this->y += value.y;
        this->z += value.z;
        return *this;
    }
    __device__ __host__ Vector3 operator-(Vector3 value) {
        return Vector3(this->x - value.x, this->y - value.y, this->z - value.z);
    }
    __device__ __host__ Vector3 operator-=(Vector3 value) {
        this->x -= value.x;
        this->y -= value.y;
        this->z -= value.z;
        return *this;
    }

    template<class T>
    __device__ __host__ Vector3 operator*(T value) {
        return Vector3(this->x * value, this->y * value, this->z * value);
    }
    template<class T>
    __device__ __host__ static friend Vector3 operator*(T value, Vector3 vec) {
        return Vector3(vec.x * value, vec.y * value, vec.z * value);
    }
    template<class T>
    __device__ __host__ Vector3 operator/(T value) {
        return Vector3(this->x / value, this->y / value, this->z / value);
    }
    template<class T>
    __device__ __host__ static friend Vector3 operator/(T value, Vector3 vec) {
        return Vector3(vec.x / value, vec.y / value, vec.z / value);
    }

    __device__ __host__ T operator*(Vector3 value) {
        return this->x * value.x + this->y * value.y + this->z * value.z;
    }
};
template<class T>
__device__ T lerp(T a, T b, float t) {
    return a + (b - a) * t;
}
Vector3<int> GetPUCount(int TargetCount) {
    int blocks = max((int)((float)(TargetCount) / 1024), 1);
    int threads = min(TargetCount, 1024);
    return Vector3<int>(blocks, threads);
}
template<class T, class Cell>
__device__ T lerpInGrid(Cell* dgrid, Vector3<float> pos, T Cell::* quantity) {
    int2 cell = make_int2((int)pos.x, (int)pos.y);

    Vector3<float> delta = Vector3<float>((pos.x - ((float)cell.x + .5f)), ((float)pos.y - (cell.y + .5f)));
    int2 direction = make_int2(sign(delta.x), sign(delta.y));

    Cell cells[4] = {
        dgrid[(cell.y) * GridResolutionX + (cell.x)],
        dgrid[(cell.y) * GridResolutionX + (cell.x + direction.x)],
        dgrid[(cell.y + direction.y) * GridResolutionX + (cell.x)],
        dgrid[(cell.y + direction.y) * GridResolutionX + (cell.x + direction.x)],
    };

    T l1 = lerp(cells[0].*quantity, cells[1].*quantity, abs(delta.x));
    T l2 = lerp(cells[2].*quantity, cells[3].*quantity, abs(delta.x));

    T l = lerp(l1, l2, abs(delta.y));

    return l;
}