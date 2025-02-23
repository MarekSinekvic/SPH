#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GLEW_STATIC
//#include <GL/glew.h>
#include <GL/freeglut.h>
//#include <GLFW/glfw3.h>

//#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <string>
#include <chrono>

#include "Library.cu";

using namespace std;

template<class T>class List {
private:
    int MaxSize;
public:
    T* data;
    int length;
    __host__ __device__ List(int MaxSize = 128) {
        this->data = new T[MaxSize];
        //memset(&(this->data),NAN,sizeof(*(this->data))*MaxSize);
        this->length = 0;

        this->MaxSize = MaxSize;
    }
    __host__ void push(T nData) {
        this->data[this->length] = nData;
        this->length++;
        if (this->length > this->MaxSize) throw "Out of max size";
    }
    __host__ void clear() {
        //memset(&(this->data), NAN, sizeof(*(this->data)) * MaxSize);
        this->length = 0;
    }
};
struct Particle {
    float2 position, velocity;
    float density,pressure,heat,charge = 1.f;
    float3 color;
    __host__ __device__ Particle(float2 Position = make_float2(0.f,0.f), float2 Velocity = make_float2(0.f, 0.f), float3 Color = make_float3(0.f, 0.f, 0.f)) {
        this->position = Position;
        this->velocity = Velocity;
        this->color = Color;
        this->density = 0;
        this->pressure = 0;
        this->heat = 0;
    }
};

int pause = 1;

Particle* particles;
deque<vector<Particle>> particles_trace;

const int N = 1024 * 5;

const int xExpansion = 2;
const int yExpansion = 1;

float2 bounds = make_float2(5000 * xExpansion, 5000 * yExpansion);
__constant__ float kernelBoundsX = 5000.f * 2;
__constant__ float kernelBoundsY = 5000.f * 1;  
const int sortGridQualityX = 250 * 2; // bounds.x/Radius*1.5
const int sortGridQualityY = 250 * 1;
const int MaxCellElementsCount = 512;

__constant__ float DT = .2f;
__constant__ float Radius = 30;

__constant__ float PressureIntensity = 4000.f;
__constant__ float DefaultDensity = .3f;
__constant__ float Viscosity = 10.f;
__constant__ float PressureProjection = 80.f;

__constant__ float HeatTransferIntensity = 0.f;
__constant__ float HeatImpactGrowth = 0.f;
__constant__ float HeatVolumeGrowth = 1.f;

__constant__ float G = 0.5f * 0;

struct Wall {
public:
    float2 p1,p2;
    float2 normal,tangent;
    float length;
    __host__ __device__ Wall(float2 p1, float2 p2) {
        this->p1 = p1;
        this->p2 = p2;
        this->tangent = make_float2(p2.x - p1.x, p2.y - p1.y);
        this->length = sqrtf(this->tangent.x* this->tangent.x + this->tangent.y* this->tangent.y);
        this->tangent = make_float2(this->tangent.x/this->length, this->tangent.y / this->length);
        this->normal = make_float2(-tangent.y, tangent.x);
    
        printf("%f %f\n", this->normal.x, this->normal.y);
    }
};

Particle* dparticles;
float* d_press;

const int wallsCount = 3;
Wall* d_walls;

size_t pdevsize = sizeof(Particle) * N;

int RenderState = 0;
Vector3<float> mousePos = Vector3<float>(0.f, 0.f);
Vector3<float> mouseMovement = Vector3<float>(0.f, 0.f);
int mouseButton = 0;

int2 CalcLinearBlocksCount() {
    int threads = 1024;
    if (N < 1024) threads = N;
    return make_int2((int)((float)N / 1024), threads);
}

__device__ float dot(float2 a, float2 b) {
    return a.x*b.x+a.y*b.y;
}

__global__ void IntegrateKernel(Particle* P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > N - 1) return;
    
    P[i].position.x += P[i].velocity.x * DT;
    P[i].position.y += P[i].velocity.y * DT;

    P[i].velocity.y += G * DT;

    float2 uv = make_float2(P[i].position.x/kernelBoundsX, P[i].position.y/ kernelBoundsY);
    if (powf((.5 - uv.x) * 2 + 0.8, 2.f) < .1f * .1f) { // + powf((.5 - uv.y) / 2, 2.f)
        //atomicAdd(&(d_heat[i]), (2.f - d_heat[i])*.1f);
        //atomicAdd(&(P[i].velocity.x), (-25.f - P[i].velocity.x)*0.08f);
        //atomicAdd(&(P[i].velocity.x), -0.05f);
    }

    if (P[i].velocity.x > 30.f) P[i].velocity.x = 30.f;
    if (P[i].velocity.y > 30.f) P[i].velocity.y = 30.f;
    if (P[i].velocity.x < -30.f) P[i].velocity.x = -30.f;
    if (P[i].velocity.y < -30.f) P[i].velocity.y = -30.f;
}
__global__ void BoundsKernel(Particle* P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > N - 1) return;

    float power = 1.;
    if (P[i].position.x < 1.f) {
        /*P[i].position.x += kernelBoundsX-1.f*2;
        P[i].velocity.y = 0.f;
        P[i].color.x = P[i].position.y / kernelBoundsY;
        P[i].color.z = 1.f-P[i].position.y / kernelBoundsY;*/
        P[i].position.x = 1.f;
        P[i].velocity.x = 0.f;
        
    }
    if (P[i].position.y < 1.f) {
        P[i].position.y = 1.f;
        P[i].velocity.y = 0.f;
    }
    if (P[i].position.x > kernelBoundsX - 1.f) {
        /*P[i].position.x -= kernelBoundsX - 1.f*2;
        P[i].velocity.y = 0.f;
        P[i].color.x = P[i].position.y / kernelBoundsY;
        P[i].color.z = 1.f - P[i].position.y / kernelBoundsY;*/
        P[i].position.x = kernelBoundsX - 1.f;
        P[i].velocity.x = 0.f;
    }
    if (P[i].position.y > kernelBoundsY - 1.f) {
        P[i].position.y = kernelBoundsY - 1.f;
        P[i].velocity.y = 0.f;
    }
}
__global__ void BoundWalls(Particle* P, Wall* dwalls, int wallsCount = 1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > N - 1) return;

    float2 newPos = P[i].position;
    for (int j = 0; j < wallsCount; j++) {
        float2 delta = make_float2(P[i].position.x - dwalls[j].p1.x, P[i].position.y - dwalls[j].p1.y);
        float deltaDst = sqrtf(delta.x*delta.x + delta.y*delta.y);
        float inlineProj = dot(delta, dwalls[j].tangent);
        float normalProj = dot(delta, dwalls[j].normal);

        int dir = signbit(normalProj) == 0 ? -1 : 1;
        float2 normal = make_float2(dwalls[j].normal.x * -dir, dwalls[j].normal.y * -dir);

        const float C = 10.f;
        if (inlineProj >= 0.f && inlineProj <= dwalls[j].length) {
            float dst = abs(normalProj);
            if (dst < C) {
                float penet = (C - dst);
                //P[i].position = make_float2(P[i].position.x + normal.x*penet, P[i].position.y + normal.y*penet);
                newPos = make_float2(P[i].position.x + normal.x * penet, P[i].position.y + normal.y * penet);

                float velproj = dot(P[i].velocity, normal);
                if (velproj < 0.f)
                    P[i].velocity = make_float2(P[i].velocity.x - normal.x * (velproj), P[i].velocity.y - normal.y * (velproj));
                //P[i].velocity = make_float2(0.f,0.f);
            }
        } else {
            if (inlineProj < 0.f) {
                float dst = sqrtf(delta.x*delta.x + delta.y*delta.y);
                if (dst < C) {
                    //P[i].position = make_float2(P[i].position.x + delta.x/dst * (C-dst), P[i].position.y + delta.x / dst * (C - dst));
                    newPos = make_float2(P[i].position.x + delta.x / dst * (C - dst), P[i].position.y + delta.x / dst * (C - dst));
                    
                    float velproj = dot(P[i].velocity, make_float2(delta.x/dst, delta.y/dst));
                    if (velproj < 0.f)
                        P[i].velocity = make_float2(P[i].velocity.x - delta.x / dst * (velproj), P[i].velocity.y - delta.y / dst * (velproj));
                    //P[i].velocity = make_float2(0.f, 0.f);
                }
            } else {
                float2 delta2 = make_float2(P[i].position.x-dwalls[j].p2.x, P[i].position.y - dwalls[j].p2.y);
                float dst = sqrtf(delta2.x*delta2.x + delta2.y*delta2.y);
                if (dst < C) {
                    //P[i].position = make_float2(P[i].position.x + delta2.x / dst * (C - dst), P[i].position.y + delta2.x / dst * (C - dst));
                    newPos = make_float2(P[i].position.x + delta2.x / dst * (C - dst), P[i].position.y + delta2.x / dst * (C - dst));
                    
                    float velproj = dot(P[i].velocity, make_float2(delta2.x / dst, delta2.y / dst));
                    if (velproj < 0.f)
                        P[i].velocity = make_float2(P[i].velocity.x - delta2.x / dst * (velproj), P[i].velocity.y - delta2.y / dst * (velproj));
                    //P[i].velocity = make_float2(0.f, 0.f);
                }
            }
        }
    }

    P[i].position = newPos;
}

/*__device__ int* GetNearBodies(float2 pos, int* d_grid_data, int* d_grid_lengths) {
    //int gx = (int)(P[i].position.x / kernelBoundsX * sortGridQualityX);
    //int gy = (int)(P[i].position.y / kernelBoundsY * sortGridQualityY);
    if (gx < 0 || gx > sortGridQualityX - 1) return;
    if (gy < 0 || gy > sortGridQualityY - 1) return;

    int bodies[MaxCellElementsCount*9];
    for (int oy = -1; oy <= 1; oy++) {
        if (gy + oy < 0 || gy + oy > sortGridQualityY - 1) continue;
        for (int ox = -1; ox <= 1; ox++) {
            if (gx + ox < 0 || gx + ox > sortGridQualityX - 1) continue;
            for (int j = 0; j < d_grid_lengths[(gy + oy) * sortGridQualityX + (gx + ox)]; j++) {
                bodies[((oy + 1) * 3 + (ox+1)) * MaxCellElementsCount + j] = d_grid_data[((gy + oy) * sortGridQualityX + gx + ox) * MaxCellElementsCount + j];
            }
        }
    }
    return bodies;

}*/
__constant__ float smoothVar = 12.f/(30*30*30*30); //12 / (Radius * Radius * Radius * Radius);
__global__ void DensitiesKernel(Particle* P, int* d_grid_data,int* d_grid_lengths) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > N - 1) return;

    float F = 0.f;

    int gx = (int)(P[i].position.x / kernelBoundsX * sortGridQualityX);
    int gy = (int)(P[i].position.y / kernelBoundsY * sortGridQualityY);
    if (gx < 0 || gx > sortGridQualityX - 1) return;
    if (gy < 0 || gy > sortGridQualityY - 1) return;

    for (int oy = -1; oy <= 1; oy++) {
        if (gy + oy < 0 || gy + oy > sortGridQualityY - 1) continue;
        for (int ox = -1; ox <= 1; ox++) {
            if (gx + ox < 0 || gx + ox > sortGridQualityX - 1) continue;
            for (int j = 0; j < d_grid_lengths[(gy+oy) * sortGridQualityX + gx+ox]; j++) {
                Particle oPart = P[d_grid_data[((gy+oy) * sortGridQualityX + gx+ox) * MaxCellElementsCount + j]];
                // printf("%d ", d_grid_data[(gy * sortGridQuality + gx) * MaxCellElementsCount + j]);

                float2 delta = make_float2(oPart.position.x - P[i].position.x, oPart.position.y - P[i].position.y);
                float mag = delta.x * delta.x + delta.y * delta.y;
                if (mag < Radius * Radius && mag > 0) {
                    mag = sqrtf(mag);

                    float v1 = Radius - mag;
                    F += 4 * v1 * v1 * v1 / (Radius * Radius * Radius * Radius);
                }
            }
        }
    }
    
    //atomicAdd(&(densities[i]), F*(1+dheat[i]));
    P[i].density = F * (1+P[i].heat*HeatVolumeGrowth);
}
__global__ void ForcesKernel(Particle* P, int* d_grid_data,int* d_grid_lengths) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float2 newV = make_float2(0.f, 0.f);
    
    int gx = (int)(P[i].position.x / kernelBoundsX * sortGridQualityX);
    int gy = (int)(P[i].position.y / kernelBoundsY * sortGridQualityY);
    if (gx < 0 || gx > sortGridQualityX - 1) return;
    if (gy < 0 || gy > sortGridQualityY - 1) return;

    //if (dens[i] < 0.0001 || dens[i] > 10) return;

    for (int oy = -1; oy <= 1; oy++) {
        if (gy + oy < 0 || gy + oy > sortGridQualityY - 1) continue;
        for (int ox = -1; ox <= 1; ox++) {
            if (gx + ox < 0 || gx + ox > sortGridQualityX - 1) continue;
            for (int j = 0; j < d_grid_lengths[(gy+oy) * sortGridQualityX + (gx+ox)]; j++) {
                int ni = d_grid_data[((gy + oy) * sortGridQualityX + gx + ox) * MaxCellElementsCount + j];

                Particle oParticle = P[ni];

                //if (oDens < 0.0001 || oDens > 10) return;

                float2 delta = make_float2(oParticle.position.x - P[i].position.x, oParticle.position.y - P[i].position.y);
                float mag = delta.x * delta.x + delta.y * delta.y;
                if (mag < Radius * Radius && mag > 0) {
                    mag = sqrtf(mag);
                    float d1 = (Radius - mag);

                    float2 oVel = oParticle.velocity;
                    float oDens = oParticle.density;

                    float2 dir = make_float2(delta.x / mag, delta.y / mag);

                    float2 velDelta = make_float2(oVel.x - P[i].velocity.x, oVel.y - P[i].velocity.y);
                    float proj = velDelta.x * dir.x + velDelta.y * dir.y;
                    //if (proj < 0) proj = 0;
                    //float Fproj = Viscosity * proj / oDens * smoothVar*2 * d1;

                    //newV.x += dir.x * (Fproj);
                    //newV.y += dir.y * (Fproj);
                    newV.x += velDelta.x * Viscosity / oDens * smoothVar * 2 * d1;
                    newV.y += velDelta.y * Viscosity / oDens * smoothVar * 2 * d1;

                    P[i].heat += sqrtf(velDelta.x * velDelta.x + velDelta.y * velDelta.y) * Viscosity * HeatImpactGrowth / oDens * smoothVar * 2 * d1;
                }
            }
        }
    }
    P[i].velocity.x += newV.x * DT;
    P[i].velocity.y += newV.y * DT;
    //atomicAdd(&(vel[i].x), newV.x*DT);
    //atomicAdd(&(vel[i].y), newV.y*DT);
}
__global__ void PressureKernel(Particle* P, int* d_grid_data, int* d_grid_lengths) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float newP = 0.f;

    int gx = (int)(P[i].position.x / kernelBoundsX * sortGridQualityX);
    int gy = (int)(P[i].position.y / kernelBoundsY * sortGridQualityY);
    if (gx < 0 || gx > sortGridQualityX - 1) return;
    if (gy < 0 || gy > sortGridQualityY - 1) return;

    //if (dens[i] < 0.0001 || dens[i] > 10) return;

    for (int oy = -1; oy <= 1; oy++) {
        if (gy + oy < 0 || gy + oy > sortGridQualityY - 1) continue;
        for (int ox = -1; ox <= 1; ox++) {
            if (gx + ox < 0 || gx + ox > sortGridQualityX - 1) continue;
            for (int j = 0; j < d_grid_lengths[(gy + oy) * sortGridQualityX + (gx + ox)]; j++) {
                int ni = d_grid_data[((gy + oy) * sortGridQualityX + gx + ox) * MaxCellElementsCount + j];

                Particle oParticle = P[ni];

                //if (oDens < 0.0001 || oDens > 10) return;

                float2 delta = make_float2(oParticle.position.x - P[i].position.x, oParticle.position.y - P[i].position.y);
                float mag = delta.x * delta.x + delta.y * delta.y;
                if (mag < Radius * Radius && mag > 0) {
                    mag = sqrtf(mag);
                    float v1 = (Radius - mag);

                    float2 oVel = oParticle.velocity;
                    float oDens = oParticle.density;
                    if (abs(oDens) < 0.01f) continue;
                    float oPress = oParticle.pressure;

                    float div = ((oVel.x - P[i].velocity.x) * delta.x / mag + (oVel.y - P[i].velocity.y) * delta.y / mag) / oDens * (4 * v1 * v1 * v1 / (Radius * Radius * Radius * Radius));
                    if (div < -0.5f) div = -0.5f; //if (div > 10.f) div = 10.f;
                    if (div > 0.5f) div = 0.5f;
                    newP += oPress * (4 * v1 * v1 * v1 / (Radius * Radius * Radius * Radius));
                    newP -= PressureProjection * div;
                    //newP += dens[i];
                }
            }
        }
    }
    P[i].pressure = (P[i].density - DefaultDensity) * PressureIntensity + newP;
}
__global__ void DivergenceKernel(Particle* P, int* d_grid_data, int* d_grid_lengths) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float2 newV = make_float2(0.f, 0.f);

    int gx = (int)(P[i].position.x / kernelBoundsX * sortGridQualityX);
    int gy = (int)(P[i].position.y / kernelBoundsY * sortGridQualityY);
    if (gx < 0 || gx > sortGridQualityX - 1) return;
    if (gy < 0 || gy > sortGridQualityY - 1) return;

    //if (dens[i] < 0.0001 || dens[i] > 10) return;

    for (int oy = -1; oy <= 1; oy++) {
        if (gy + oy < 0 || gy + oy > sortGridQualityY - 1) continue;
        for (int ox = -1; ox <= 1; ox++) {
            if (gx + ox < 0 || gx + ox > sortGridQualityX - 1) continue;
            for (int j = 0; j < d_grid_lengths[(gy + oy) * sortGridQualityX + (gx + ox)]; j++) {
                int ni = d_grid_data[((gy + oy) * sortGridQualityX + gx + ox) * MaxCellElementsCount + j];

                Particle oParticle = P[ni];

                //if (oDens < 0.0001 || oDens > 10) return;

                float2 delta = make_float2(oParticle.position.x - P[i].position.x, oParticle.position.y - P[i].position.y);
                float mag = delta.x * delta.x + delta.y * delta.y;
                if (mag < Radius * Radius && mag > 0) {
                    mag = sqrtf(mag);
                    float d1 = (Radius - mag);

                    float2 oVel = oParticle.velocity;
                    float oDens = oParticle.density;
                    float oPress = oParticle.pressure;

                    float2 dir = make_float2(delta.x / mag, delta.y / mag);

                    //float Pressure = ((P[i].density - DefaultDensity));
                    //float PressureOther = ((oDens - DefaultDensity));

                    //float N = (Pressure + PressureOther) / 2 / oDens * PressureIntensity;
                    float N = (P[i].pressure + oPress)/2/oDens;
                    float Fpress = -N * smoothVar * d1 * d1;

                    newV.x += dir.x * (Fpress);
                    newV.y += dir.y * (Fpress);
                }
                /*if (mag > 0) {
                    mag = sqrtf(mag);
                    float2 dir = make_float2(delta.x / mag, delta.y / mag);
                    newV.x += dir.x * 2.f / (1.f + mag);
                    newV.y += dir.y * 2.f / (1.f + mag);
                }*/
            }
        }
    }
    P[i].velocity.x += newV.x * DT;
    P[i].velocity.y += newV.y * DT;
}
__global__ void HeatingKernel(Particle* P, int* d_grid_data, int* d_grid_lengths) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int gx = (int)(P[i].position.x / kernelBoundsX * sortGridQualityX);
    int gy = (int)(P[i].position.y / kernelBoundsY * sortGridQualityY);
    if (gx < 0 || gx > sortGridQualityX - 1) return;
    if (gy < 0 || gy > sortGridQualityY - 1) return;

    //if (dens[i] < 0.0001 || dens[i] > 10) return;

    float newHeat = 0.f;
    for (int oy = -1; oy <= 1; oy++) {
        if (gy + oy < 0 || gy + oy > sortGridQualityY - 1) continue;
        for (int ox = -1; ox <= 1; ox++) {
            if (gx + ox < 0 || gx + ox > sortGridQualityX - 1) continue;
            for (int j = 0; j < d_grid_lengths[(gy + oy) * sortGridQualityX + (gx + ox)]; j++) {
                int ni = d_grid_data[((gy + oy) * sortGridQualityX + gx + ox) * MaxCellElementsCount + j];

                Particle oParticle = P[ni];

                float2 delta = make_float2(oParticle.position.x - P[i].position.x, oParticle.position.y - P[i].position.y);
                float mag = delta.x * delta.x + delta.y * delta.y;
                if (mag < Radius * Radius && mag > 0) {
                    mag = sqrtf(mag);

                    float oDens = oParticle.density;
                    float oHeat = oParticle.heat;

                    if (oHeat != P[i].heat) {
                        float v = (oHeat - P[i].heat) / 2 / oDens * smoothVar * 2;
                        newHeat += v;
                    }
                }

            }
        }
    }

    P[i].heat += newHeat * HeatTransferIntensity * DT;

}
__global__ void extForcesKernel(Particle* P, int* d_grid_data, int* d_grid_lengths, float2 mousePos, int mouseState) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (mouseState == 2) {
        float2 delta = make_float2(mousePos.x * (kernelBoundsX)-P[i].position.x, mousePos.y * kernelBoundsY - P[i].position.y);
        float len = sqrtf(delta.x*delta.x + delta.y*delta.y);
        P[i].velocity.x += delta.x / len / (1.f+len * 0.3f) * 30.f;
        P[i].velocity.y += delta.y / len / (1.f+len * 0.3f) * 30.f;
    }

    int gx = (int)(P[i].position.x / kernelBoundsX * sortGridQualityX);
    int gy = (int)(P[i].position.y / kernelBoundsY * sortGridQualityY);
    if (gx < 0 || gx > sortGridQualityX - 1) return;
    if (gy < 0 || gy > sortGridQualityY - 1) return;

    //if (dens[i] < 0.0001 || dens[i] > 10) return;

    /*const int R = 8;
    for (int oy = -R; oy <= R; oy++) {
        if (gy + oy < 0 || gy + oy > sortGridQualityY - 1) continue;
        for (int ox = -R; ox <= R; ox++) {
            if (gx + ox < 0 || gx + ox > sortGridQualityX - 1) continue;
            for (int j = 0; j < d_grid_lengths[(gy + oy) * sortGridQualityX + (gx + ox)]; j++) {
                int ni = d_grid_data[((gy + oy) * sortGridQualityX + gx + ox) * MaxCellElementsCount + j];

                Particle oParticle = P[ni];

                //if (oDens < 0.0001 || oDens > 10) return;

                float2 delta = make_float2(oParticle.position.x - P[i].position.x, oParticle.position.y - P[i].position.y);
                float mag = delta.x * delta.x + delta.y * delta.y;
                if (mag > Radius*Radius) {
                    mag = sqrtf(mag);

                    const float F = 14.f;
                    P[i].velocity.x += delta.x / mag / powf(mag * 2.f, 2.f) * F * DT;
                    P[i].velocity.y += delta.y / mag / powf(mag * 2.f, 2.f) * F * DT;
                }
            }
        }
    }*/
    for (int j = 0; j < N && 1; j++) {
        Particle oParticle = P[j];
        float2 delta = make_float2(oParticle.position.x - P[i].position.x, oParticle.position.y - P[i].position.y);
        float mag = delta.x * delta.x + delta.y * delta.y;
        //if (mag > 300.f*300.f) continue;
        if (mag > Radius * Radius && mag < 1000.f * 1000.f) {//
            mag = sqrtf(mag);

            const float F = 200.f; // * oParticle.charge * P[i].charge
            P[i].velocity.x += delta.x / mag / powf(mag * 2.f,2.f) * F * DT;
            P[i].velocity.y += delta.y / mag / powf(mag * 2.f,2.f) * F * DT;
        }
    }
    float2 delta = make_float2(kernelBoundsX/2 - P[i].position.x, kernelBoundsY / 2 - P[i].position.y);
    float M = sqrtf(delta.x * delta.x + delta.y * delta.y);
    //P[i].velocity.x += delta.x / M * .2;
    //P[i].velocity.y += delta.y / M * .2;
}

/*__global__ void SortKernel(int* d_data, int* d_lens) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= sortGridQualityY || j >= sortGridQualityX || k*2 >= d_lens[i*sortGridQualityX + j]) return;

    int ind = (i * sortGridQualityX + j) * sortGridQualityY + k*2;
    int p1 = d_data[ind];
    int p2 = d_data[ind+1];
    int p3 = d_data[ind + 2];
    if (p2 < p1) {
        atomicExch(&(d_data[ind]), p2);
        atomicExch(&(d_data[ind+1]), p1);
    }
    if (p3 < p2) {
        atomicExch(&(d_data[ind+1]), p3);
        atomicExch(&(d_data[ind + 2]), p2);
    }
}*/


__device__ void swap(int ind1, int ind2, Particle* P) {
    float oldx = atomicExch(&(P[ind1].position.x), P[ind2].position.x);
    atomicExch(&(P[ind2].position.x), oldx);
    float oldy = atomicExch(&(P[ind1].position.y), P[ind2].position.y);
    atomicExch(&(P[ind2].position.y), oldy);


    float oldvelx = atomicExch(&(P[ind1].velocity.x), P[ind2].velocity.x);
    atomicExch(&(P[ind2].velocity.x), oldvelx);
    float oldvely = atomicExch(&(P[ind1].velocity.y), P[ind2].velocity.y);
    atomicExch(&(P[ind2].velocity.y), oldvely);

    float oldclrx = atomicExch(&(P[ind1].color.x), P[ind2].color.x);
    atomicExch(&(P[ind2].color.x), oldclrx);
    float oldclry = atomicExch(&(P[ind1].color.y), P[ind2].color.y);
    atomicExch(&(P[ind2].color.y), oldclry);
    float oldclrz = atomicExch(&(P[ind1].color.z), P[ind2].color.z);
    atomicExch(&(P[ind2].color.z), oldclrz);



    float olddens = atomicExch(&(P[ind1].density), P[ind2].density);
    atomicExch(&(P[ind2].density), olddens);
}
__global__ void SortKernel(Particle* P, int offset = 0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i*2 >= N) return;
    
    int ind1 = i*2+offset;
    int ind2 = ind1 + 1;
    int ind3 = ind1 - 1;

    if (ind1 >= N) return;
    if (ind2 >= N) return;
    if (ind3 < 0) return;

    float hasha = P[ind1].position.y * kernelBoundsX + P[ind1].position.x;
    float hashb = P[ind2].position.y * kernelBoundsX + P[ind2].position.x;
    float hashc = P[ind3].position.y * kernelBoundsX + P[ind3].position.x;

    if (hasha > hashb) {
        swap(ind1,ind2, P);
    }
    if (hasha < hashc) {
        swap(ind1, ind3, P);
    }
}

__global__ void FillGridKernel(int* d_data, int* d_lens, Particle* P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > N - 1) return;

    int gx = (int)(P[i].position.x / kernelBoundsX * sortGridQualityX);
    int gy = (int)(P[i].position.y / kernelBoundsY * sortGridQualityY);
    if (gx < 0 || gx > sortGridQualityX - 1) return;
    if (gy < 0 || gy > sortGridQualityY - 1) return;

    int k = atomicAdd(&(d_lens[gy * sortGridQualityX + gx]), 1);
    /*for (int j = 0; j < k; j++) {
        int pind = (gy * sortGridQualityX + gx) * MaxCellElementsCount + j;
        int p1 = d_data[pind];
        if (p1 > i) {
            atomicExch(&p1, i);
            int past = p1;
            for (int r = k; r >= pind + 1; --r) {
                atomicExch(&(d_data[r+1]), d_data[r]);
            }
            atomicExch(&(d_data[pind + 1]),p1);
            break;
        }
    }*/
    d_data[(gy * sortGridQualityX + gx) * MaxCellElementsCount + k] = i;
}
struct Grid {
private:
    int* h_data;
    int* h_lens;

    void AllocateMemory() {
        this->h_data = new int[sortGridQualityX * sortGridQualityY * MaxCellElementsCount];
        this->h_lens = new int[sortGridQualityX * sortGridQualityY];

        cudaMalloc(&(this->d_data), this->dataSize);
        cudaMalloc(&(this->d_lens), this->lenSize);
    }
    void TransformData() {
        for (int i = 0; i < sortGridQualityY; i++) {
            for (int j = 0; j < sortGridQualityX; j++) {
                this->grid[i][j].clear();
            }
        }
        for (int i = 0; i < sortGridQualityY; i++) {
            for (int j = 0; j < sortGridQualityX; j++) {
                for (int k = 0; k < h_lens[i * sortGridQualityX + j]; k++) {
                    //cout << "Debug: " << this->grid[i][j][k] << ',' << h_data[i * sortGridQuality + j * 128 + k] << endl;
                    this->grid[i][j].push_back(h_data[(i * sortGridQualityX + j) * MaxCellElementsCount + k]);
                }
            }
        }
    }
public:
    vector<int> grid[sortGridQualityY][sortGridQualityX];
    int* d_data;
    int* d_lens;
    size_t dataSize;
    size_t lenSize;
    Grid() {
        this->lenSize = sizeof(int) * sortGridQualityX * sortGridQualityY;
        this->dataSize = sizeof(int) * sortGridQualityX * sortGridQualityY * MaxCellElementsCount;

        this->AllocateMemory();
    }
    void UpdateGrid(Particle* P) {
        //int dimSize = sortGridQualityX;
        //ClearGrid << < dim3(1, 1, 1), dim3(dimSize, dimSize, 1) >> > (this->d_data, this->d_lens);
        cudaMemset(this->d_data, 0, this->dataSize);
        cudaMemset(this->d_lens, 0, this->lenSize);

        int blocksCount = (int)((float)N / 1024) + 1;
        FillGridKernel << < blocksCount, 1024 >> > (this->d_data, this->d_lens, P);
    }
    void dumpGrid() {
        cudaMemcpy(h_data, d_data, this->dataSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_lens, d_lens, this->lenSize, cudaMemcpyDeviceToHost);

        this->TransformData();
    }
};

Grid mainGrid = Grid();

float S(float x) {
    return (powf(2.718f, x) - 1) / (powf(2.718f, x) + 1);
}
const float V = 3.7f;
void resetSim() {
    cudaMemcpy(particles,dparticles,pdevsize,cudaMemcpyDeviceToHost);

    /*for (int n = 0; n < N; n++) {
        float rx = ((float)(rand()) / RAND_MAX - 0.5f) * 2.f;
        float ry = ((float)(rand()) / RAND_MAX - 0.5f) * 2.f;

        float vx = ((float)(rand()) / RAND_MAX - 0.5f) * 2.f;
        float vy = ((float)(rand()) / RAND_MAX - 0.5f) * 2.f;

        particles[n].position = make_float2((rx + 1.f)/2.f*bounds.x/2, bounds.y / 2 + ry * bounds.y / 2);
        //particles[n].position = make_float2(bounds.x / 2 + rx * bounds.x / 2, (ry / 2 + 0.5f) * bounds.y/2);
    }*/
    const float L = sqrtf(N/2)*10;
    const float2 center = make_float2(bounds.x/2, bounds.y / 2);
    for (int n = 0; n < N/2; n++) {
        float rx =0.f, ry =0.f;
        if (n > 0) {
            rx = (float)(n % (int)sqrtf(N / 2)) / sqrtf(N / 2)-.5f;
            ry = (float)(floorf((float)n / sqrtf(N / 2))) / sqrtf(N / 2)-.5f;
        }
        particles[n].position = make_float2(center.x+rx*L-600, center.y + ry*L);
        particles[n].velocity = make_float2(0, V);
        //particles[n].color = make_float3(abs(rx) * 2, abs(ry) * 2, .5f);
        particles[n].color = make_float3(.7f, .2f, .2f);
        //particles[n].position = make_float2(bounds.x / 2 + rx * bounds.x / 2, (ry / 2 + 0.5f) * bounds.y/2);
    }
    for (int n = N/2; n < N; n++) {
        float rx = 0.f, ry = 0.f;
        if (n > 0) {
            rx = (float)((n - N / 2) % (int)sqrtf(N / 2))/sqrtf(N/2)-.5f;
            ry = (float)(floorf((float)(n - N / 2) / sqrtf(N / 2))) / sqrtf(N / 2)-.5f;
        }

        particles[n].position = make_float2(center.x + rx * L+600, center.y + ry * L);
        particles[n].velocity = make_float2(0, -V);
        //particles[n].color = make_float3(abs(rx) * 2, abs(ry) * 2, .5f);
        particles[n].color = make_float3(.2f, .2f, .7f);
        //particles[n].position = make_float2(bounds.x / 2 + rx * bounds.x / 2, (ry / 2 + 0.5f) * bounds.y/2);
    }
    std::qsort(particles, N, sizeof(Particle), [](const void* a, const void* b) {
        Particle _a = *((Particle*)a);
        Particle _b = *((Particle*)b);

        float hash_a = _a.position.y * bounds.x + _a.position.x;
        float hash_b = _b.position.y * bounds.x + _b.position.x;

        if (hash_a > hash_b) return 1;
        //if (ga.x > gb.x) return -1;
        //if (hash_b < hash_a) return 1;
        return -1;
    });

    for (int n = 0; n < N; n++) {
        float rx = (particles[n].position.x / bounds.x - .5f) * 2.f;
        float ry = (particles[n].position.y / bounds.y - .5f) * 2.f;

        //particles[n].position = make_float2(-0.f / (1.f + (powf(rx * 8.f, 2.f) + powf(ry * 8.f, 2.f))), 0);
        //particles[n].velocity = make_float2(2.f, 0);
        particles[n].density = 0.f;
        particles[n].heat = S(10 * ry) * 0.2f * 0;
        particles[n].charge = ((roundf((float)rand() / RAND_MAX) - 0.5f) * 2);

        //particles[n].color = make_float3(abs(rx), abs(ry), .5f);
        //particles[n].color = make_float3(particles[n].position.y / bounds.y, .2f, 1.f-particles[n].position.y / bounds.y);
    }

    cudaMemcpy(dparticles, particles, pdevsize, cudaMemcpyHostToDevice);

}
void update() {
    if (pause != 0) return;
    int2 blocksCount = CalcLinearBlocksCount();
    for (int t = 0; t < 3; t++) {
        IntegrateKernel << < blocksCount.x, blocksCount.y >> > (dparticles);
        BoundsKernel << < blocksCount.x, blocksCount.y >> > (dparticles);
        BoundWalls <<< blocksCount.x, blocksCount.y >>> (dparticles, d_walls, wallsCount);

        for (int i = 0; i < 2; i++) {
            SortKernel << < (int)((float)N / 2 / 1024), 1024 >> > (dparticles, 0);
            SortKernel << < (int)((float)N / 2 / 1024), 1024 >> > (dparticles, 1);
        }

        mainGrid.UpdateGrid(dparticles);

        DensitiesKernel << < blocksCount.x, blocksCount.y >> > (dparticles, mainGrid.d_data, mainGrid.d_lens);
        ForcesKernel << < blocksCount.x, blocksCount.y >> > (dparticles, mainGrid.d_data, mainGrid.d_lens);
        PressureKernel << < blocksCount.x, 1024 >> > (dparticles, mainGrid.d_data, mainGrid.d_lens);
        DivergenceKernel << < blocksCount.x, blocksCount.y >> > (dparticles, mainGrid.d_data, mainGrid.d_lens);
        extForcesKernel <<< blocksCount.x, blocksCount.y >>> (dparticles, mainGrid.d_data, mainGrid.d_lens, make_float2(mousePos.x, mousePos.y), mouseButton);
        //HeatingKernel <<< blocksCount.x, 1024 >>> (d_pos,d_dens,d_heat, mainGrid.d_data, mainGrid.d_lens);
    }
}
void RenderString(float x, float y, const unsigned char* string, float3 const& rgb = make_float3(1.f, 1.f, 1.f))
{
    char* c;

    glColor3f(rgb.x, rgb.y, rgb.z);
    glRasterPos2f(x, y);

    glutBitmapString(GLUT_BITMAP_HELVETICA_18, string);
}
float RenderIntens = 1.f;
int frame = 0;
void shootTrace() {
    if (particles_trace.size() > 10)
        particles_trace.pop_back();

    particles_trace.push_front(vector<Particle>());
    for (int i = 0; i < N; i++) {
        particles_trace.front().push_back(Particle(
            make_float2(particles[i].position.x, particles[i].position.y),
            make_float2(particles[i].velocity.x, particles[i].velocity.y),
            make_float3(particles[i].color.x, particles[i].color.y, particles[i].color.z)
        ));
    }
}
void display() {
    clock_t start = clock();
    update();

    //cout << cudaGetErrorString(cudaGetLastError()) << endl;

    int2 blocksCount = CalcLinearBlocksCount();
    cudaMemcpyAsync(particles, dparticles, pdevsize, cudaMemcpyDeviceToHost);
    /*std::qsort(particles, N, sizeof(Particle), [](const void* a, const void* b) {
        Particle _a = *((Particle*)a);
        Particle _b = *((Particle*)b);

        float hash_a = _a.position.y * bounds.x + _a.position.x;
        float hash_b = _b.position.y * bounds.x + _b.position.x;

        if (hash_a > hash_b) return 1;
        //if (ga.x > gb.x) return -1;
        //if (hash_b < hash_a) return 1;
        return -1;
    });

    cudaMemcpyAsync(dparticles,particles, pdevsize, cudaMemcpyHostToDevice, 0);*/

    //glPointSize(800.f / texWid + +1.f);//

    auto renderstart = chrono::high_resolution_clock::now();

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);

    //cout << particles_trace.size() << endl;
    

    //shootTrace();
    /*for (int t = 0; t < min(1, (int)particles_trace.size()); t++) {
        for (int i = 0; i < particles_trace[t].size(); i++) {
            int ind = (int)particles_trace.size() - 1 - t;
            Particle P = particles_trace[ind][i];
            if (RenderState == 0)
                glColor3f(P.color.x * RenderIntens + (P.heat * 4.f), P.color.y * RenderIntens, P.color.z * RenderIntens);
            else if (RenderState == 1)
                glColor3f(P.heat * 2.5f, .2f, -P.heat * 2.5f);
            else if (RenderState == 2) {
                float v = (P.pressure) * RenderIntens * .02f;
                glColor3f(v, 0.3f, -v);
            }
            else if (RenderState == 3) {
                glColor3f((float)i / N * RenderIntens, 0.3f, 0.3f);
            }

            glVertex2f((P.position.x / bounds.x - .5f) * 2.f, -(P.position.y / bounds.y - .5f) * 2.f);
        }
    }*/
    for (int i = 0; i < N; i++) {
        if (RenderState == 0)
            glColor3f(particles[i].color.x* RenderIntens * (1.f+particles[i].heat * 2.f), particles[i].color.y* RenderIntens * (1.f + particles[i].heat * 2.f) * (1.f + particles[i].heat *2.f), particles[i].color.z* RenderIntens);
        else if (RenderState == 1)
            glColor3f(particles[i].heat * 2.5f* RenderIntens, .2f, -particles[i].heat * 2.5f* RenderIntens);
        else if (RenderState == 2) {
            float v = (particles[i].pressure)* RenderIntens*.02f;
            glColor3f(v, 0.3f, -v);
        }
        else if (RenderState == 3) {
            glColor3f((float)i/N* RenderIntens, 0.3f, 0.3f);
        }

        glVertex2f((particles[i].position.x / bounds.x - .5f) * 2.f, -(particles[i].position.y / bounds.y - .5f) * 2.f);
    }
    glEnd();
    RenderString(-1.f, .93f, (unsigned char*)((to_string((int)(float(clock() - start) / CLOCKS_PER_SEC * 1000)) + " delta time").c_str()));
    RenderString(-1.f, .89f, (unsigned char*)(("N: " + to_string(N)).c_str()));
    glFlush();
    glutPostRedisplay();
    frame++;
    float deltaTime = (float(clock() - start) / CLOCKS_PER_SEC) * 1000;
    //printf("Delta time: %fms | Frame: %i/%i | render time: %fms\n", deltaTime, saveFrame, T, chrono::duration<float, std::milli>(chrono::high_resolution_clock::now() - renderstart).count());
}
void keyboard(unsigned char key, int x, int y) {
    printf("%i\n", key);

    if (key == 32) {
        cudaDeviceSynchronize();
        pause = 1-pause;
    }
    if (key == 49)
        RenderState = 0;
    if (key == 50)
        RenderState = 1;
    if (key == 51)
        RenderState = 2;
    if (key == 52)
        RenderState = 3;
    if (key == 114)
        resetSim();

    if (key == 43)
        RenderIntens *= 2.f;
    if (key == 45)
        RenderIntens /= 2.f;
}

void mouseClick(int button, int state, int x, int y) {
    if (state == 1) {
        mouseMovement.x = 0.f;
        mouseMovement.y = 0.f;
        mouseButton = 0;
    }
    else {
        mouseButton = button;
    }
    mousePos.x = (float)x / (750*xExpansion);
    mousePos.y = (float)y / (750*yExpansion);
    printf("%i %i: %f %f\n",mouseButton, state, mousePos.x, mousePos.y);
}
void mouseMove(int x, int y) {
    Vector3<float> uv = Vector3<float>((float)x / (750.f * xExpansion), (float)y / (750.f * yExpansion));

    mouseMovement.x = uv.x - mousePos.x;
    mouseMovement.y = uv.y - mousePos.y;

    mousePos = uv;
}


int main(int argc, char* argv[])
{
    cudaSetDevice(0);
    particles = new Particle[N];

    srand(time(NULL));

    cudaHostRegister(particles, pdevsize, cudaHostAllocMapped);
    cudaMalloc(&dparticles, pdevsize);

    resetSim();

    const float gap = 160.f;
    const float slope = 50.f;
    /*Wall* host_walls = new Wall[wallsCount]{
        Wall(make_float2((0.5f+0.2f-0.0f)*bounds.x,(0.2f)*bounds.y),make_float2((0.5f+0.2f + 0.0f) * bounds.x,(0.95f) * bounds.y)),
        Wall(make_float2((0.5f + 0.f - 0.0f) * bounds.x,(0.2f) * bounds.y),make_float2((0.5f + 0.f + 0.0f) * bounds.x,(0.95f) * bounds.y)),
        Wall(make_float2((0.5f + 0.f - 0.0f) * bounds.x,(0.95f) * bounds.y),make_float2((0.5f + 0.2f + 0.0f) * bounds.x,(0.95f) * bounds.y)),

        //Wall(make_float2(-50.f,bounds.y / 2),make_float2(bounds.x/2-gap,bounds.y / 2 + slope)),
        //Wall(make_float2(bounds.x/2+gap,bounds.y / 2 + slope),make_float2(bounds.x+50.f,bounds.y / 2))
    };*/
    cudaMalloc(&d_walls, sizeof(Wall)* wallsCount);
    //cudaMemcpy(d_walls, host_walls, sizeof(Wall)* wallsCount, cudaMemcpyHostToDevice);

    //cudaMalloc(&d_history, sizeof(float2) * T * N);



    int2 blocksCount = CalcLinearBlocksCount();
    //gridFile << "[";
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(750 * xExpansion, 750 * yExpansion);
    glutCreateWindow("A Simple OpenGL Windows Application with GLUT");

    glClearColor(0, 0, 0, 1);
    glPointSize(2.f);
    //glPointSize(750.f/texWid*2+1);
    glutDisplayFunc(display);
    glutKeyboardFunc(&keyboard);
    glutMouseFunc(&mouseClick);
    glutMotionFunc(&mouseMove);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH, GL_NICEST);

    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH, GL_NICEST);

    glutMainLoop();

    return 0;
}