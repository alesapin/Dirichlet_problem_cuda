#ifndef _CUDA_MESH_CUH_
#define _CUDA_MESH_CUH_
#include "mesh.h"
#include <iostream>
#define CHECK(cudacall) { int err=cudacall; if (err!=cudaSuccess) std::cerr<<"CUDA ERROR "<<err<<" at line "<<__LINE__<<"'s "<<#cudacall<<"\n";}

class CudaMesh {
private:
   cudaPitchedPtr data;
   double *pointCacheX;
   double *pointCacheY;
   PointUi leftDownCorner;
   PointUi rightUpCorner;
   long rowsShift;
   long colsShift;
   long parentRows;
   long parentCols;
   void setEdge(long xFrom, long yFrom, long xSize, long ySize, const double *data);
   void getEdge(double *data, long xFrom, long yFrom, long xSize, long ySize) const;
   __host__ __device__ inline static double f(double t) {
       return (std::pow(1+t, 1.5) - 1) / 1.82842712474;
   }
public:
   CudaMesh(long rows, long cols, MeshConfig conf);
    __device__ double operator()(long i, long j) const {
        return *((double*)((char*)data.ptr + i * data.pitch) + j);
    }
    __device__ double& operator()(long i, long j) {
        return *((double*)((char*)data.ptr + i * data.pitch) + j);
    }

    __device__ double getPointX(long i) const {
        if(!pointCacheX) {
            double dblI = static_cast<double>(i) + rowsShift;
            double xcoeff = dblI / (parentRows - 1);
            return rightUpCorner.first * f(xcoeff);
        } else {
            return pointCacheX[i];
        }
    }
    __device__ double getPointY(long j) const {
        if(!pointCacheY) {
            double dblJ = static_cast<double>(j) + colsShift;
            double ycoeff = dblJ / (parentCols - 1);
            return rightUpCorner.second * f(ycoeff);
        } else {
            return pointCacheY[j];
        }
    }
    __device__ double getHShtrX(long i) const {
        double prevX = getPointX(i-1);
        double curX = getPointX(i);
        double nextX = getPointX(i+1);
        double h1 = nextX - curX; //x[i+1] - x[i]
        double hprev1 = curX - prevX; //x[i] - x[i-1]
        return (h1 + hprev1) / 2; //h'
    }
    __device__ double getHShtrY(long j) const {
        double prevY = getPointY(j-1);
        double curY = getPointY(j);
        double nextY = getPointY(j+1);
        double h2 = nextY - curY; //y[i+1] - y[i]
        double hprev2 = curY - prevY; //y[i] - y[i-1]
        return (h2 + hprev2) / 2; //h'
    }
    void setLeft(const Vec &left, bool shift=false);
    void setRight(const Vec &right, bool shift=false);
    void setUp(const Vec &up, bool shift=false);
    void setDown(const Vec &ldown, bool shift=false);

    void getLeft(Vec &left) const;
    void getRight(Vec &right) const;
    void getUp(Vec &up) const;
    void getDown(Vec &down) const;

    __host__ __device__ long rowsCount() const { return data.xsize; }
    __host__ __device__ long colsCount() const { return data.ysize; }
    __host__ __device__ long getParentRows() const { return parentRows; }
    __host__ __device__ long getParentCols() const { return parentCols; }
    __host__ __device__ long getRowsShift() const { return rowsShift; }
    __host__ __device__ long getColsShift() const { return colsShift; }
    friend void countPointCache(CudaMesh &mesh, int left, int up);
    void toMesh(int fromX, int fromY, Mesh &mesh) const;

};
#endif
