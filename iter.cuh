#ifndef _KERNELS_CUH_
#define _KERNELS_CUH_
#include "cuda_mesh.cuh"
#include "mesh.h"
#include <mpi.h>

extern const long BLOCK_SIZE_X;
extern const long BLOCK_SIZE_Y;
extern const long BLOCK_SIZE;
extern const dim3 BLOCK_DIM;

struct Fraction {
    double nom;
    double denom;
};

static inline int getColsAdd(int left, int right) {
    return (left!=-1) + (right!=-1);
}
static inline int getRowsAdd(int up, int down) {
    return (up!=-1) + (down!=-1);
}

class CudaIterator {
private:
    int left, right, up, down;
    CudaMesh rMesh;
    CudaMesh gMesh;
    CudaMesh pMesh;
    double tau;
    int rank, size;
    int iter;
    dim3 gridDim;

    double zeroIteration(CudaMesh &pMesh);
    void getMeshBorders(CudaMesh &mesh);
    Fraction collectBlocks(Fraction *numbers, long size) const;
    void calcG(CudaMesh gMesh, CudaMesh rMesh, double alpha);
    void calcR(CudaMesh rMesh, CudaMesh pMesh);
    double calcP(CudaMesh pMesh, CudaMesh gMesh);
    Fraction calcAlpha(CudaMesh rMesh, CudaMesh gMesh);
    Fraction calcTau(CudaMesh gMesh, CudaMesh rMesh);
    double zeroIteration();
public:
    double iterate();
    CudaIterator(dim3 grid, const Mesh& m, int rank, int left, int right, int up, int down, int size):
        gridDim(grid),
        rMesh(m.getRows() + getRowsAdd(up,down), m.getColumns() + getColsAdd(left, right), m.getMeshConfig()),
        pMesh(m.getRows() + getRowsAdd(up,down), m.getColumns() + getColsAdd(left, right), m.getMeshConfig()),
        gMesh(m.getRows() + getRowsAdd(up,down), m.getColumns() + getColsAdd(left, right), m.getMeshConfig()),
        rank(rank),left(left),right(right),up(up),down(down),size(size), iter(0)
    {
        countPointCache(pMesh, left!=-1, up !=-1);
        countPointCache(rMesh, left!=-1, up !=-1);
        countPointCache(gMesh, left!=-1, up !=-1);
        if (left == -1) {
            pMesh.setLeft(m.getLeftCol(), up!=-1);
        }
        if (right == -1) {
            pMesh.setRight(m.getRightCol(), up!=-1);
        }
        if (up == -1) {
            pMesh.setUp(m.getUpRow(),left!=-1);
        }
        if (down == -1){
            pMesh.setDown(m.getDownRow(),left!=-1);
        }
    }
    double getCurrentIter() const { return iter; }
    void getPMesh(Mesh &mesh) const {
        pMesh.toMesh(left!=-1, up!=-1, mesh);
    }
};
#endif
