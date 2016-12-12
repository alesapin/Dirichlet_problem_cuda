#include "cuda_mesh.cuh"

CudaMesh::CudaMesh(long rows, long cols, MeshConfig conf):
    leftDownCorner(conf.leftDownCorner),
    rightUpCorner(conf.rightUpCorner),
    rowsShift(conf.rowsShift),
    colsShift(conf.colsShift),
    parentRows(conf.parentRows),
    parentCols(conf.parentCols)
{
    data.xsize = rows;
    data.ysize = cols;
    CHECK(
        cudaMallocPitch(&data.ptr, &data.pitch, data.ysize*sizeof(double), data.xsize)
    );
    CHECK(
        cudaMemset2D(data.ptr, data.pitch, 0, data.ysize*sizeof(double), data.xsize);
    );
}

void CudaMesh::setEdge(long xFrom, long yFrom, long xSize, long ySize, const double *ptr) {
    CHECK (
        cudaMemcpy2D(
            (char*)data.ptr + yFrom*sizeof(double) + xFrom*data.pitch,
            data.pitch,
            ptr,
            xSize*sizeof(double),
            xSize*sizeof(double),
            ySize,
            cudaMemcpyHostToDevice
        )
    );
}

void CudaMesh::setLeft(const Vec &left, bool shift) {
    setEdge(shift, 0, 1, left.size(), left.memptr());
}
void CudaMesh::setRight(const Vec &right, bool shift) {
    setEdge(shift, colsCount()-1, 1, right.size(), right.memptr());
}
void CudaMesh::setUp(const Vec &up, bool shift) {
    setEdge(0, shift, up.size(), 1, up.memptr());
}
void CudaMesh::setDown(const Vec &down, bool shift) {
    setEdge(rowsCount() - 1, shift, down.size(), 1, down.memptr());
}

void CudaMesh::getEdge(double *ptr, long xFrom, long yFrom, long xSize, long ySize) const {
    CHECK (
        cudaMemcpy2D(
            ptr,
            xSize * sizeof(double),
            (char*)data.ptr + data.pitch * xFrom + yFrom * sizeof(double),
            data.pitch,
            xSize * sizeof(double),
            ySize,
            cudaMemcpyDeviceToHost
        )
    );
}

void CudaMesh::getLeft(Vec &left) const {
    getEdge(left.memptr(), 0, 1, 1, left.size());
}
void CudaMesh::getRight(Vec &right) const {
    getEdge(right.memptr(), 0, colsCount()-2, 1, right.size());
}
void CudaMesh::getUp(Vec &up) const {
    getEdge(up.memptr(), 1, 0, up.size(), 1);
}
void CudaMesh::getDown(Vec &down) const  {
    getEdge(down.memptr(), rowsCount()-2, 0, down.size(), 1);
}
void countPointCache(CudaMesh &mesh, int left, int up) {
    cudaMalloc(&mesh.pointCacheX, mesh.rowsCount()*sizeof(double));
    cudaMalloc(&mesh.pointCacheY, mesh.colsCount()*sizeof(double));
    double localX[mesh.rowsCount()];
    double localY[mesh.colsCount()];
    for(int i = 0; i < mesh.rowsCount(); ++i) {
        double dblI = static_cast<double>(i-up) + mesh.rowsShift;
        double xcoeff = dblI / (mesh.parentRows - 1);
        localX[i] = mesh.rightUpCorner.first * mesh.f(xcoeff);
    }
    for(int j = 0; j < mesh.colsCount(); ++j){
        double dblJ = static_cast<double>(j-left) + mesh.colsShift;
        double ycoeff = dblJ / (mesh.parentCols - 1);
        localY[j]= mesh.rightUpCorner.second * mesh.f(ycoeff);
    }
    CHECK(
        cudaMemcpy(mesh.pointCacheX, localX, mesh.rowsCount()*sizeof(double), cudaMemcpyHostToDevice)
    );
    CHECK(
        cudaMemcpy(mesh.pointCacheY, localY, mesh.colsCount()*sizeof(double), cudaMemcpyHostToDevice);
    );
}

void CudaMesh::toMesh(int fromX, int fromY, Mesh &mesh) const {
   double *resptr = mesh.getData();
   getEdge(resptr, fromX, fromY, mesh.getColumns(), mesh.getRows());
}

