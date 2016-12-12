#include "iter.cuh"


const long BLOCK_SIZE_X = 16;
const long BLOCK_SIZE_Y = 32;
const long BLOCK_SIZE = BLOCK_SIZE_X * BLOCK_SIZE_Y;
const dim3 BLOCK_DIM(BLOCK_SIZE_X, BLOCK_SIZE_Y);

__device__ double func(double x, double y){
    double mult = sqrt(4 + x*y);
    return (x*x + y*y) / (4*mult*mult*mult);
}
namespace {
__device__ double fiveDotScheme(CudaMesh m, long i,long j) {
    double prevX = m.getPointX(i-1);
    double prevY = m.getPointY(j-1);
    double curX = m.getPointX(i);
    double curY = m.getPointY(j);
    double nextX = m.getPointX(i+1);
    double nextY = m.getPointY(j+1);

    double h1 = nextX - curX;
    double h2 = nextY- curY;
    double hprev1 = curX - prevX;
    double hprev2 = curY - prevY;
    double hs1 = (h1 + hprev1) / 2;
    double hs2 = (h2 + hprev2) / 2;

    double leftPoint = m(i, j-1);
    double rightPoint = m(i, j+1);
    double downPoint = m(i+1, j);
    double upPoint = m(i-1, j);
    double ypart =  ((m(i,j) - upPoint)/hprev1 - (downPoint - m(i,j))/h1)/hs1;
    double xpart = ((m(i,j) - leftPoint)/hprev2 - (rightPoint - m(i,j))/h2)/hs2;
    return xpart + ypart;
}

template <unsigned int blockSize>
__device__ void reduce(Fraction *sdata, Fraction *global, long tid, double nom, double denom) {
    sdata[tid].nom = nom;
    sdata[tid].denom = denom;
    __syncthreads();
    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid].nom += sdata[tid + 256].nom;
            sdata[tid].denom += sdata[tid + 256].denom;
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid].nom += sdata[tid + 128].nom;
            sdata[tid].denom += sdata[tid + 128].denom;
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid].nom += sdata[tid + 64].nom;
            sdata[tid].denom += sdata[tid + 64].denom;
        }
        __syncthreads();
    }
    if (tid < 32) {
        if (blockSize >= 64) {
            sdata[tid].nom += sdata[tid + 32].nom;
            sdata[tid].denom += sdata[tid + 32].denom;
        }
        __syncthreads();
        if (blockSize >= 32) {
            sdata[tid].nom += sdata[tid + 16].nom;
            sdata[tid].denom += sdata[tid + 16].denom;
        }
        __syncthreads();
        if (blockSize >= 16) {
            sdata[tid].nom += sdata[tid + 8].nom;
            sdata[tid].denom += sdata[tid + 8].denom;
        }
        __syncthreads();
        if (blockSize >= 8) {
            sdata[tid].nom += sdata[tid + 4].nom;
            sdata[tid].denom += sdata[tid + 4].denom;
        }
        __syncthreads();
        if (blockSize >= 4) {
            sdata[tid].nom += sdata[tid + 2].nom;
            sdata[tid].denom += sdata[tid + 2].denom;
        }
        __syncthreads();
        if (blockSize >= 2) {
            sdata[tid].nom += sdata[tid + 1].nom;
            sdata[tid].denom += sdata[tid + 1].denom;
        }
        __syncthreads();
    }
    if (tid == 0) {
        long blockIndex = gridDim.x * blockIdx.y + blockIdx.x;
        global[blockIndex].nom = sdata[0].nom;
        global[blockIndex].denom = sdata[0].denom;
    }
}

__global__ void kernelR(CudaMesh rMesh, CudaMesh pMesh) {
    long i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    long j = blockDim.y * blockIdx.y + threadIdx.y + 1;
    if (i < rMesh.rowsCount() - 1 && i > 0 &&
            j < rMesh.colsCount() - 1 && j > 0) {
        rMesh(i,j) = fiveDotScheme(pMesh, i, j) -
            func(pMesh.getPointX(i), pMesh.getPointY(j));
    }
}

__global__ void kernelG(CudaMesh gMesh, CudaMesh rMesh, double alpha ) {
    long i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    long j = blockDim.y * blockIdx.y + threadIdx.y + 1;
    if (i < rMesh.rowsCount() - 1 && i > 0 &&
            j < rMesh.colsCount() - 1 && j > 0) {
        gMesh(i, j) = rMesh(i, j) - alpha*gMesh(i,j);
    }
}

__global__ void kernelP(CudaMesh pMesh, CudaMesh gMesh, double tau, Fraction *erros) {
    extern __shared__ Fraction shared[];

    long i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    long j = blockDim.y * blockIdx.y + threadIdx.y + 1;
    long tid = threadIdx.y * BLOCK_SIZE_X + threadIdx.x;

    double err = 0;
    if (i < pMesh.rowsCount() - 1 && i > 0 &&
            j < pMesh.colsCount() - 1 && j > 0) {
        double val = pMesh(i,j) - tau*gMesh(i,j);
        double errV = pMesh(i,j) - val;
        double hsX = pMesh.getHShtrX(i);
        double hsY = pMesh.getHShtrY(j);
        err = errV*errV*hsX*hsY;
        pMesh(i, j) = val;
    }
    reduce<BLOCK_SIZE>(shared, erros, tid, err, 0); //dirty hack
}

__global__ void kernelCalcTau(CudaMesh gMesh, CudaMesh rMesh, Fraction *taus) {
    extern __shared__ Fraction shared[];
    long i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    long j = blockDim.y * blockIdx.y + threadIdx.y + 1;
    long tid = threadIdx.y * BLOCK_SIZE_X + threadIdx.x;
    double tnumerator = 0, tdenumerator = 0;
    if (i < rMesh.rowsCount() - 1 && i > 0 &&
            j < rMesh.colsCount() - 1 && j > 0) {
        tnumerator = gMesh(i,j) * rMesh(i,j);
        tdenumerator = fiveDotScheme(gMesh, i, j) * gMesh(i, j);
    }
    reduce<BLOCK_SIZE>(shared, taus, tid, tnumerator, tdenumerator);
}
__global__ void kernelCalcAlpha(CudaMesh rMesh, CudaMesh gMesh, Fraction *alphas) {
    extern __shared__ Fraction shared[];
    long i = blockDim.x * blockIdx.x + threadIdx.x +1 ;
    long j = blockDim.y * blockIdx.y + threadIdx.y +1;
    double anumerator = 0, adenumerator = 0;
    long tid = threadIdx.y * BLOCK_SIZE_X + threadIdx.x;
    if (i < rMesh.rowsCount() - 1 && i > 0 &&
            j < rMesh.colsCount() - 1 && j > 0) {
        anumerator = fiveDotScheme(rMesh, i, j) * gMesh(i, j);
        adenumerator = fiveDotScheme(gMesh, i, j) * gMesh(i, j);
    }
    reduce<BLOCK_SIZE>(shared, alphas, tid, anumerator, adenumerator);
}
}

Fraction CudaIterator::collectBlocks(Fraction *numbers, long size) const {
    std::tr1::shared_ptr<Fraction> localData = std::tr1::shared_ptr<Fraction>(new Fraction[size], array_deleter<Fraction>());
    CHECK (
        cudaMemcpy(localData.get(), numbers, size*sizeof(Fraction), cudaMemcpyDeviceToHost);
    );
    Fraction result;
    result.nom = 0;
    result.denom = 0;
    for (long i = 0; i < size; ++i){
        result.nom += localData.get()[i].nom;
        result.denom += localData.get()[i].denom;
    }
    return result;
}

void CudaIterator::calcR(CudaMesh rMesh, CudaMesh pMesh) {
    kernelR<<<gridDim, BLOCK_DIM>>>(rMesh, pMesh);
}
void CudaIterator::calcG(CudaMesh gMesh, CudaMesh rMesh, double alpha) {
    kernelG<<<gridDim, BLOCK_DIM>>>(gMesh, rMesh, alpha);
}
Fraction CudaIterator::calcAlpha(CudaMesh rMesh, CudaMesh gMesh) {
    Fraction *fractions;
    CHECK (
        cudaMalloc(&fractions, gridDim.x*gridDim.y*sizeof(Fraction));
    )
    kernelCalcAlpha<<<gridDim, BLOCK_DIM, BLOCK_SIZE*sizeof(Fraction)>>>(rMesh, gMesh, fractions);
    return collectBlocks(fractions, gridDim.x*gridDim.y);
}
Fraction CudaIterator::calcTau(CudaMesh gMesh, CudaMesh rMesh) {
    Fraction *fractions;
    CHECK (
        cudaMalloc(&fractions, gridDim.x*gridDim.y*sizeof(Fraction));
    )
    kernelCalcTau<<<gridDim, BLOCK_DIM, BLOCK_SIZE*sizeof(Fraction)>>>(gMesh, rMesh, fractions);
    return collectBlocks(fractions, gridDim.x*gridDim.y);
}
double CudaIterator::calcP(CudaMesh pMesh, CudaMesh gMesh) {
    Fraction *fractions;
    CHECK (
        cudaMalloc(&fractions, gridDim.x*gridDim.y*sizeof(Fraction));
    )
    kernelP<<<gridDim, BLOCK_DIM, BLOCK_SIZE*sizeof(Fraction)>>>(pMesh, gMesh, tau, fractions);
    return collectBlocks(fractions, gridDim.x*gridDim.y).nom;
}

double CudaIterator::iterate() {
    double error;
    if(iter == 0) {
        getMeshBorders(pMesh);
        calcR(rMesh, pMesh);
        calcR(gMesh, pMesh);
        getMeshBorders(rMesh);
        getMeshBorders(gMesh);
        Fraction tFr = calcTau(rMesh, rMesh);
        double allNumerator, allDenumerator;
        MPI_Allreduce(&tFr.nom, &allNumerator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&tFr.denom, &allDenumerator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tau = allNumerator/allDenumerator;
        error = 100000;
    } else {
        double localErr = calcP(pMesh, gMesh);
        MPI_Allreduce(&localErr, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        getMeshBorders(pMesh);
        calcR(rMesh, pMesh);

        getMeshBorders(rMesh);
        Fraction aFr = calcAlpha(rMesh, gMesh);
        double allAnumerator, allAdenumerator;
        MPI_Allreduce(&aFr.nom, &allAnumerator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&aFr.denom, &allAdenumerator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double alpha = allAnumerator / allAdenumerator;
        calcG(gMesh, rMesh, alpha);

        getMeshBorders(gMesh);
        Fraction tFr = calcTau(gMesh, rMesh);
        double allNumerator, allDenumerator;
        MPI_Allreduce(&tFr.nom, &allNumerator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&tFr.denom, &allDenumerator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tau = allNumerator/allDenumerator;
    }
    iter++;
    return sqrt(error);
}

void CudaIterator::getMeshBorders(CudaMesh &mesh) {
    MPI_Status status[4];
    MPI_Request send[4];
    MPI_Request recv[4];
    std::tr1::shared_ptr<double> upBuf, downBuf, rightBuf, leftBuf;
    if (up >= 0 && up < size) {
        upBuf = std::tr1::shared_ptr<double>(new double[mesh.colsCount()]);
        Vec upCur(mesh.colsCount());
        mesh.getUp(upCur);
        MPI_Isend(upCur.memptr(), upCur.size(), MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &send[0]);
        MPI_Irecv(upBuf.get(), mesh.colsCount(), MPI_DOUBLE, up, MPI_ANY_TAG, MPI_COMM_WORLD, &recv[0]);

        MPI_Wait(&send[0],&status[0]);
        MPI_Wait(&recv[0],&status[0]);
    }
    if(down >= 0 && down < size) {
        downBuf = std::tr1::shared_ptr<double> (new double[mesh.colsCount()]);
        Vec downCur(mesh.colsCount());
        mesh.getDown(downCur);
        MPI_Isend(downCur.memptr(), downCur.size(), MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &send[1]);
        MPI_Irecv(downBuf.get(), mesh.colsCount(), MPI_DOUBLE, down, MPI_ANY_TAG, MPI_COMM_WORLD, &recv[1]);

        MPI_Wait(&send[1],&status[1]);
        MPI_Wait(&recv[1],&status[1]);
    }
    if (right >= 0 && right < size) {
        rightBuf = std::tr1::shared_ptr<double> (new double[mesh.rowsCount()]);
        Vec rightCur(mesh.rowsCount());
        mesh.getRight(rightCur);
        MPI_Isend(rightCur.memptr(), rightCur.size(), MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &send[2]);
        MPI_Irecv(rightBuf.get(), mesh.rowsCount(), MPI_DOUBLE, right, MPI_ANY_TAG, MPI_COMM_WORLD, &recv[2]);

        MPI_Wait(&send[2],&status[2]);
        MPI_Wait(&recv[2],&status[2]);
    }
    if(left >= 0 && left < size) {
        leftBuf = std::tr1::shared_ptr<double> (new double[mesh.rowsCount()]);
        Vec leftCur(mesh.rowsCount());
        mesh.getLeft(leftCur);
        MPI_Isend(leftCur.memptr(), leftCur.size(), MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &send[3]);
        MPI_Irecv(leftBuf.get(), mesh.rowsCount(), MPI_DOUBLE, left, MPI_ANY_TAG, MPI_COMM_WORLD, &recv[3]);

        MPI_Wait(&send[3],&status[3]);
        MPI_Wait(&recv[3],&status[3]);
    }
    if (upBuf) {
        Vec upR(upBuf, mesh.colsCount());
        mesh.setUp(upR);
    }
    if (downBuf) {
        Vec downR(downBuf, mesh.colsCount());
        mesh.setDown(downR);
    }
    if (rightBuf) {
        Vec rightR(rightBuf, mesh.rowsCount());
        mesh.setRight(rightR);
    }
    if (leftBuf) {
        Vec leftR(leftBuf, mesh.rowsCount());
        mesh.setLeft(leftR);
    }
}
