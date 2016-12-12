#include "iter.cuh"
#include "mesh.h"
#include <mpi.h>
#include <getopt.h>
#include <cstring>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <cmath>

double phi(PointD p){
    double x = p.first;
    double y = p.second;
    return sqrt(4 + x*y);
}
PointUi splitFunction(int N0, int N1, int p) {
    double n0, n1;
    int p0, i;

    n0 = (double) N0; n1 = (double) N1;
    p0 = 0;

    for(i = 0; i < p; i++) {
        if(n0 > n1) {
            n0 = n0 / 2.0;
            ++p0;
        } else {
            n1 = n1 / 2.0;
        }
    }
    return PointUi(p0, p-p0);
}

int getPowOfTwo(int val){
    int pwr = 0;
    while(val >>= 1) ++pwr;
    return pwr;
}

struct Params {
    long a;
    long b;
    long rows;
    long cols;
    std::string fname;
};

Params getParams(int argc, char** argv) {
      int c;
      char *opt_val = NULL;
      Params result;
      while ((c = getopt (argc, argv, "a:b:r:c:f:")) != -1) {
          switch(c) {
            case 'a':
                opt_val = optarg;
                result.a = strtol(opt_val, NULL, 10);
                break;
            case 'b':
                opt_val = optarg;
                result.b = strtol(opt_val, NULL, 10);
                break;
            case 'r':
                opt_val = optarg;
                result.rows = strtol(opt_val, NULL, 10);
                break;
            case 'c':
                opt_val = optarg;
                result.cols = strtol(opt_val, NULL, 10);
                break;
            case 'f':
                opt_val = optarg;
                result.fname = std::string(opt_val);
                break;
          }
      }
      return result;
}


const double EPSILON = 0.0001;

int main(int argc, char **argv) {
    int size, rank;
    Params pars = getParams(argc, argv);
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    cudaSetDevice(0);
    long A = pars.a, B = pars.b;
    int totalRows = pars.cols, totalCols = pars.rows;
    long rowsShift, colsShift;
    long rows, cols;
    MPI_Status status;
    double start;
    int sizePower = getPowOfTwo(size);
    PointUi ps = splitFunction(totalRows, totalCols, sizePower);
    int procRows =  1 << ps.first, procCols = 1<< ps.second;
    std::map<int, Mesh> splited;
    if (rank == 0) {
        start = MPI_Wtime();
        MeshConfig globalConf = {PointUi(0,0), PointUi(A,B), 0, 0, totalRows, totalCols};
        Mesh result(totalRows, totalCols, globalConf);
        initMeshBoundaries(result, phi);
        splited = splitMesh(result, sizePower);
        for(std::map<int, Mesh>::iterator itr = splited.begin(); itr != splited.end(); ++itr) {
            long r = itr->second.getRows();
            long c = itr->second.getColumns();
            long rShift = itr->second.getRowsShift();
            long cShift = itr->second.getColumnsShift();
            if(itr->first != rank) {
                MPI_Send(&r, 1, MPI_LONG, itr->first, 0, MPI_COMM_WORLD);
                MPI_Send(&c, 1, MPI_LONG, itr->first, 0, MPI_COMM_WORLD);
                MPI_Send(&rShift, 1, MPI_LONG, itr->first, 0, MPI_COMM_WORLD);
                MPI_Send(&cShift, 1, MPI_LONG, itr->first, 0, MPI_COMM_WORLD);
            } else {
                rows = r;
                cols = c;
                rowsShift = rShift;
                colsShift = cShift;
            }
        }
    } else {
        MPI_Recv(&rows, 1, MPI_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&cols, 1, MPI_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&rowsShift, 1, MPI_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&colsShift, 1, MPI_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
    Mesh curMesh;
    if (rank == 0) {
        for(std::map<int, Mesh>::iterator itr = splited.begin(); itr != splited.end(); ++itr) {
            if(itr->first != rank) {
                MPI_Send(itr->second.getData(), itr->second.getRows()*itr->second.getColumns(),
                        MPI_DOUBLE, itr->first, 0, MPI_COMM_WORLD);
            }
        }
        curMesh = splited[0];
    } else {
        double *recdata = new double[rows*cols];
        MPI_Recv(recdata, rows*cols, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MeshConfig conf = { PointUi(0,0), PointUi(A,B),rowsShift, colsShift, totalRows, totalCols };
        curMesh = Mesh(rows, cols, recdata, conf);
    }
    int procCol = rank%procCols;
    int procRow = (rank - procCol) / procCols;

    long left = procCol - 1 >=0 ? procRow*procCols + procCol - 1 : -1;
    long right = procCol + 1 < procCols ? procRow*procCols + procCol + 1 : -1;
    long up = procRow - 1 >=0? (procRow - 1)*procCols + procCol : -1;
    long down = procRow + 1 < procRows ? (procRow + 1)*procCols + procCol : -1;

    MPI_Barrier(MPI_COMM_WORLD);
    dim3 gridDim;
    gridDim.x = (int)((curMesh.getRows() + 2) / BLOCK_SIZE_X + 1);
    gridDim.y = (int)((curMesh.getColumns() + 2) / BLOCK_SIZE_Y + 1);

    CudaIterator iter(gridDim, curMesh, rank, left, right, up, down, size);
    double err = iter.iterate();
    int iterCount = 1;
    while(err > EPSILON) {
        err = iter.iterate();
        if (rank == 0) {
            std::cout <<"Iteration: " << iterCount++ <<" Error: " << err <<"\n";
        }
    }
    iter.getPMesh(curMesh);
    if (rank != 0) {
        MPI_Send(&rows, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&cols, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&rowsShift, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&colsShift, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
        MPI_Send(curMesh.getData(), rows*cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        std::map<int, Mesh> submeshs;
        submeshs[0] = curMesh;
        std::vector<MPI_Request> requests;
        for (int i = 1; i < size; ++i ) {
            MPI_Recv(&rows, 1, MPI_LONG, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&cols, 1, MPI_LONG, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&rowsShift, 1, MPI_LONG, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&colsShift, 1, MPI_LONG, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            double *recIdata = new double[rows*cols];
            MPI_Recv(recIdata, rows*cols, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MeshConfig conf = { PointUi(0,0), PointUi(A,B), rowsShift, colsShift,  totalRows, totalCols };
            Mesh curMesh(rows, cols, recIdata, conf);
            submeshs[i] = curMesh;
        }
        Mesh result = collectMesh(submeshs);
        double elapsed = MPI_Wtime() - start;
        std::ofstream ofs(pars.fname.c_str());
        dropToStream(ofs, result);
        ofs <<"stats:" <<iterCount <<'\t' <<totalRows << '\t' << totalCols << '\t' << elapsed;
    }

    MPI_Finalize();
    return 0;
}
