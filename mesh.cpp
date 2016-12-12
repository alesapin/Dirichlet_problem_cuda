#include "mesh.h"
#include <iostream>
#include <iomanip>

namespace {
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
}

const double Mesh::COEFF = 1.5;
const double Mesh::COEFF_DENOM = 1.82842712474;
Mesh::Mesh(long rows, long cols, MeshConfig conf):
    leftDownCorner(conf.leftDownCorner),
    rightUpCorner(conf.rightUpCorner),
    rowsShift(conf.rowsShift),
    colsShitf(conf.colsShift),
    parentRows(conf.parentRows),
    parentCols(conf.parentCols),
    data(rows, cols, 0)
    {
    initPointCache();
}

Mesh::Mesh(long rows, long cols, double *data, MeshConfig conf):
    leftDownCorner(conf.leftDownCorner),
    rightUpCorner(conf.rightUpCorner),
    rowsShift(conf.rowsShift),
    colsShitf(conf.colsShift),
    parentRows(conf.parentRows),
    parentCols(conf.parentCols),
    data(data, rows, cols) {
    initPointCache();
}
Mesh::Mesh(const Mat &data, MeshConfig conf):
    leftDownCorner(conf.leftDownCorner),
    rightUpCorner(conf.rightUpCorner),
    rowsShift(conf.rowsShift),
    colsShitf(conf.colsShift),
    parentRows(conf.parentRows),
    parentCols(conf.parentCols),
    data(data) {
    initPointCache();
}

void Mesh::initPointCache() {
    pointCache.clear();
    for(long i = 0; i < data.rowsCount()+2; ++i ) {
        pointCache.push_back(std::vector<PointD>(data.colsCount()+2));
    }
    for(long i = 0; i < data.rowsCount()+2; ++i){
        for(long j = 0; j < data.colsCount()+2; ++j){
            double dblI = static_cast<double>(i-1) + rowsShift;
            double dblJ = static_cast<double>(j-1) + colsShitf;
            double xcoeff = dblI / (getParentRows() - 1);
            double ycoeff = dblJ / (getParentCols() - 1);
            double x = rightUpCorner.first * f(xcoeff);
            double y = rightUpCorner.second * f(ycoeff);
            pointCache[i][j] = PointD(x,y);
        }
    }
}
PointD Mesh::getPoint(long i, long j) const {
    return pointCache[i+1][j+1];
}

std::ostream &operator<< (std::ostream &os, const Mesh& m) {
    for(int i = 0; i<m.getRows(); ++i){
        for(int j = 0; j < m.getColumns(); ++j){
            os <<std::setw(10) << m(i,j) << "\t";
        }
        os << "\n";
    }
    return os;
}
void initMeshBoundaries(Mesh &mesh, Function f) {
    for (std::size_t i = 0; i < mesh.getRows() ; ++i ){
        mesh(i,0) = f(mesh.getPoint(i,0));
        mesh(i, mesh.getColumns() - 1) = f(mesh.getPoint(i, mesh.getColumns() - 1));
    }
    for (std::size_t j = 0; j < mesh.getColumns(); ++j) { mesh(0,j) = f(mesh.getPoint(0,j));
        mesh(mesh.getRows() - 1,j) = f(mesh.getPoint(mesh.getRows() - 1, j));
    }
}

std::map<int, Mesh> splitMesh(const Mesh &mesh, long procPower) {
    PointUi size = splitFunction(mesh.getRows(), mesh.getColumns(), procPower);
    long rows = 1 << size.first;
    long cols = 1 << size.second;
    std::map<int, Mesh> result;
    std::map<std::pair<int,int>, Mat> submats = split(mesh.data, rows, cols);
    int procnum = 0;
    long shiftRows = -1, shiftCols = -1;
    for(std::map<std::pair<int,int>,Mat>::iterator it = submats.begin(); it != submats.end(); ++it) {
       int r = it->first.first;
       int c = it->first.second;
       if(shiftRows == -1){
           shiftRows = it->second.rowsCount();
           shiftCols = it->second.colsCount();
       }
       MeshConfig conf = { mesh.leftDownCorner, mesh.rightUpCorner, r*shiftRows, c*shiftCols, mesh.getRows(), mesh.getColumns() };
       result[procnum++] = Mesh(it->second, conf);
    }
    return result;
}

std::ostream &dropToStream(std::ostream& os, const Mesh &mesh) {
    for (long i = 0; i< mesh.getRows(); ++i){
        for(long j = 0; j < mesh.getColumns(); ++j) {
            PointD p = mesh.getPoint(i, j);
            double val = mesh(i,j);
            os << p.first << "\t" << p.second << "\t" << val <<"\n";
        }
    }
    return os;
}

Mesh collectMesh(const std::map<int, Mesh> &submeshs) {
    Mesh first = submeshs.at(0);
    MeshConfig conf = { first.getLeftDownCorner(), first.getRightUpCorner(), 0, 0, first.getParentRows(), first.getParentCols() };
    Mesh result(first.getParentRows(), first.getParentCols(), conf);
    for(std::map<int, Mesh>::const_iterator itr = submeshs.begin(); itr!=submeshs.end(); ++itr){
        for(long i = 0; i < itr->second.getRows(); ++i) {
            for(long j = 0; j < itr->second.getColumns(); ++j) {
                result(i+itr->second.getRowsShift(), j + itr->second.getColumnsShift()) = itr->second(i,j);
            }
        }
    }
    return result;
}
