#ifndef _MESH_H_
#define _MESH_H_
#include <cmath>
#include <map>
#include <cfloat>
#include <vector>
#include "mat.h"

typedef std::pair<std::size_t, std::size_t> PointUi;
typedef std::pair<double, double> PointD;
static const PointD NAN_POINT(DBL_MAX, DBL_MAX);
typedef double (*Function)(PointD);
struct MeshConfig {
    PointUi leftDownCorner;
    PointUi rightUpCorner;
    long rowsShift;
    long colsShift;
    long parentRows;
    long parentCols;
};

class Mesh {
    private:
        Mat data;
        PointUi leftDownCorner;
        PointUi rightUpCorner;
        std::vector<std::vector<PointD> >  pointCache;
        long rowsShift, colsShitf, parentRows, parentCols;
        static const double COEFF;
        static const double COEFF_DENOM;
        inline static double f(double t) {
            return (std::pow(1+t, COEFF) - 1) / COEFF_DENOM;
        }
        PointD countPoint(long i, long j) const;
        void initPointCache();
    public:
        Mesh(long rows, long cols, MeshConfig conf);
        Mesh(long rows, long cols, double *data, MeshConfig conf);
        Mesh(const Mat &data, MeshConfig conf);
        Mesh():leftDownCorner(0,0), rightUpCorner(0,0), rowsShift(0), colsShitf(0), parentRows(0),parentCols(0){}
        Mesh(const Mesh &other):
            data(other.data),
            leftDownCorner(other.leftDownCorner),
            rightUpCorner(other.rightUpCorner),
            rowsShift(other.rowsShift),
            colsShitf(other.colsShitf),
            parentRows(other.parentRows),
            parentCols(other.parentCols),
            pointCache(other.pointCache)
        {
        }
        Mesh &operator=(const Mesh &other) {
            data = other.data;
            leftDownCorner = other.leftDownCorner;
            rightUpCorner = other.rightUpCorner;
            rowsShift = other.rowsShift;
            colsShitf = other.colsShitf;
            parentRows = other.parentRows;
            parentCols = other.parentCols;
            pointCache = other.pointCache;
            return *this;
        }
        MeshConfig getMeshConfig() const {
            MeshConfig result = {
                leftDownCorner,
                rightUpCorner,
                rowsShift,
                colsShitf,
                parentRows,
                parentCols
            };
            return result;
        }
        PointD getPoint(long i, long j) const;
        double operator()(long i, long j) const {
            return data(i,j);
        }
        double &operator()(long i, long j) { return data(i,j); }
        long getRows() const { return data.rowsCount(); }
        long getColumns() const { return data.colsCount(); }
        Vec getUpRow() const {return data.getRow(0);}
        Vec getDownRow() const {return data.getRow(data.rowsCount() - 1);}
        Vec getLeftCol() const {return data.getCol(0);}
        Vec getRightCol() const {return data.getCol(data.colsCount() - 1);}
        double *getData() {return data.barememptr();}
        const double *getData() const {return data.barememptr();}
        friend void initMeshBoundaries(Mesh &mesh, Function phi);
        friend std::map<int, Mesh> splitMesh(const Mesh &mesh, long procnum);
        friend Mesh collectMesh(const std::map<int, Mesh> &submeshs);
        PointUi getLeftDownCorner() const {return leftDownCorner; }
        PointUi getRightUpCorner() const {return rightUpCorner; }
        long getRowsShift() const { return rowsShift; }
        long getColumnsShift() const {return colsShitf; }
        long getParentRows() const {return parentRows; }
        long getParentCols() const {return parentCols; }
        friend std::ostream &operator<< (std::ostream &os, const Mesh& m);
        friend std::ostream &dropToStream(std::ostream& os, const Mesh &mesh);
};
#endif
