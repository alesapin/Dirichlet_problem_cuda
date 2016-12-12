#ifndef _MAT_H_
#define _MAT_H_
#define __volatile volatile // defining __volatile to volatile
#include <tr1/memory>
#include <map>
#include <cmath>

template< typename T >
struct array_deleter
{
  void operator ()( T const * p){
    delete[] p;
  }
};

class Mat;

class Vec {
friend class Mat;
private:
    std::tr1::shared_ptr<double> data;
    long sz;
public:
    Vec():sz(0) {}
    Vec(long size):sz(size) {
        data = std::tr1::shared_ptr<double>(new double[size], array_deleter<double>());
    }
    Vec(long size, double val):sz(size) {
        data = std::tr1::shared_ptr<double>(new double[size], array_deleter<double>());
        for(int i = 0; i < size; ++i){
            data.get()[i]= val;
        }
    }
    Vec(double *data, long size): sz(size) {
        this->data = std::tr1::shared_ptr<double>(data);
    }
    Vec(std::tr1::shared_ptr<double> dt, long size):sz(size),data(dt){}
    Vec(const Vec& other) {
        sz = other.sz;
        data = other.data;
    }
    Vec &operator=(const Vec& other) {
        sz = other.sz;
        data = other.data;
        return *this;
    }
    double operator() (long index) const {
        return data.get()[index];
    }
    long size() const {
        return sz;
    }
    double *memptr() {
        return data.get();
    }
    const double *memptr() const{
        return data.get();
    }
};

class Mat {
private:
    std::tr1::shared_ptr<double> data;
    long rows;
    long cols;
public:
    Mat():rows(0),cols(0){}
    Mat(long r, long c):rows(r), cols(c) {
        data = std::tr1::shared_ptr<double>(new double[rows * cols], array_deleter<double>());
    }
    Mat(long r, long c, double val):rows(r), cols(c) {
        data = std::tr1::shared_ptr<double>(new double[rows * cols], array_deleter<double>());
        for(long i = 0; i < rows * cols; ++i) {
            data.get()[i] = val;
        }
    }
    Mat(double *data, long r, long c): data(data, array_deleter<double>()), rows(r), cols(c) {}
    Mat(const Mat& other) {
        rows = other.rows;
        cols = other.cols;
        data = std::tr1::shared_ptr<double>(new double[rows * cols], array_deleter<double>());
        for(long i = 0; i < rows; ++i) {
            for(long j = 0; j < cols; ++j) {
                this->operator()(i,j) = other(i,j);
            }
        }
    }
    Mat &operator=(const Mat &other) {
        rows = other.rows;
        cols = other.cols;
        data = std::tr1::shared_ptr<double>(new double[rows * cols], array_deleter<double>());
        for(long i = 0; i < rows; ++i){
            for(long j = 0; j < cols; ++j){
                this->operator()(i,j) = other(i,j);
            }
        }
        return *this;
    }
    double &operator() (long i, long j) {
        return data.get()[i*cols+j];
    }
    double operator() (long i, long j) const {
        return data.get()[i*cols+j];
    }
    double *barememptr() {
        return data.get();
    }
    const double *barememptr() const {
        return data.get();
    }
    long rowsCount() const { return rows; }
    long colsCount() const { return cols; }
    Vec getRow(long index) const { //better to copy
        Vec result(cols);
        for(long i = 0; i < cols; ++i) {
            result.data.get()[i] = data.get()[index*cols + i];
        }
        return result;
    }
    Vec getCol(long index) const { //we have to copy ((
        Vec result(rows);
        for(long i = 0; i < rows; ++i){
            result.data.get()[i] = data.get()[i*cols + index];
        }
        return result;
    }
    Mat getSubmat(long startr, long startc, long rws, long cls) const {
        if(startr + rws > rows){ //trim outofbounds
            rws = rows - startr;
        }
        if(startc + cls > cols) {
            cls = cols - startc;
        }
        Mat result(rws,cls);
        for(long i = 0; i < rws; ++i){
            for(long j = 0; j < cls; ++j) {
                result(i,j) = this->operator()(i + startr, j+startc);
            }
        }
        return result;
    }
    friend std::map<std::pair<int,int>,Mat> split(const Mat& m, long rows, long cols) {
       long rElem = (long)(m.rowsCount() / (double)rows + 1);
       long cElem = (long)(m.colsCount() / (double)cols + 1);
       std::map<std::pair<int,int>, Mat> result;
       for(long i = 0; i < rows; ++i){
           for(long j = 0; j < cols; ++j){
               result[std::pair<int,int>(i,j)] = m.getSubmat(i*rElem, j*cElem, rElem, cElem);
           }
       }
       return result;
    }
};
#endif
