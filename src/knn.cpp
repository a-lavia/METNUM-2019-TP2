#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"

using namespace std;


KNNClassifier::KNNClassifier(unsigned int n_neighbors) : _k(n_neighbors)
{
}

void KNNClassifier::fit(const SparseMatrix& X, const Vector& y)
{
    _X = X;
    _y = y;
}


/**
* Devuelve un vector con la distancia al cuadrado de 'row' con respecto a cada elemento del entrenamiento
*/
Vector KNNClassifier::_distanceToRow(const RowVector& row)
{
    auto ret = Vector(_X.rows());
    for (Eigen::Index i = 0; i < _X.rows(); ++i)
    {
        auto diff = _X.row(i) - row;
        ret(i) = diff.squaredNorm();
    }
    return ret;
}

/**
* Predice la clasificacion utilizando KNN
* Devuelve 1.0 (positiva), 0.0 (negativa)
*/
double KNNClassifier::_predictRow(const RowVector& row)
{
    auto dist = _distanceToRow(row);
    
    size_t n = 0;
    IntVector neighbors(_X.rows());
    std::generate(std::begin(neighbors), std::end(neighbors), [&n]{return n++; });
    std::sort(std::begin(neighbors), std::end(neighbors), [&dist](int i1, int i2){return dist(i1) < dist(i2); });
    double kNeighborsScore = 0.0;
    for(size_t i = 0; i < _k; ++i) {
        kNeighborsScore += _y(neighbors(i));
    }

    double res = kNeighborsScore >= _k * 0.5 ? 1.0 : 0.0;
    return res;
}

Vector KNNClassifier::predict(const SparseMatrix& X)
{
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (Eigen::Index i = 0; i < X.rows(); ++i)
    {
        ret(i) = _predictRow(X.row(i));
    }

    return ret;
}
