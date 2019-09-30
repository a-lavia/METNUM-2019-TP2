#pragma once

#include "types.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(const SparseMatrix& X, const Vector& y);
    Vector predict(const SparseMatrix& X);

private:

    Vector _distanceToRow(const Vector& row);
    double _predictRow(const Vector& row);

    size_t _k;
    SparseMatrix _X;
    Vector _y;
};
