#pragma once

#include "types.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(const Matrix& X, const Vector& y);
    Vector predict(const Matrix& X);

private:

    Vector _distanceToRow(const RowVector& row);
    double _predictRow(const RowVector& row);

    size_t _k;
    Matrix _X;
    Vector _y;
};
