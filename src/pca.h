#pragma once
#include "types.h"

class PCA {
public:
    PCA(unsigned int n_components);

    void fit(const Matrix& X);

    Matrix transform(const SparseMatrix& X);

private:
    size_t _alpha;
    Matrix _base;
};
