//
// Created by pachi on 5/6/19.
//

#include <iostream>
#include "pca.h"
#include "eigen.h"
#include "knn.h"

int main(int argc, char** argv){
    /*
    KNNClassifier classifier(2);
    SparseMatrix x(3,4);
    x.insert(0,0)=1;
    x.insert(0,1)=0;
    x.insert(0,2)=0;
    x.insert(0,3)=0;

    x.insert(1,0)=0;
    x.insert(1,1)=1;
    x.insert(1,2)=0;
    x.insert(1,3)=0;

    x.insert(2,0)=0;
    x.insert(2,1)=0;
    x.insert(2,2)=1;
    x.insert(2,3)=0;

    Vector y(3);
    y(0) = 0.;
    y(1) = 0.;
    y(2) = 1.;
    classifier.fit(x,y);

    std::cout << classifier.predict(x) << std::endl;
    */

    /*
    Vector d(5);
    d << 5.0, 4.0, 3.0, 2.0, 1.0;
    Matrix D = d.asDiagonal();

    Matrix v = Matrix::Constant(5,1,1.0);
    v = v / v.norm();

    Matrix B = Matrix::Identity(5,5) - 2.0 *(v*v.transpose());//Householder

    Matrix M = B.transpose() * D * B;

    std::pair<double, Vector> e = power_iteration(M);
    std::cout << e.first << std::endl << e.second << std::endl;

    std::pair<Vector, Matrix> es = get_first_eigenvalues(M, 5);
    std::cout << es.first << std::endl << es.second << std::endl;
    */

    /*
    Matrix m(3,3);
    m << 1,1,1,2,2,2,3,3,3;
    Matrix center = m.rowwise() - m.colwise().mean();
    Matrix cov = (center.transpose() * center) / double(m.rows() - 1);
    std::cout << cov << std::endl;
    std::cout << get_first_eigenvalues(cov, cov.rows()).first << std::endl;
    std::cout << get_first_eigenvalues(cov, cov.rows()).second << std::endl;

    Matrix base = get_first_eigenvalues(cov, cov.rows()).second;
    std::cout << m * base.block(0, 0, base.rows(), 1) << std::endl;
     */
    return 0;
}
