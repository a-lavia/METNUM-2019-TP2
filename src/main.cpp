//
// Created by pachi on 5/6/19.
//

#include <iostream>
#include "pca.h"
#include "eigen.h"
#include "knn.h"

int main(int argc, char** argv){

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

    return 0;
}
