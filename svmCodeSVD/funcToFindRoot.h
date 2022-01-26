/**
 * @file funcToFindRoot.h
 */
#ifndef FUNC_TO_FIND_ROOT_H 
#define FUNC_TO_FIND_ROOT_H 

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/Core>
#include<iostream>

struct my_f_params { int tmpSize; Eigen::MatrixXd ham; Eigen::MatrixXd tmpEnergy; };
double func_to_find_root_ (double x, void * params);

#endif

