/**
 * @file funcToFindRoot.cc
 */
#include "basis.h"
#include "funcToFindRoot.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>

double func_to_find_root_ (double x, void * p) {

	my_f_params * params = (struct my_f_params *)p;
	int tmpSize = (params->tmpSize);
	double ret;
	Eigen::MatrixXd ham = (params->ham);
	Eigen::MatrixXd tmpEnergy  = (params->tmpEnergy);
	
        double tmp = 0;
	if(tmpSize == 0) {
		ret = x - ham(tmpSize,tmpSize);
	}
	else {
		for(int i=0; i < tmpSize; i++) {
        	        tmp += ham(i,tmpSize)*(ham(i,tmpSize))/(x - tmpEnergy(i,0));
        	}
		ret = x - ham(tmpSize,tmpSize) - tmp;
	}

	return ret;
}
