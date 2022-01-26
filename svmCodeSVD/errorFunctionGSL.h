/**
 *   @file errorFunctionGSL.h
 *   
 */
#ifndef ERRORFUNCTIONGSL_H
#define ERRORFUNCTIONGSL_H

#include<gsl/gsl_sf_erf.h>
extern "C" {
// These routines compute the error function erf(x), where erf(x) = (2/\sqrt(\pi)) \int_0^x dt \exp(-t^2). 
    double gsl_sf_erf (double x);
// These routines compute the complementary error function erfc(x) = 1 - erf(x) = (2/\sqrt(\pi)) \int_x^\infty \exp(-t^2). 
    double gsl_sf_erfc (double x);
}
class Erf
{
  public:
    double operator() (double x)
    {
      return gsl_sf_erf(x);
    };
};
class Erfc
{
  public:
    double operator() (double x)
    {
      return gsl_sf_erfc(x);
    };
};
#endif // ERRORFUNCTIONGSL_H
