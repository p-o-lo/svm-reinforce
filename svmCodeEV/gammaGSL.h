/**
 *   @file gammaGSL.h
 *   
 */
#ifndef GAMMAGSL_H
#define GAMMAGSL_H

#include<gsl/gsl_sf_gamma.h>
extern "C" {
    double gsl_sf_gamma(double x);
    double gsl_sf_lngamma(double x);
    double gsl_sf_fact (unsigned int n);
    double gsl_sf_doublefact (unsigned int n);
}
/**
 * @brief Interface to the gamma function \f$ \Gamma(x) = \int_0^\infty dt
 *        \,t^{x-1} \exp(-t) \f$ implemented in the Gnu Scientific Library.
 *
 *  These routines compute the Gamma function \f$ \Gamma(x)\f$, subject to \f$
 *  x\f$ not being a negative integer or zero. The function is computed using
 *  the real Lanczos method. The maximum value of \f$ x\f$ such that
 *  \f$\Gamma(x)\f$ is not considered an overflow is given by the macro
 *  GSL_SF_GAMMA_XMAX and is 171.0.
 */
class Gamma
{
  public:
    /**
     * @brief Interface to the gsl gamma function implemented in the GSL library.
     *
     * @param x The variable \f$ x \f$
     *
     * @return The value of \f$ \Gamma(x)\f$
     */
    double operator() (double x)
    {
      return gsl_sf_gamma (x);
    };
};
/**
 * @brief Interface to the gamma function \f$ \log(\Gamma(x)) \f$
 *        implemented in the Gnu Scientific Library.
 *
 * These routines compute the logarithm of the Gamma function, \f$
 * \log(\Gamma(x))\f$ , subject to x not being a negative integer or zero. For
 * \f$ x<0\f$ the real part of \f$\log(\Gamma(x))\f$ is returned, which is
 * equivalent to\f$ \log(|\Gamma(x)|)\f$. The function is computed using the
 * real Lanczos method.
 */
class LnGamma
{
  public:
    /**
     * @brief Interface to the gsl logarithm of gamma function implemented in the GSL library.
     *
     * @param x The variable \f$ x \f$
     *
     * @return The value of \f$ \log(\Gamma(x))\f$
     */
    double operator() (double x)
    {
      return gsl_sf_lngamma (x);
    };
};

/*
 * @brief Interface to the factorial function \f$ n!\f$
 *        implemented in the Gnu Scientific Library.
 *
 * These routines compute the factorial n!. The factorial is related to the
 * Gamma function by n! = \Gamma(n+1). The maximum value of n such that n! is
 * not considered an overflow is given by the macro GSL_SF_FACT_NMAX and is
 * 170.
 */
class Factorial
{
  public:
    double operator()(unsigned int n)
    {
      return gsl_sf_fact(n);
    }
};

/*
 * @brief Interface to the double factorial function \f$ n!!\f$
 *        implemented in the Gnu Scientific Library.
 * These routines compute the double factorial n!! = n(n-2)(n-4) \dots. The
 * maximum value of n such that n!! is not considered an overflow is given by
 * the macro GSL_SF_DOUBLEFACT_NMAX and is 29
 *
 */
class DoubleFactorial
{
  public:
    double operator()(unsigned int n)
     {
       return gsl_sf_doublefact(n);
     }
};

#endif // GAMMAGSL_H
