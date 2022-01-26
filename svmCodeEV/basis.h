/**
 * @file basis.h
 */
#ifndef BASIS_H 
#define BASIS_H 

#include"coordinates.h"
#include"inputData.h"
#include"gammaGSL.h"

#include<vector>
#include<list>

#include <Eigen/Eigenvalues> 
#include <Eigen/LU>
#include <Eigen/Core>
#include <random>
#include<iostream>
#include<fstream>

class
Basis
{
  typedef 
      std::tuple<Eigen::VectorXd, // eigenvalues
                 std::pair<Eigen::MatrixXd,Eigen::VectorXd>, // <a,u>
                 Eigen::VectorXd, // alphas
                 Eigen::VectorXd,//spin
                 Eigen::VectorXd> // isospin

       allVariables; 

  private:
   int np_; // the number of particles 
   int factorialNp_ ;
   InputData & indata_;
   Coordinates & coo_; 

   int state_=0; // the eigenvalues to calculate

   int k_ ; // the current dimension of the basis
   
   // we fix the quantum numbers; for the moment
   // L is common to all the basis elements
   int L_ ;
   // Total spin
   double S_;
   // Total isospin
   double T_;

   size_t maxDimension_ ; // the max dimension of the basis

   // the vector that contains the correlation of the 
   // Gaussian basis
   // |A> \prop exp(-alpha_*rij+ ...)
   std::list<Eigen::VectorXd> alpha_;

   // the Gaussian basis |A>. The matrix is symmetric and
   // positive 
   std::list<std::pair<Eigen::MatrixXd,Eigen::VectorXd> > Au_;

   // the list of the spin lambda's
   std::list<Eigen::VectorXd> spin_;
   // the list of the isospin lambda's
   std::list<Eigen::VectorXd> isospin_;
   


  // generation 
  //random numbers...
  //std::random_device rd_;
  std::mt19937 mt_;//(rd_());
  // a vector of random numbers, one for each pair
  std::vector<
  std::vector<std::uniform_real_distribution<double> > > dist_;
  // this is for u
  std::uniform_real_distribution<double> udist_;
  // this is for spin
  std::uniform_real_distribution<double> spinDist_;
  // this is for isospin
  std::uniform_real_distribution<double> isospinDist_;


   // matrix of the scalar products between the Gaussian
   // if you want, the metric... g_ij = <A_i|A_j>
   Eigen::MatrixXd gij_;

   // this is the matrix of the casimir on the Gaussian basis
   // casimir = <A_i| Casimir |A_j>
   Eigen::MatrixXd casimir_;

   // only spacial casimir
   Eigen::MatrixXd casimirSpace_;
   // spin 
   Eigen::MatrixXd casimirSpin_;
   // isospin
   Eigen::MatrixXd casimirIso_;
   // spin and sospin
   Eigen::MatrixXd casimirSpinIso_;

   // Hamiltonian in the Gaussian basis
   Eigen::MatrixXd haij_;
   Eigen::MatrixXd vaij_;
   Eigen::MatrixXd qaij_;
   Eigen::MatrixXd taij_; // this is only for the three-body
   Eigen::MatrixXd ta4ij_; // this is only for the three-body
   Eigen::MatrixXd tapij_; // spin-dependent part
   Eigen::MatrixXd trapij_; // external harmonic trap
   // matrix of the 
   // 1/#pairs \sum_{i<j} r_{ij}^2
   // the quadratic radius in the gaussian basis
   Eigen::MatrixXd raij_; 

   // Components of the basis - eigenvectors 
   // on the Gaussian basis
   // |phi_i> = c_ji |A_j>
   // the normalization is such that  c'*g*c = 1
   Eigen::MatrixXd cij_;


   // scalar products between eigenvectors and basis
   // phiij_ = <phi_i|A_j>
   Eigen::MatrixXd phiij_;

   // the eigenvalues
   Eigen::MatrixXd energy_;

   // file of the random coefficients 
   std::fstream fileOutput_;

   // private function to set up the 
   // hamiltonian for the k-th elements
   Eigen::MatrixXd  calculateHamiltonian(Eigen::MatrixXd &a, Eigen::VectorXd &u,
       Eigen::VectorXd &spin,Eigen::VectorXd &isospin);
  
   // function which integrates the potential (A.129) p.282 SV
   double J_(int n, double c, int pair, int kind);
   // evaluate directly the potential
   double evaluatePotential_(int pair, double norma, double c, double rho,
       double gamma, double gammap,
     Eigen::VectorXd lpS, Eigen::VectorXd lS, 
     Eigen::VectorXd lpI, Eigen::VectorXd lI, 
     int kind);

   // fuction for the spin exchange term
   double spinExange(Eigen::VectorXd lp, Eigen::VectorXd l, int pair=0);
   // function for the isospin exchange term
   double isospinExange(Eigen::VectorXd lp, Eigen::VectorXd l, int pair=0);

   // function to project to s=0,1
   std::pair<double,double>
   spinProjector(Eigen::VectorXd lp, Eigen::VectorXd l, int pair);
   // function to project to t=0,1
   std::pair<double,double>
   isospinProjector(Eigen::VectorXd lp, Eigen::VectorXd l, int pair);

   // just the gaussian integral
   double
   gaussian_(int n,double c, double v0, double r0);

   // try to import the Simplex 
    std::pair<
      std::vector<double> , // the x which minimize
      std::tuple<Eigen::VectorXd, // eigenvalues
                 std::pair<Eigen::MatrixXd,Eigen::VectorXd>, // <a,u>
                 Eigen::VectorXd, // alphas
                 Eigen::VectorXd,//spin
                 Eigen::VectorXd> >// isospin
   simplex_(
			 //std::vector<double> init,    //initial guess of the parameters
       allVariables input,
			 std::vector<std::vector<double> > x =  std::vector<std::vector<double> >(),
			 //x: The Simplex
			 //double tol=1E10*std::numeric_limits<double>::epsilon(), //termination criteria
			 double tol=1E-4, //termination criteria
			 int iterations=100000); //iteration step number

    // function to calculate the overlaps gij with the matrix a ; it 
    // add the new value to the old one... for symmetrization 
    void add_hamiltonian_(Eigen::MatrixXd &a, Eigen::VectorXd &u,
    Eigen::VectorXd &lS, Eigen::VectorXd &lI,int sign=1);

    // the same fo the casimir
    void add_casimir_(Eigen::MatrixXd &a, Eigen::VectorXd &u,
    Eigen::VectorXd &lS, Eigen::VectorXd &lI, int sign=1);


  public:
    
    // standard contructor
    // for the moment can just create 1 state
    Basis(InputData & indata, Coordinates & coo, size_t mxDim=3100) ;


    // Add a new basis element
    int newElement(std::pair<Eigen::MatrixXd,Eigen::VectorXd>,// <a,u>
               Eigen::VectorXd, // alphas
               Eigen::VectorXd,//spin lambdas
               Eigen::VectorXd ); // isospin lambdas
    // Test a new element
    std::tuple<Eigen::VectorXd, // eigenvalues
               std::pair<Eigen::MatrixXd,Eigen::VectorXd>, // <a,u>
               Eigen::VectorXd, // alphas
               Eigen::VectorXd,//spin
               Eigen::VectorXd> // isospin
    testElement();
    // Test a new element
    std::tuple<Eigen::VectorXd, // eigenvalues
               std::pair<Eigen::MatrixXd,Eigen::VectorXd>, // <a,u>
               Eigen::VectorXd, // alphas
               Eigen::VectorXd,//spin
               Eigen::VectorXd> // isospin
    testElementAlpha(Eigen::VectorXd alpha);
    // The same but generated by the Simplex
    std::pair<
      double,  // this is the value of the eigenvalue 
      std::tuple<Eigen::VectorXd, // eigenvalues
                 std::pair<Eigen::MatrixXd,Eigen::VectorXd>, // <a,u>
                 Eigen::VectorXd, // alphas
                 Eigen::VectorXd,//spin
                 Eigen::VectorXd> >// isospin
    testSimplex(std::vector<double> x, allVariables input);
    // to set private variable
    void setState(int state) {state_=state;};

    /* 
     * Refinement the basis
     */
    // it takes how many new trials for each element
    // and it gives back the number of basis elements
    // that has been changed
    int refineBasis(int nOftrials);

    // in this case we use the simplex algorithm but starting from
    // one point of the basis
    int refineSimplex(int startingPoint = 0);

    // just access the private variables
    Eigen::MatrixXd * getGIJ(void)         { return & gij_;};
    Eigen::MatrixXd * getCasimir(void)     { return & casimir_;};
    Eigen::MatrixXd * getCasimirSpace(void)     { return & casimirSpace_;};
    Eigen::MatrixXd * getCasimirSpin(void)      { return & casimirSpin_;};
    Eigen::MatrixXd * getCasimirIso(void)       { return & casimirIso_;};
    Eigen::MatrixXd * getCasimirSpinIso(void)   { return & casimirSpinIso_;};
    Eigen::MatrixXd * getCIJ(void)         { return & cij_;};
    Eigen::MatrixXd * getHAIJ(void)        { return & haij_;};
    Eigen::MatrixXd * getVAIJ(void)        { return & vaij_;};
    Eigen::MatrixXd * getQAIJ(void)        { return & qaij_;};
    Eigen::MatrixXd * getTAIJ(void)        { return & taij_;};
    Eigen::MatrixXd * getTA4IJ(void)       { return & ta4ij_;};
    Eigen::MatrixXd * getTRAPIJ(void)      { return & trapij_;};
    Eigen::MatrixXd * getRAIJ(void)        { return & raij_;};
    Eigen::MatrixXd * getPHIIJ(void)       { return & phiij_;};
    Eigen::MatrixXd * getEnergy(void)      { return & energy_;};
    int               getDimension(void)   { return k_;};


    /*
     * Save all of the data in a HDF5 file - very simple
     */
    int saveAll(std::string fileName);

};

#endif // BASIS_H 
