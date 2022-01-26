/**
 * @file coordinates.h
 */
#ifndef COORDINATES_H
#define COORDINATES_H

#include<vector>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Core>
class
Coordinates
{
  private:
    static bool test_(std::vector<int>,std::vector<int>);
    // cfr notes of 08/12/2017
    double C_(int ir, int pr, double s, double m);

  public:
    // Vector of pairs
    std::vector<std::pair<int,int> > pairs;
    // I need a  vector of triplets and I want to keep trace
    // of the pairs in each triplet - there are three
    // of them. I store their position in pairs
    std::vector<std::tuple<int,int,int> > triplets;
    // I need a  vector of quadruplets and I want to keep trace
    // of the pairs in each quadruplets - there are six
    // of them. I store their position in pairs
    std::vector<std::tuple<int,int,int,int,int,int> > quadruplets;



    // vector of permutations of the particle numbers
    // 0 1 2 ...  1 2 3...   etc...
    // The permutations should be understood as matrices, 
    // but only their action
    std::vector< Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> > permutations;
    std::vector<int> signPermutations;
    
    // The vetor of the transpositions in r
    std::vector< Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> > transpositions;
     // Vector of tranposition in x
    std::vector<Eigen::MatrixXd> transpositionJacobi;
     // Vector of permutations in x
    std::vector<Eigen::MatrixXd> permutationsJacobi;

    // U matrix  eq. (2.5) Suzuki&Varga 

    /* 
     * They allow to pass from particle coordinates 
     * to Jacobi coordinates (K-system)
     * There is also the CM coordinate
     *
     * x = U*r
     *
     * r_1 ... r_np  the coordinates of the particles
     * x_1 = r_1-r_2 etc... up to x_{np-1}
     * x_np is the center of mass  
     */

    Eigen::MatrixXd U;
    Eigen::MatrixXd UINV;

    // from UINV and masses we contruct the matrix Omega we use
    // in the case of harmonic trap 
    // It is the inverse of Lambda
    Eigen::MatrixXd Omega;


    // This is to calculate the r^2; it the same as Omega without
    // the masses...
    // <r^2> = \sum_i r_i^2/Nparticles
    Eigen::MatrixXd Radius2;


    // Lambda matrix for the kinetic energy Eq.(2.11) Suzuki&Varga
    // it is diagonal and has dimension np-1,np-1 because we 
    // substract the cm motion
    Eigen::MatrixXd Lambda;

    // Matrix for the transformation between the relative distances
    // and the Jacobi coordinates
    // r_i-r_j  = W * x
    // W is a np*(np-1)/2 * (np-1) matrix
    Eigen::MatrixXd W;
    // same as above for momenta
    // p_i-p_j = Z * px
    Eigen::MatrixXd Z;



    // Our notation for Jacobi coordinates
    /* 
     * This is the usual way we use, see our articles, 
     * where the last Jacobi is the relative one
     * and with the kinetic energy with the same weight 
     * for all the Jacobi. We calculate it just for check
     * but it must be KIN = 2*I
     *
     * x = V*r
     */
    Eigen::MatrixXd V;
    Eigen::MatrixXd VINV;
    Eigen::MatrixXd KIN;
    // our W matrix....
    Eigen::MatrixXd WV;

    //Definitions of zita vector for non-local potential  from (C.27) and (C.25)
    //of Few Body System (2008) 42: 33-72 Y. Suzuki, W. Horiuchi
    std::vector<Eigen::VectorXd> zita;
    //Definitions of transformation matrices for different Jacobi coordinates 
    //T of (2.31) Suzuki-Varga pages 17
    //or (C.4) of Few Body System (2008) 42: 33-72 Y. Suzuki, W. Horiuchi
    std::vector<Eigen::MatrixXd> Tk;
    
    // we want to construct spin/isospin state
    // here we put the primitive spin/isospin 
    std::vector<std::vector<int> > primitiveS;
    std::vector<std::vector<int> > primitiveI;

    /*
     * These are all defined on the primitive basis
     */
    // and we want the representation of the transpositions
    std::vector<Eigen::MatrixXd> traspositionSpin;
    std::vector<Eigen::MatrixXd> traspositionIsospin;
    // and we want the representation of the permutations
    std::vector<Eigen::MatrixXd> permutationsSpin;
    std::vector<Eigen::MatrixXd> permutationsIsospin;
    // and we want the representation of the projectors
    std::vector<std::pair<Eigen::MatrixXd,Eigen::MatrixXd> > projectorSpin;
    std::vector<std::pair<Eigen::MatrixXd,Eigen::MatrixXd> > projectorIsospin;
    // pair-charge operator on isospin space
    std::vector<Eigen::MatrixXd> chargeOperator; 

    // irreducible representation
    std::vector<std::vector<int> > irrepsS;
    // for each T compatible with Tz we have all the irreps
    // and for each irreps we have its representation with
    // up(1) and down(2)
    std::vector<std::vector<std::vector<int> > > irrepsI;
    
    // the coefficients of the irreps on the primitive basis
    // lS(i,j) - jth component of the ith irreps basis element
    Eigen::MatrixXd lS;
    // we can have more irreps - so the vector
    std::vector<Eigen::MatrixXd> lI;
    
    // Constructors
    // with data 
    Coordinates (int numberOfParticles, std::vector<double> masses, double spin=-1, double isospin=-1,bool ifCoulomb=false) ;

};

#endif // COORDINATES_H
