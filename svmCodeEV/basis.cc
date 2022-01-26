/**
 * @file basis.cc
 */
#include"basis.h"
#include<functional>
// to use the hdf5 library
#include "hdf5.h"

Basis::Basis(InputData & indata, Coordinates & coo, size_t maxDimension) : indata_(indata),coo_(coo),
  maxDimension_(maxDimension)
{
  L_ = indata_.angularMomentum;
  S_ = indata_.totalSpin;
  T_ = indata_.totalIsospin;
  np_ = indata_.npart;
  Factorial fact;
  factorialNp_ = fact(np_);
  // prepare the metric matrix
  gij_ = Eigen::MatrixXd::Zero(maxDimension,maxDimension);
  // prepare the casimir matrix
  casimir_      = Eigen::MatrixXd::Zero(maxDimension,maxDimension);
  casimirSpace_ = Eigen::MatrixXd::Zero(maxDimension,maxDimension);
  casimirSpin_  = Eigen::MatrixXd::Zero(maxDimension,maxDimension);
  casimirIso_   = Eigen::MatrixXd::Zero(maxDimension,maxDimension);
  casimirSpinIso_   = Eigen::MatrixXd::Zero(maxDimension,maxDimension);
  // prepare the hamiltonian in the gaussian basis
  haij_ = Eigen::MatrixXd::Zero(maxDimension,maxDimension);
  vaij_ = Eigen::MatrixXd::Zero(maxDimension,maxDimension);
  qaij_ = Eigen::MatrixXd::Zero(maxDimension,maxDimension);
  taij_ = Eigen::MatrixXd::Zero(maxDimension,maxDimension);
  ta4ij_ = Eigen::MatrixXd::Zero(maxDimension,maxDimension);
  trapij_ = Eigen::MatrixXd::Zero(maxDimension,maxDimension);
  tapij_ = Eigen::MatrixXd::Zero(maxDimension,maxDimension);
  // the radius square
  raij_ = Eigen::MatrixXd::Zero(maxDimension,maxDimension);
  // prepare the eigenfunction vector
  cij_ = Eigen::MatrixXd::Zero(maxDimension,maxDimension);
  // prepare the eigenvalues
  energy_ = Eigen::MatrixXd::Zero(maxDimension,1);
  // prepare the scalar products
  phiij_ = Eigen::MatrixXd::Zero(maxDimension,maxDimension);
  // set the initial dimension of the basis
  k_=0;

  
  // generation 
  //random numbers...
  int nofpairs = np_*(np_-1)/2;
  int nofeigs = indata.alphaRanges[0].size()/2;
  dist_.resize(nofeigs);
  std::random_device rd_;
  mt_ = std::mt19937(rd_());
  for(int i=0; i<nofpairs; i++)
  {
    for(int j=0; j<(int)indata.alphaRanges[0].size()/2; j++)
    {
      double a = indata.alphaRanges[i][0+2*j];
      double b = indata.alphaRanges[i][1+2*j];
      dist_[j].push_back(std::uniform_real_distribution<double> (a,b));
    }
  }

  // for the u vector
  udist_ = std::uniform_real_distribution<double> (-1,1);

  // for spin/isospin
  if(S_>=0) // spin is defined
    spinDist_  = std::uniform_real_distribution<double>(-M_PI/2.,M_PI/2.);
  if(T_>=0) // isospin is defined
    isospinDist_  = std::uniform_real_distribution<double>(-M_PI/2.,M_PI/2.);


  
}
std::tuple<Eigen::VectorXd, // eigenvalues
           std::pair<Eigen::MatrixXd,Eigen::VectorXd>, // <a,u>
           Eigen::VectorXd,  // alphas
           Eigen::VectorXd, // spin
           Eigen::VectorXd> // isospin
Basis::testElementAlpha(Eigen::VectorXd alpha_input)
{
  //int R = 20; // assume R is known
  /* Generation of the new matrix
   * ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   * the matrix can be generated in rij space
   * and then 'rotated' to the Jacobi system
   * Other possibilities are to be explored,
   * as to rotate to different Jacobi systems
   */
  int nofpairs = coo_.pairs.size();
  // alpha is the diagonal matrix in rij space
  Eigen::VectorXd alpha(nofpairs) ;
  for(auto i=0; i< nofpairs; i++)
    //alpha(i) = 1.0/pow(alpha_input(i) * R,2);
    alpha(i) = 1.0/pow(alpha_input(i),2);

  // rotate to Jacobi x - see W matrix in coordinates 
  // rij = W * x 
  Eigen::MatrixXd a = coo_.W.transpose()*alpha.asDiagonal()*coo_.W;
  

  // generation of the u-vector for non-zero angular momenta
  int uDim = 1;
  if(indata_.angularMomentum > 0)
    uDim = np_-1;
  Eigen::VectorXd u(uDim);
  for(auto i=0; i<uDim; i++)
    u(i) = udist_(mt_);


  // generation of spin variables
  int irSize =coo_.irrepsS.size();
  Eigen::VectorXd lS(coo_.primitiveS.size());
  Eigen::VectorXd lambdaS(irSize);
  double sinus=1;
  for(auto i=0; i<irSize-1; i++)
  { 
    double angle = spinDist_(mt_);
    lambdaS(i) = cos(angle)*sinus;
    sinus *= sin(angle);
  }
  if(S_>=0)
    lambdaS(irSize-1) = sinus;


  lS = lambdaS.transpose()*coo_.lS;

  // generation of isospin variables

  // these are the components on the primitive basis
  Eigen::VectorXd lI = Eigen::VectorXd::Zero(coo_.primitiveI.size());

  // First I make linear combination of the irreps
  // with different isospin
  int numberOfIrreps = coo_.irrepsI.size();
  Eigen::VectorXd l1(numberOfIrreps);
  //sinus=1;
  double cosinus=1;
  for(auto i=0; i<numberOfIrreps-1; i++)
  { 
    double angle = isospinDist_(mt_);
    l1(i) = sin(angle)*cosinus;
    //l1(i) = cos(angle)*sinus;
    //sinus *= sin(angle);
    cosinus *= cos(angle);
  }
  if(T_>=0)
    l1(numberOfIrreps-1) = cosinus;
    //l1(numberOfIrreps-1) = sinus;
  

  // DEBUG
  /*
  l1(0)=0.0;
  l1(1)=0.0;
  l1(2)=0.0;
  l1(3)=1.0;
  */
  /*
  std::cout << "Coefficienti sui diversi T" << std::endl;
  std::cout << l1 << std::endl;
  std::cout << "*********" << std::endl;
  */


  // then we make linear combination
  // of the irreps with same isospin
  for(auto i=0; i<numberOfIrreps; i++)
  {
    auto ir = coo_.irrepsI[i];
    irSize =ir.size();
    Eigen::VectorXd l2(irSize);

    sinus=1;
    for(auto i=0; i<irSize-1; i++)
    { 
      double angle = isospinDist_(mt_);
      l2(i) = cos(angle)*sinus;
      sinus *= sin(angle);
    }
    if(T_>=0)
      l2(irSize-1) = sinus;
    l2 *= l1(i);

  // DEBUG
  /*
  std::cout << "Coefficienti sui diversi T" << std::endl;
  std::cout << l2 << std::endl;
  std::cout << "*********" << std::endl;
  */

    lI += l2.transpose()*coo_.lI[i];
  } // loop over the irreps same T

  /*
  std::cout << "Vettore finale sulla base primitiva" << std::endl;
  std::cout << lI << std::endl;
  */


  /*
   * DEBUG - USO LA BASE PRIMITIVA 
   */
/*
   if(indata_.ifCoulomb)
   {
    int primSize = coo_.primitiveI.size();
    sinus=1;
    for(auto i=0; i<primSize-1; i++)
    {
      double angle = isospinDist_(mt_);
      lI(i) = cos(angle)*sinus;
      sinus *= sin(angle);
    }
    if(T_>=0)
      lI(primSize-1) = sinus;
  }
*/


  // I pass the components of spin/isospin on the primitive basis
  Eigen::MatrixXd h = calculateHamiltonian(a,u,lS,lI);
  
  // now the diagonalization of the hamiltonian
  //std::cout << " h = " << std::endl << h << std::endl;


  Eigen::SelfAdjointEigenSolver <Eigen::MatrixXd> es(h,Eigen::EigenvaluesOnly);
  //std::cout << "The eigenvalues of h are:" << std::endl << es.eigenvalues() << std::endl;
  // std::cout << "The eigenvectors of h are:" << std::endl << es.eigenvectors() << std::endl;

  // the new c_ij are
  //cij_.block(0,0,k_+1,k_+1) = cij_.block(0,0,k_+1,k_+1) * es.eigenvectors();
  
  // the energy
  //energy_.block(0,0,k_+1,1) = es.eigenvalues();

  /*
  std::cout << "Test->  " << std::endl << cij_.transpose().block(0,0,k_+1,k_+1) * 
                              gij_.block(0,0,k_+1,k_+1) *
                              cij_.block(0,0,k_+1,k_+1) << std::endl;
  */

  // finished with the new element
  //A.push_back(a);
  //k_++;

  auto out =
    std::make_tuple(es.eigenvalues(),std::make_pair(a,u),alpha,lS,lI);
  return out;
}

std::tuple<Eigen::VectorXd, // eigenvalues
           std::pair<Eigen::MatrixXd,Eigen::VectorXd>, // <a,u>
           Eigen::VectorXd,  // alphas
           Eigen::VectorXd, // spin
           Eigen::VectorXd> // isospin
Basis::testElement()
{

  /* Generation of the new matrix
   * ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   * the matrix can be generated in rij space
   * and then 'rotated' to the Jacobi system
   * Other possibilities are to be explored,
   * as to rotate to different Jacobi systems
   */
  int nofpairs = coo_.pairs.size();
  // alpha is the diagonal matrix in rij space
  Eigen::VectorXd alpha(nofpairs) ;
  for(auto i=0; i< nofpairs; i++)
    alpha(i) = 1.0/pow(dist_[state_][i](mt_),2);

  // rotate to Jacobi x - see W matrix in coordinates 
  // rij = W * x 
  Eigen::MatrixXd a = coo_.W.transpose()*alpha.asDiagonal()*coo_.W;
  

  // generation of the u-vector for non-zero angular momenta
  int uDim = 1;
  if(indata_.angularMomentum > 0)
    uDim = np_-1;
  Eigen::VectorXd u(uDim);
  for(auto i=0; i<uDim; i++)
    u(i) = udist_(mt_);


  // generation of spin variables
  int irSize =coo_.irrepsS.size();
  Eigen::VectorXd lS(coo_.primitiveS.size());
  Eigen::VectorXd lambdaS(irSize);
  double sinus=1;
  for(auto i=0; i<irSize-1; i++)
  { 
    double angle = spinDist_(mt_);
    lambdaS(i) = cos(angle)*sinus;
    sinus *= sin(angle);
  }
  if(S_>=0)
    lambdaS(irSize-1) = sinus;


  lS = lambdaS.transpose()*coo_.lS;

  // generation of isospin variables

  // these are the components on the primitive basis
  Eigen::VectorXd lI = Eigen::VectorXd::Zero(coo_.primitiveI.size());

  // First I make linear combination of the irreps
  // with different isospin
  int numberOfIrreps = coo_.irrepsI.size();
  Eigen::VectorXd l1(numberOfIrreps);
  //sinus=1;
  double cosinus=1;
  for(auto i=0; i<numberOfIrreps-1; i++)
  { 
    double angle = isospinDist_(mt_);
    l1(i) = sin(angle)*cosinus;
    //l1(i) = cos(angle)*sinus;
    //sinus *= sin(angle);
    cosinus *= cos(angle);
  }
  if(T_>=0)
    l1(numberOfIrreps-1) = cosinus;
    //l1(numberOfIrreps-1) = sinus;
  

  // DEBUG
  /*
  l1(0)=0.0;
  l1(1)=0.0;
  l1(2)=0.0;
  l1(3)=1.0;
  */
  /*
  std::cout << "Coefficienti sui diversi T" << std::endl;
  std::cout << l1 << std::endl;
  std::cout << "*********" << std::endl;
  */


  // then we make linear combination
  // of the irreps with same isospin
  for(auto i=0; i<numberOfIrreps; i++)
  {
    auto ir = coo_.irrepsI[i];
    irSize =ir.size();
    Eigen::VectorXd l2(irSize);

    sinus=1;
    for(auto i=0; i<irSize-1; i++)
    { 
      double angle = isospinDist_(mt_);
      l2(i) = cos(angle)*sinus;
      sinus *= sin(angle);
    }
    if(T_>=0)
      l2(irSize-1) = sinus;
    l2 *= l1(i);

  // DEBUG
  /*
  std::cout << "Coefficienti sui diversi T" << std::endl;
  std::cout << l2 << std::endl;
  std::cout << "*********" << std::endl;
  */

    lI += l2.transpose()*coo_.lI[i];
  } // loop over the irreps same T

  /*
  std::cout << "Vettore finale sulla base primitiva" << std::endl;
  std::cout << lI << std::endl;
  */


  /*
   * DEBUG - USO LA BASE PRIMITIVA 
   */
/*
   if(indata_.ifCoulomb)
   {
    int primSize = coo_.primitiveI.size();
    sinus=1;
    for(auto i=0; i<primSize-1; i++)
    {
      double angle = isospinDist_(mt_);
      lI(i) = cos(angle)*sinus;
      sinus *= sin(angle);
    }
    if(T_>=0)
      lI(primSize-1) = sinus;
  }
*/


  // I pass the components of spin/isospin on the primitive basis
  Eigen::MatrixXd h = calculateHamiltonian(a,u,lS,lI);
  
  // now the diagonalization of the hamiltonian
  //std::cout << " h = " << std::endl << h << std::endl;


  Eigen::SelfAdjointEigenSolver <Eigen::MatrixXd> es(h,Eigen::EigenvaluesOnly);
  //std::cout << "The eigenvalues of h are:" << std::endl << es.eigenvalues() << std::endl;
  // std::cout << "The eigenvectors of h are:" << std::endl << es.eigenvectors() << std::endl;

  // the new c_ij are
  //cij_.block(0,0,k_+1,k_+1) = cij_.block(0,0,k_+1,k_+1) * es.eigenvectors();
  
  // the energy
  //energy_.block(0,0,k_+1,1) = es.eigenvalues();

  /*
  std::cout << "Test->  " << std::endl << cij_.transpose().block(0,0,k_+1,k_+1) * 
                              gij_.block(0,0,k_+1,k_+1) *
                              cij_.block(0,0,k_+1,k_+1) << std::endl;
  */

  // finished with the new element
  //A.push_back(a);
  //k_++;

  auto out =
    std::make_tuple(es.eigenvalues(),std::make_pair(a,u),alpha,lS,lI);
  return out;
}


std::pair<
      double,  // this is the value of the eigenvalue 
      Basis::allVariables>
Basis::testSimplex(std::vector<double> x,Basis::allVariables input)
{

  /* Generation of the new matrix
   * ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   * the matrix can be generated in rij space
   * and then 'rotated' to the Jacobi system
   * Other possibilities are to be explored,
   * as to rotate to different Jacobi systems
   */
  int nofpairs = coo_.pairs.size();
  // alpha is the diagonal matrix in rij space
  int alphaStart = 0;
  int alphaDim = nofpairs;
  Eigen::VectorXd alpha(nofpairs) ;
  for(auto i=0; i<alphaDim; i++)
    alpha(i) = x[alphaStart+i];

  // rotate to Jacobi x - see W matrix in coordinates 
  // rij = W * x 
  Eigen::MatrixXd a = coo_.W.transpose()*alpha.asDiagonal()*coo_.W;
  // symmetrization
  /*
  for(auto p : coo_.permutationsJacobi)
    a += p.transpose()*b*p;
  */

  /*
  // generation of the u-vector for non-zero angular momenta
  Eigen::VectorXd u(np_-1);
  int uStart=alphaDim;
  int uDim = np_-1; 
  for(auto i=0; i<uDim; i++)
    u(i) = x[uStart+i];

  // generation of spin variables
  Eigen::VectorXd lS(coo_.primitiveS.size());

  int spinDim = coo_.primitiveS.size();
  int spinStart = alphaDim+uDim;
  for(auto i=0; i<spinDim; i++)
    lS(i) = x[i+spinStart];


  // generation of isospin variables
  Eigen::VectorXd lI(coo_.primitiveI.size());

  int isoDim = coo_.primitiveI.size();
  int isoStart = alphaDim+uDim+spinDim;
  for(auto i=0; i<isoDim; i++)
    lI(i) = x[i+isoStart];
   */


  Eigen::MatrixXd h = calculateHamiltonian(a,std::get<1>(input).second,
      std::get<3>(input),std::get<4>(input));
  
  // now the diagonalization of the hamiltonian
  // std::cout << " h = " << std::endl << h << std::endl;
  Eigen::SelfAdjointEigenSolver <Eigen::MatrixXd> es(h,Eigen::EigenvaluesOnly);
  //std::cout << "The eigenvalues of h are:" << std::endl << es.eigenvalues() << std::endl;
  // std::cout << "The eigenvectors of h are:" << std::endl << es.eigenvectors() << std::endl;

  // the new c_ij are
  //cij_.block(0,0,k_+1,k_+1) = cij_.block(0,0,k_+1,k_+1) * es.eigenvectors();
  
  // the energy
  //energy_.block(0,0,k_+1,1) = es.eigenvalues();

  /*
  std::cout << "Test->  " << std::endl << cij_.transpose().block(0,0,k_+1,k_+1) * 
                              gij_.block(0,0,k_+1,k_+1) *
                              cij_.block(0,0,k_+1,k_+1) << std::endl;
  */

  // finished with the new element
  //A.push_back(a);
  //k_++;

  auto out =
    std::make_tuple(es.eigenvalues(),std::make_pair(a,std::get<1>(input).second),
          std::get<2>(input),std::get<3>(input),std::get<4>(input));
  return std::make_pair(es.eigenvalues()[state_],out);
}
/*
 *------------------------------------------------------------------------------------------------
 */

int
Basis::newElement(std::pair<Eigen::MatrixXd,Eigen::VectorXd> au,Eigen::VectorXd alpha,
       Eigen::VectorXd spin, Eigen::VectorXd isospin)
{
  
  // The hamiltonian in the new basis
  Eigen::MatrixXd h = calculateHamiltonian(au.first,au.second,spin,isospin);

  // DEBUG
  /*
  int sizeNew = k_+1;
  Eigen::MatrixXd h3 = cij_.transpose().block(0,0,sizeNew,sizeNew) 
                    * (haij_+vaij_+taij_+qaij_).block(0,0,sizeNew,sizeNew)
                    * cij_.block(0,0,sizeNew,sizeNew);
  Eigen::SelfAdjointEigenSolver <Eigen::MatrixXd> e3(h3,Eigen::EigenvaluesOnly);
  std::cout << "Coulomb on without coulomb --- " << indata_.energyUnits*e3.eigenvalues()[0]  << "  "
    << indata_.energyUnits*e3.eigenvalues()[1]<< "  " << indata_.energyUnits*e3.eigenvalues()[2]  << std::endl;
  //std::cout << "Three-body on two body --- " << indata_.energyUnits*e3.eigenvalues()[0] << std::endl;
  */

  // now the diagonalization of the hamiltonian
  /*
  std::cout << "*************************************" << std::endl;
  std::cout << " h = " << std::endl << h << std::endl;
  std::cout << "*************************************" << std::endl;
  */
  Eigen::SelfAdjointEigenSolver <Eigen::MatrixXd> es(h);
  //std::cout << "The eigenvalues of h are:" << std::endl << es.eigenvalues() << std::endl;
  // std::cout << "The eigenvectors of h are:" << std::endl << es.eigenvectors() << std::endl;

  // the new c_ij are
  cij_.block(0,0,k_+1,k_+1) = cij_.block(0,0,k_+1,k_+1) * es.eigenvectors();
  
  // the energy
  energy_.block(0,0,k_+1,1) = es.eigenvalues();

  // finished with the new element
  Au_.push_back(au);
  alpha_.push_back(alpha);
  spin_.push_back(spin);
  isospin_.push_back(isospin);
  k_++;

  return 0;
}
/*
 *--------------------------------------------------------------------------------------
 */
int
Basis::refineSimplex(int startingPoint)
{

  int nOfChanges=0;

  int indx=startingPoint; // index of the element going to the last position
  Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(k_);
  // loop over all the elements but the last one
  for(auto aold = std::next(Au_.begin(),startingPoint); 
           aold != std::prev(Au_.end(),1); aold++)
  {
    perm.setIdentity(); // we have always to do this otherwise the 
                        // the transposition acts on traspositions...
    // change the indx with the last
    perm.applyTranspositionOnTheRight(indx,k_-1);
    //std::cout << perm.toDenseMatrix() << std::endl;;
    gij_.block(0,0,k_,k_) = perm *gij_.block(0,0,k_,k_)*perm;
    haij_.block(0,0,k_,k_) = perm *haij_.block(0,0,k_,k_)*perm;
    vaij_.block(0,0,k_,k_) = perm *vaij_.block(0,0,k_,k_)*perm;
    qaij_.block(0,0,k_,k_) = perm *qaij_.block(0,0,k_,k_)*perm;
    taij_.block(0,0,k_,k_) = perm *taij_.block(0,0,k_,k_)*perm;
    ta4ij_.block(0,0,k_,k_) = perm *ta4ij_.block(0,0,k_,k_)*perm;
    trapij_.block(0,0,k_,k_) = perm *trapij_.block(0,0,k_,k_)*perm;
    tapij_.block(0,0,k_,k_) = perm *tapij_.block(0,0,k_,k_)*perm;
    raij_.block(0,0,k_,k_) = perm *raij_.block(0,0,k_,k_)*perm;
    casimir_.block(0,0,k_,k_)      = perm *casimir_.block(0,0,k_,k_)*perm;
    casimirSpace_.block(0,0,k_,k_) = perm *casimirSpace_.block(0,0,k_,k_)*perm;
    casimirSpin_.block(0,0,k_,k_)  = perm *casimirSpin_.block(0,0,k_,k_)*perm;
    casimirIso_.block(0,0,k_,k_)   = perm *casimirIso_.block(0,0,k_,k_)*perm;
    casimirSpinIso_.block(0,0,k_,k_)   = perm *casimirSpinIso_.block(0,0,k_,k_)*perm;

    //swap the present element with the last one 
    //before *aold is the matrix in indx potion
    //after  *aold is what was the k_th matrix now in indx
    // now the last element (was indx'th) is A_.back()
    std::iter_swap(aold,std::prev(Au_.end(),1));
    std::iter_swap(std::next(alpha_.begin(),indx),std::prev(alpha_.end(),1));

    std::iter_swap(std::next(spin_.begin(),indx), std::prev(spin_.end(),1));
    std::iter_swap(std::next(isospin_.begin(),indx), std::prev(isospin_.end(),1));

    // up to now the dimension of the basis is k_
    // we take back the last basis element saving the old
    // values

    std::tuple<
      Eigen::VectorXd,std::pair<Eigen::MatrixXd,Eigen::VectorXd>,Eigen::VectorXd,
      Eigen::VectorXd, Eigen::VectorXd>
    oldOut = std::make_tuple(energy_.block(0,0,k_,1),Au_.back(),alpha_.back(),
        spin_.back(),isospin_.back());
    Au_.pop_back();
    alpha_.pop_back();
    spin_.pop_back();
    isospin_.pop_back();
    k_ = k_-1;

    // and set the new cij_ of the reduced basis
    Eigen::GeneralizedSelfAdjointEigenSolver <Eigen::MatrixXd> 
      es((haij_+vaij_).block(0,0,k_,k_),gij_.block(0,0,k_,k_));
    cij_.block(0,0,k_,k_) =  es.eigenvectors();
    // I've to put at zero the k_+1 elements...
    for(int i=0;i<k_+1;i++)
    {
      cij_(i,k_) = 0.0;
      cij_(k_,i) = 0.0;
    }
    
    //now we can test a new last element
    //
    int l=state_;
    // DEBUG
    /*
    std::cerr << k_+1 << "  ";
    for(auto i=0; i<get<2>(oldOut).size(); i++)
      std::cerr << get<2>(oldOut)[i] << " ";
    */
    auto result = simplex_(oldOut);
    /*
    for(auto i=0; i<get<2>(result.second).size(); i++)
      std::cerr << get<2>(result.second)[i] << " ";
    std::cerr << std::endl;
    */
    std::cout << " Raffinato "  << std::get<0>(result.second)[l] *  indata_.energyUnits<< std::endl;

    if(std::get<0>(result.second)(l) < std::get<0>(oldOut)[l])
    {
      oldOut = result.second;
     } else {
      std::cout << "Skip" << std:: endl;
    }

/*
    int l=state_;
    for(auto j=0; j< nOfProve; j++)
    {
      auto tmpOut = testElement();
      if(std::get<0>(tmpOut)(l) != std::get<0>(tmpOut)(l))
        continue;
//      std::cout << std::get<0>(tmpOut)(l) << " "<<  std::get<0>(oldOut)(l) << std::endl;
      if(std::get<0>(tmpOut)(l) < std::get<0>(oldOut)(l))
        oldOut = tmpOut;
    }
    //if(std::get<0>(oldOut)(l) != std::get<0>(oldOut)(l))
    //  continue;
*/
    newElement(std::get<1>(oldOut),std::get<2>(oldOut),std::get<3>(oldOut),
        std::get<4>(oldOut));  


    indx++; // next one

  }


  return nOfChanges;
}

/*
 *--------------------------------------------------------------------------------------
 */
int
Basis::refineBasis(int nOfProve)
{

  int nOfChanges=0;

  int indx=0; // index of the element going to the last position
  Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(k_);
  // loop over all the elements but the last one
  for(auto aold = Au_.begin(); aold != std::prev(Au_.end(),1); aold++)
  {
    perm.setIdentity(); // we have always to do this otherwise the 
                        // the transposition acts on traspositions...
    // change the indx with the last
    perm.applyTranspositionOnTheRight(indx,k_-1);
    //std::cout << perm.toDenseMatrix() << std::endl;;
    gij_.block(0,0,k_,k_) = perm *gij_.block(0,0,k_,k_)*perm;
    haij_.block(0,0,k_,k_) = perm *haij_.block(0,0,k_,k_)*perm;
    vaij_.block(0,0,k_,k_) = perm *vaij_.block(0,0,k_,k_)*perm;
    taij_.block(0,0,k_,k_) = perm *taij_.block(0,0,k_,k_)*perm;
    ta4ij_.block(0,0,k_,k_) = perm *ta4ij_.block(0,0,k_,k_)*perm;
    trapij_.block(0,0,k_,k_) = perm *trapij_.block(0,0,k_,k_)*perm;
    tapij_.block(0,0,k_,k_) = perm *tapij_.block(0,0,k_,k_)*perm;
    qaij_.block(0,0,k_,k_) = perm *qaij_.block(0,0,k_,k_)*perm;
    raij_.block(0,0,k_,k_) = perm *raij_.block(0,0,k_,k_)*perm;
    casimir_.block(0,0,k_,k_)      = perm *casimir_.block(0,0,k_,k_)*perm;
    casimirSpace_.block(0,0,k_,k_) = perm *casimirSpace_.block(0,0,k_,k_)*perm;
    casimirSpin_.block(0,0,k_,k_)  = perm *casimirSpin_.block(0,0,k_,k_)*perm;
    casimirIso_.block(0,0,k_,k_)   = perm *casimirIso_.block(0,0,k_,k_)*perm;
    casimirSpinIso_.block(0,0,k_,k_)   = perm *casimirSpinIso_.block(0,0,k_,k_)*perm;

    //swap the present element with the last one 
    //before *aold is the matrix in indx potion
    //after  *aold is what was the k_th matrix now in indx
    // now the last element (was indx'th) is A_.back()
    std::iter_swap(aold,std::prev(Au_.end(),1));
    std::iter_swap(std::next(alpha_.begin(),indx),std::prev(alpha_.end(),1));

    std::iter_swap(std::next(spin_.begin(),indx), std::prev(spin_.end(),1));
    std::iter_swap(std::next(isospin_.begin(),indx), std::prev(isospin_.end(),1));

    // up to now the dimension of the basis is k_
    // we take back the last basis element saving the old
    // values

    std::tuple<
      Eigen::VectorXd,std::pair<Eigen::MatrixXd,Eigen::VectorXd>,Eigen::VectorXd,
      Eigen::VectorXd, Eigen::VectorXd>
    oldOut = std::make_tuple(energy_.block(0,0,k_,1),Au_.back(),alpha_.back(),
        spin_.back(),isospin_.back());
    Au_.pop_back();
    alpha_.pop_back();
    spin_.pop_back();
    isospin_.pop_back();
    k_ = k_-1;

    // and set the new cij_ of the reduced basis
    Eigen::GeneralizedSelfAdjointEigenSolver <Eigen::MatrixXd> 
      es((haij_+vaij_).block(0,0,k_,k_),gij_.block(0,0,k_,k_));
    cij_.block(0,0,k_,k_) =  es.eigenvectors();
    // I've to put at zero the k_+1 elements...
    for(int i=0;i<k_+1;i++)
    {
      cij_(i,k_) = 0.0;
      cij_(k_,i) = 0.0;
    }
    
    //now we can test a new last element
    int l=state_;
    for(auto j=0; j< nOfProve; j++)
    {
      auto tmpOut = testElement();
      if(std::get<0>(tmpOut)(l) != std::get<0>(tmpOut)(l))
        continue;
//      std::cout << std::get<0>(tmpOut)(l) << " "<<  std::get<0>(oldOut)(l) << std::endl;
      if(std::get<0>(tmpOut)(l) < std::get<0>(oldOut)(l))
        oldOut = tmpOut;
    }
    //if(std::get<0>(oldOut)(l) != std::get<0>(oldOut)(l))
    //  continue;
    newElement(std::get<1>(oldOut),std::get<2>(oldOut),std::get<3>(oldOut),
        std::get<4>(oldOut));  

    indx++; // next one

  }


  return nOfChanges;
}

/*
 *--------------------------------------------------------------------------------------
 */

Eigen::MatrixXd 
Basis::calculateHamiltonian(Eigen::MatrixXd &a, Eigen::VectorXd &u,
    Eigen::VectorXd &lS, Eigen::VectorXd &lI)
{
  int indx_new = k_;
  int sizeNew = k_+1;

  // I put in the list the element i'm testing
  Au_.push_back(std::make_pair(a,u));
  spin_.push_back(lS);
  isospin_.push_back(lI);

  // Let's set to zero the new matrix elements..
  // this is in order to sum them up once we 
  // do symmetrization
  gij_.block(indx_new,0,1,sizeNew) = Eigen::MatrixXd::Zero(1,sizeNew);
  gij_.block(0,indx_new,sizeNew,1) = Eigen::MatrixXd::Zero(sizeNew,1);
  casimir_.block(indx_new,0,1,sizeNew) = Eigen::MatrixXd::Zero(1,sizeNew);
  casimir_.block(0,indx_new,sizeNew,1) = Eigen::MatrixXd::Zero(sizeNew,1);
  casimirSpace_.block(indx_new,0,1,sizeNew) = Eigen::MatrixXd::Zero(1,sizeNew);
  casimirSpace_.block(0,indx_new,sizeNew,1) = Eigen::MatrixXd::Zero(sizeNew,1);
  casimirSpin_.block(indx_new,0,1,sizeNew) = Eigen::MatrixXd::Zero(1,sizeNew);
  casimirSpin_.block(0,indx_new,sizeNew,1) = Eigen::MatrixXd::Zero(sizeNew,1);
  casimirIso_.block(indx_new,0,1,sizeNew) = Eigen::MatrixXd::Zero(1,sizeNew);
  casimirIso_.block(0,indx_new,sizeNew,1) = Eigen::MatrixXd::Zero(sizeNew,1);
  casimirSpinIso_.block(indx_new,0,1,sizeNew) = Eigen::MatrixXd::Zero(1,sizeNew);
  casimirSpinIso_.block(0,indx_new,sizeNew,1) = Eigen::MatrixXd::Zero(sizeNew,1);
  haij_.block(indx_new,0,1,sizeNew) = Eigen::MatrixXd::Zero(1,sizeNew);
  haij_.block(0,indx_new,sizeNew,1) = Eigen::MatrixXd::Zero(sizeNew,1);
  vaij_.block(indx_new,0,1,sizeNew) = Eigen::MatrixXd::Zero(1,sizeNew);
  vaij_.block(0,indx_new,sizeNew,1) = Eigen::MatrixXd::Zero(sizeNew,1);
  taij_.block(indx_new,0,1,sizeNew) = Eigen::MatrixXd::Zero(1,sizeNew);
  taij_.block(0,indx_new,sizeNew,1) = Eigen::MatrixXd::Zero(sizeNew,1);
  ta4ij_.block(indx_new,0,1,sizeNew) = Eigen::MatrixXd::Zero(1,sizeNew);
  ta4ij_.block(0,indx_new,sizeNew,1) = Eigen::MatrixXd::Zero(sizeNew,1);
  trapij_.block(indx_new,0,1,sizeNew) = Eigen::MatrixXd::Zero(1,sizeNew);
  trapij_.block(0,indx_new,sizeNew,1) = Eigen::MatrixXd::Zero(sizeNew,1);
  tapij_.block(indx_new,0,1,sizeNew) = Eigen::MatrixXd::Zero(1,sizeNew);
  tapij_.block(0,indx_new,sizeNew,1) = Eigen::MatrixXd::Zero(sizeNew,1);
  qaij_.block(indx_new,0,1,sizeNew) = Eigen::MatrixXd::Zero(1,sizeNew);
  qaij_.block(0,indx_new,sizeNew,1) = Eigen::MatrixXd::Zero(sizeNew,1);
  raij_.block(indx_new,0,1,sizeNew) = Eigen::MatrixXd::Zero(1,sizeNew);
  raij_.block(0,indx_new,sizeNew,1) = Eigen::MatrixXd::Zero(sizeNew,1);


  int kind = indata_.mapSymmetrization[indata_.kindSymmetrization];
  switch(kind)
  {
    case InputData::NONE:
    {
      add_hamiltonian_(a,u,lS,lI);
      add_casimir_(a,u,lS,lI);
    }
    break;

    case InputData::BOSONS:
    { 
      for(auto p : coo_.permutationsJacobi)
      {
        Eigen::MatrixXd b = p.transpose()*a*p;
        add_hamiltonian_(b,u,lS,lI);
        add_casimir_(b,u,lS,lI);
      }
    }
    break;

    case InputData::FERMIONS:
    { 
      int i=0;
      for(auto p : coo_.permutationsJacobi)
      {
        Eigen::MatrixXd b = p.transpose()*a*p;

        Eigen::VectorXd lpS = coo_.permutationsSpin[i].transpose()   *lS;
        Eigen::VectorXd lpI = coo_.permutationsIsospin[i].transpose()*lI;

        add_hamiltonian_(b,u,lpS,lpI,coo_.signPermutations[i]);
        add_casimir_(b,u,lpS,lpI);
        i++;
      }
    }

    break;
    default:
      std::cout << "Non ancora implementata questa simmetrizazione" << std::endl;
      abort();
  }//switch

  gij_.block(indx_new,0,1,sizeNew) /= factorialNp_;
  gij_.block(0,indx_new,sizeNew-1,1) /= factorialNp_;
  haij_.block(indx_new,0,1,sizeNew) /= factorialNp_;
  haij_.block(0,indx_new,sizeNew-1,1) /= factorialNp_;
  vaij_.block(indx_new,0,1,sizeNew) /= factorialNp_;
  vaij_.block(0,indx_new,sizeNew-1,1) /= factorialNp_;
  taij_.block(indx_new,0,1,sizeNew) /= factorialNp_;
  taij_.block(0,indx_new,sizeNew-1,1) /= factorialNp_;
  ta4ij_.block(indx_new,0,1,sizeNew) /= factorialNp_;
  ta4ij_.block(0,indx_new,sizeNew-1,1) /= factorialNp_;
  trapij_.block(indx_new,0,1,sizeNew) /= factorialNp_;
  trapij_.block(0,indx_new,sizeNew-1,1) /= factorialNp_;
  tapij_.block(indx_new,0,1,sizeNew) /= factorialNp_;
  tapij_.block(0,indx_new,sizeNew-1,1) /= factorialNp_;
  qaij_.block(indx_new,0,1,sizeNew) /= factorialNp_;
  qaij_.block(0,indx_new,sizeNew-1,1) /= factorialNp_;
  raij_.block(indx_new,0,1,sizeNew) /= factorialNp_;
  raij_.block(0,indx_new,sizeNew-1,1) /= factorialNp_;
  casimir_.block(indx_new,0,1,sizeNew) /= factorialNp_;
  casimir_.block(0,indx_new,sizeNew-1,1)/= factorialNp_;
  casimirSpace_.block(indx_new,0,1,sizeNew) /= factorialNp_;
  casimirSpace_.block(0,indx_new,sizeNew-1,1) /= factorialNp_;
  casimirSpin_.block(indx_new,0,1,sizeNew) /= factorialNp_;
  casimirSpin_.block(0,indx_new,sizeNew-1,1) /= factorialNp_;
  casimirIso_.block(indx_new,0,1,sizeNew) /= factorialNp_;
  casimirIso_.block(0,indx_new,sizeNew-1,1) /= factorialNp_;
  casimirSpinIso_.block(indx_new,0,1,sizeNew) /= factorialNp_;
  casimirSpinIso_.block(0,indx_new,sizeNew-1,1) /= factorialNp_;
   
  // Calculation of phiij_. The sizes of the matrices are k_+1
  phiij_.block(0,0,sizeNew,sizeNew) = cij_.transpose().block(0,0,sizeNew,sizeNew)
                              * gij_.block(0,0,sizeNew,sizeNew);

 
  // calculation of the normalization for the Gram-Schmidt 
  double tmp=0;
  for(auto i=0; i<indx_new; i++)
    tmp += phiij_(i,indx_new)*phiij_(i,indx_new);
  double n = sqrt(gij_(indx_new,indx_new)-tmp);

  // Calculation of cij_
  cij_(indx_new,indx_new) = 1/n;
  for(auto i=0; i<=sizeNew; i++)
  {
    if(i==indx_new)
      continue;
    cij_(i,indx_new)=0;
    for(auto j=0; j<sizeNew; j++)
    {
      if(j==indx_new)
        continue;
      cij_(i,indx_new) -= cij_(i,j)*phiij_(j,indx_new);
    }
    cij_(i,indx_new) /= n;
  }




  // I remove the state we are testing 
  Au_.pop_back();
  spin_.pop_back();
  isospin_.pop_back();


  // The hamiltonian in the new basis
  double quantoCasimir=0.;
  Eigen::MatrixXd h = cij_.transpose().block(0,0,sizeNew,sizeNew) 
                     //* (haij_+vaij_+taij_+qaij_).block(0,0,sizeNew,sizeNew) 
                    // * (haij_+vaij_+taij_).block(0,0,sizeNew,sizeNew) 
                    * (haij_+vaij_+taij_+ tapij_ + ta4ij_+ qaij_ + trapij_+ quantoCasimir*casimir_/indata_.energyUnits).block(0,0,sizeNew,sizeNew) 
                    * cij_.block(0,0,sizeNew,sizeNew);
                    // + .2*15*Eigen::MatrixXd::Identity(sizeNew,sizeNew);

  return h;
}



double 
Basis::evaluatePotential_(int pair, 
    double norma, double c, double rho, double gamma, double gammap,
    Eigen::VectorXd lpS, Eigen::VectorXd lS, Eigen::VectorXd lpI, Eigen::VectorXd lI, 
    int kind)
{

  Factorial fact;

  double value=0;

  switch(kind)
  {
    case InputData::COULOMB:
    {
      double e = 1.44/indata_.energyUnits/indata_.lengthUnits;

      // coordinates integration, see. (A.130) and (A.132) SV
      double partialpot =0;
      for(int n=0; n<=L_; n++)
      {
        partialpot += 1./fact(L_-n)
          * pow(gamma*gammap/c/rho,n)
          * sqrt(2*c/M_PI)*pow(-1,n)/(2.*n+1.)/fact(n);
      }
      // charge operator acting on isospin
      double charge = lpI.transpose()*coo_.chargeOperator[pair]*lI;
      charge /= lpI.transpose()*lI; // because it on norma...
           
      value += partialpot*fact(L_)*norma* e*charge ;
    }
    break; // real gaussian with spin and isospin

    // gaussian for the four spin/isospin channels
    case InputData::REALGAUSSIAN:
    {
      // Gaussian Parameters  V_S_T
      double v00 = indata_.twoBodyParameters[pair][0]/indata_.energyUnits;
      double r00 = indata_.twoBodyParameters[pair][1]/indata_.lengthUnits;

      double v01 = indata_.twoBodyParameters[pair][2]/indata_.energyUnits;
      double r01 = indata_.twoBodyParameters[pair][3]/indata_.lengthUnits;

      double v10 = indata_.twoBodyParameters[pair][4]/indata_.energyUnits;
      double r10 = indata_.twoBodyParameters[pair][5]/indata_.lengthUnits;

      double v11 = indata_.twoBodyParameters[pair][6]/indata_.energyUnits;
      double r11 = indata_.twoBodyParameters[pair][7]/indata_.lengthUnits;

      double ps0,ps1,pt0,pt1;
      std::tie(ps0,ps1) = spinProjector(lpS,lS,pair);
      std::tie(pt0,pt1) = isospinProjector(lpI,lI,pair);


      value=0;
      // S=0 T=0
      if(v00!=0)
      {
        double partialpot =0;
        for(int n=0; n<=L_; n++)
        {
          partialpot += 1./fact(L_-n)
            * pow(gamma*gammap/c/rho,n)
            * gaussian_(n,c,v00,r00);
        }
        value += partialpot*fact(L_)*norma* ps0*pt0;
      }
      // S=0 T=1
      if(v01!=0)
      {
        double partialpot =0;
        for(int n=0; n<=L_; n++)
        {
          partialpot += 1./fact(L_-n)
            * pow(gamma*gammap/c/rho,n)
            * gaussian_(n,c,v01,r01);
        }
        value += partialpot*fact(L_)*norma* ps0*pt1;
      }
      // S=1 T=0
      if(v10!=0)
      {
        double partialpot =0;
        for(int n=0; n<=L_; n++)
        {
          partialpot += 1./fact(L_-n)
            * pow(gamma*gammap/c/rho,n)
            * gaussian_(n,c,v10,r10);
        }
        value += partialpot*fact(L_)*norma* ps1*pt0;
      }
      // S=1 T=1
      if(v11!=0)
      {
        double partialpot =0;
        for(int n=0; n<=L_; n++)
        {
          partialpot += 1./fact(L_-n)
            * pow(gamma*gammap/c/rho,n)
            * gaussian_(n,c,v11,r11);
        }
        value += partialpot*fact(L_)*norma* ps1*pt1;
      }

    }
    break; // real gaussian with spin and isospin


    // This is form Minnesota case - see PRC 52, 2885 table I 
    // There are spin/isospin and space Exchange operators...
    case InputData::MINNESOTA:
    {
      // Gaussian parameters
      double v1 = 200/indata_.energyUnits;
      double r1 = 1/sqrt(1.487);
      double v2 = -178/indata_.energyUnits;
      double r2 = 1/sqrt(0.639);
      double v3 = -91.85/indata_.energyUnits;
      double r3 = 1/sqrt(0.465);

      double w1 = 0.5;
      double w2 = 0.25;
      double w3 = 0.25;

      double m1 = 0.5;
      double m2 = 0.25;
      double m3 = 0.25;

      double b1 = 0;
      double b2 = 0.25;
      double b3 = -0.25;

      double h1 = 0;
      double h2 = 0.25;
      double h3 = -0.25;
      
      // Gaussian Parameters  V_S_T
      double v00_1 =  v1*(w1-b1+h1-m1);
      double v01_1 =  v1*(w1-b1-h1+m1);
      double v10_1 =  v1*(w1+b1+h1+m1);
      double v11_1 =  v1*(w1+b1-h1-m1);

      double v00_2 =  v2*(w2-b2+h2-m2);
      double v01_2 =  v2*(w2-b2-h2+m2);
      double v10_2 =  v2*(w2+b2+h2+m2);
      double v11_2 =  v2*(w2+b2-h2-m2);

      double v00_3 =  v3*(w3-b3+h3-m3);
      double v01_3 =  v3*(w3-b3-h3+m3);
      double v10_3 =  v3*(w3+b3+h3+m3);
      double v11_3 =  v3*(w3+b3-h3-m3);


      double ps0,ps1,pt0,pt1;
      std::tie(ps0,ps1) = spinProjector(lpS,lS,pair);
      std::tie(pt0,pt1) = isospinProjector(lpI,lI,pair);

      value=0;
      // S=0 T=0
      double partialpot_1 =0;
      double partialpot_2 =0;
      double partialpot_3 =0;
      for(int n=0; n<=L_; n++)
      {
        partialpot_1 += 1./fact(L_-n)
          * pow(gamma*gammap/c/rho,n)
          * gaussian_(n,c,v00_1,r1);
        partialpot_2 += 1./fact(L_-n)
          * pow(gamma*gammap/c/rho,n)
          * gaussian_(n,c,v00_2,r2);
        partialpot_3 += 1./fact(L_-n)
          * pow(gamma*gammap/c/rho,n)
          * gaussian_(n,c,v00_3,r3);
      }
      value += (partialpot_1 + partialpot_2 + partialpot_3)*fact(L_)*norma* ps0*pt0;
      // S=0 T=1
      partialpot_1 =0;
      partialpot_2 =0;
      partialpot_3 =0;
      for(int n=0; n<=L_; n++)
      {
        partialpot_1 += 1./fact(L_-n)
          * pow(gamma*gammap/c/rho,n)
          * gaussian_(n,c,v01_1,r1);
        partialpot_2 += 1./fact(L_-n)
          * pow(gamma*gammap/c/rho,n)
          * gaussian_(n,c,v01_2,r2);
        partialpot_3 += 1./fact(L_-n)
          * pow(gamma*gammap/c/rho,n)
          * gaussian_(n,c,v01_3,r3);
      }
      value += (partialpot_1 + partialpot_2 + partialpot_3)*fact(L_)*norma* ps0*pt1;
      // S=1 T=0
      partialpot_1 =0;
      partialpot_2 =0;
      partialpot_3 =0;
      for(int n=0; n<=L_; n++)
      {
        partialpot_1 += 1./fact(L_-n)
          * pow(gamma*gammap/c/rho,n)
          * gaussian_(n,c,v10_1,r1);
        partialpot_2 += 1./fact(L_-n)
          * pow(gamma*gammap/c/rho,n)
          * gaussian_(n,c,v10_2,r2);
        partialpot_3 += 1./fact(L_-n)
          * pow(gamma*gammap/c/rho,n)
          * gaussian_(n,c,v10_3,r3);
      }
      value += (partialpot_1 + partialpot_2 + partialpot_3)*fact(L_)*norma* ps1*pt0;
      // S=1 T=1
      partialpot_1 =0;
      partialpot_2 =0;
      partialpot_3 =0;
      for(int n=0; n<=L_; n++)
      {
        partialpot_1 += 1./fact(L_-n)
          * pow(gamma*gammap/c/rho,n)
          * gaussian_(n,c,v11_1,r1);
        partialpot_2 += 1./fact(L_-n)
          * pow(gamma*gammap/c/rho,n)
          * gaussian_(n,c,v11_2,r2);
        partialpot_3 += 1./fact(L_-n)
          * pow(gamma*gammap/c/rho,n)
          * gaussian_(n,c,v11_3,r3);
      }
      value += (partialpot_1 + partialpot_2 + partialpot_3)*fact(L_)*norma* ps1*pt1;
    }
    break; // Minnesota potential

    case InputData::GAUSSIAN:
    {
      double v0 = indata_.twoBodyParameters[pair][0]/indata_.energyUnits;
      double r0 = indata_.twoBodyParameters[pair][1]/indata_.lengthUnits;
      double partialpot =0;
      for(int n=0; n<=L_; n++)
      {
        partialpot += 1./fact(L_-n)
          * pow(gamma*gammap/c/rho,n)
          * gaussian_(n,c,v0,r0);
      }
      value = partialpot*fact(L_)*norma;
    }
    break; // Gaussian

    case InputData::VOLKOV:
    {
      double v1 = indata_.twoBodyParameters[pair][0];
      double r1 = indata_.twoBodyParameters[pair][1];
      double v2 = indata_.twoBodyParameters[pair][2];
      double r2 = indata_.twoBodyParameters[pair][3];
      // switch n <= comes from angular momentum
      double partialpot =0;
      for(int n=0; n<=L_; n++)
      {
        partialpot += 1./fact(L_-n)
          * pow(gamma*gammap/c/rho,n)
          * gaussian_(n,c,v1,r1);
      }
      value = partialpot*fact(L_)*norma;

      partialpot =0;
      for(int n=0; n<=L_; n++)
      {
        partialpot += 1./fact(L_-n)
          * pow(gamma*gammap/c/rho,n)
          * gaussian_(n,c,v2,r2);
      }
      value += partialpot*fact(L_)*norma;
     }
     break; // Volkov


    // The default case just go back to the old code... Gaussian and 
    // Volkov, without spin and other things... I'm not sure this 
    // is the best way to handle things
    default:
     /*
    {
      double partialpot =0;
      for(int n=0; n<=L_; n++)
      {
        partialpot += 1./fact(L_-n)
          * pow(gamma*gammap/c/rho,n)
          * J_(n,c,pair,kind);
      }
      value = partialpot*fact(L_)*norma;
    }
    */
    break; // end default
  } // end swith kind of potential

  return value;
}
          

// function which integrates the potential (A.129) p.282 SV
/*
double
Basis::J_(int n, double c, int pair, int kind)
{
  double value=0;

  // switch the potential
  switch(kind)
  {
    case InputData::GAUSSIAN:
    {
      double v0 = indata_.twoBodyParameters[pair][0];
      double r0 = indata_.twoBodyParameters[pair][1];
     value = gaussian_(n,c,v0,r0);
    }//Gaussian
    break;

    case InputData::VOLKOV:
    {
      double v1 = indata_.twoBodyParameters[pair][0];
      double r1 = indata_.twoBodyParameters[pair][1];
      double v2 = indata_.twoBodyParameters[pair][2];
      double r2 = indata_.twoBodyParameters[pair][3];
      // switch n <= comes from angular momentum
      switch(n)
      {
        case 0:
        {
          value = v1/pow(1+2./c/r1/r1,1.5)
                + v2/pow(1+2./c/r2/r2,1.5);
        }
        break;
        case 1:
        {
          value = -v1* 2.*pow(c,1.5)*pow(r1,3)/pow(2+c*r1*r1,2.5)
                  -v2* 2.*pow(c,1.5)*pow(r2,3)/pow(2+c*r2*r2,2.5);
        }
        break;
        case 2:
        {
          value = v1* 2*pow(c,1.5)*pow(r1,3)/pow(2+c*r1*r1,3.5)
                + v2* 2*pow(c,1.5)*pow(r2,3)/pow(2+c*r2*r2,3.5);
        }
        break;
      }
     }
     break; // Volkov
   
  }// switch kind of potential
  return value;
}
*/

/*
 * Spin potential - one component
 */

double
Basis::gaussian_(int n,double c, double v0, double r0)
{
      // n = 3; Integrate[(1/Sqrt[Pi]/(2*n+1)!)*Exp[-2*(x^2/c/r0^2) - x^2]*
      //   HermiteH[1, x]*HermiteH[2*n + 1, x], {x, 0, Infinity}, 
      //    Assumptions -> {c > 0, r0 > 0}]
  double value=0;
  switch(n)
  {
    case 0:
    {
      value = v0/pow(1+2./c/r0/r0,1.5);
    }
    break;

    case 1:
    { 
      value = -v0* 2.*pow(c,1.5)*pow(r0,3)/pow(2+c*r0*r0,2.5);
    }
    break;

    case 2:
    { 
      value = v0* 2*pow(c,1.5)*pow(r0,3)/pow(2+c*r0*r0,3.5);
    }
    break;
  }

  return value; 
}

/*
 * Spin
 */
double
Basis::spinExange(Eigen::VectorXd lp, Eigen::VectorXd l, int pair)
{
  double value=1;
  if(S_>=0) // If is negative I do calculations without spin
  {
    value = lp.transpose()*coo_.traspositionSpin[pair]*l;
  }// end if  there is spin

  return value;
}
std::pair<double,double>
Basis::spinProjector(Eigen::VectorXd lp, Eigen::VectorXd l, int pair)
{
  std::pair<double,double> value= std::make_pair(1,1);
  if(S_>=0) // If is negative I do calculations without spin
  {
    double zero = lp.transpose()*coo_.projectorSpin[pair].first*l;
    double uno = lp.transpose()*coo_.projectorSpin[pair].second*l;
    zero /= lp.transpose()*l;
    uno  /= lp.transpose()*l;
    value = std::make_pair(zero,uno);
  }// end if  there is spin

  return value;
}

/*
 * Isospin
 */
double
Basis::isospinExange(Eigen::VectorXd lp, Eigen::VectorXd l, int pair)
{
  double value=1;
  if(T_>=0) // If is negative I do calculations without isospin
  {
    value = lp.transpose()*coo_.traspositionIsospin[pair]*l;
  }// end if there is the isospin

  return value;
}
std::pair<double,double>
Basis::isospinProjector(Eigen::VectorXd lp, Eigen::VectorXd l, int pair)
{
  std::pair<double,double> value= std::make_pair(1,1);
  if(T_>=0) // If is negative I do calculations without spin
  {
    double zero = lp.transpose()*coo_.projectorIsospin[pair].first*l;
    double uno = lp.transpose()*coo_.projectorIsospin[pair].second*l;
    zero /= lp.transpose()*l;
    uno  /= lp.transpose()*l;
    value = std::make_pair(zero,uno);
  }// end if  there is spin

  return value;
}



/*
 * Simplex
 */

std::pair<
  std::vector<double>, // the x which minimize
  Basis::allVariables> // a,u,spin,isospin
Basis::simplex_(
     //std::vector<double> init,    //initial guess of the parameters
     allVariables input,
     std::vector<std::vector<double> > x ,
     double tol, //termination criteria
     //x: The Simplex
     int iterations)//iteration step number
{    

    allVariables out; 
    //  we use the old values as starting point
    //  for the simplex
    std::vector<double> init;
    // alphas
    auto tmp = std::get<2>(input);
    for(int i=0; i<tmp.size(); i++)
      init.push_back(tmp[i]);
    /*
    // 
    tmp = std::get<1>(oldOut).second;
    for(int i=0; i<tmp.size(); i++)
      init.push_back(tmp[i]);
    // spin
    tmp = std::get<3>(oldOut);
    for(int i=0; i<tmp.size(); i++)
      init.push_back(tmp[i]);
    // isospin
    tmp = std::get<4>(oldOut);
    for(int i=0; i<tmp.size(); i++)
      init.push_back(tmp[i]);
     */


    int N=init.size();                         //space dimension
    const double a=1.0, b=1.0, g=0.5, h=0.5;   //coefficients
                                               //a: reflection  -> xr  
                                               //b: expansion   -> xe 
                                               //g: contraction -> xc
                                               //h: full contraction to x1
    std::vector<double> xcentroid_old(N,0);   //simplex center * (N+1)
    std::vector<double> xcentroid_new(N,0);   //simplex center * (N+1)
    std::vector<double> vf(N+1,0);            //f evaluated at simplex vertexes       
    int x1=0, xn=0, xnp1=0;         //x1:   f(x1) = min { f(x1), f(x2)...f(x_{n+1} }
                                    //xnp1: f(xnp1) = max { f(x1), f(x2)...f(x_{n+1} }
                                    //xn:   f(xn)<f(xnp1) && f(xn)> all other f(x_i)
    int cnt=0; //iteration step number


    if(x.size()==0) //if no initial simplex is specified
      { //construct the trial simplex
	//based upon the initial guess parameters
      std::vector<double> del( init );
      std::transform(del.begin(), del.end(), del.begin(), 
		     std::bind2nd( std::divides<double>() , 20) );//'20' is picked 
                                                             //assuming initial trail close to true
      
      for(int i=0; i<N; ++i){
	std::vector<double> tmp( init );
	tmp[i] +=  del[i];
	x.push_back( tmp );
      }
      x.push_back(init);//x.size()=N+1, x[i].size()=N
      
      //xcentriod
      std::transform(init.begin(), init.end(), 
		xcentroid_old.begin(), std::bind2nd(std::multiplies<double>(), N+1) );
      }//constructing the simplex finished
    
    //optimization begins
    for(cnt=0; cnt<iterations; ++cnt){

      for(int i=0;i<N+1;++i){
        std::tie(vf[i],out) = testSimplex(x[i],input);
      }
      
      x1=0; xn=0; xnp1=0;//find index of max, second max, min of vf.
      
      for(unsigned int i=0;i<vf.size();++i){
	if(vf[i]<vf[x1]){
	  x1=i;
	}
	if(vf[i]>vf[xnp1]){
	  xnp1=i;
	}
      }
      
      xn=x1;
      
      for(unsigned int i=0; i<vf.size();++i){ 
	if( vf[i]<vf[xnp1] && vf[i]>vf[xn] )
	  xn=i;
      }
      //x1, xn, xnp1 are found

      std::vector<double> xg(N, 0);//xg: centroid of the N best vertexes
      for(unsigned int i=0; i<x.size(); ++i){
	if((int)i!=xnp1)
	  std::transform(xg.begin(), xg.end(), x[i].begin(), xg.begin(), std::plus<double>() );
      }
      std::transform(xg.begin(), xg.end(), 
		x[xnp1].begin(), xcentroid_new.begin(), std::plus<double>());
      std::transform(xg.begin(), xg.end(), xg.begin(), 
		std::bind2nd(std::divides<double>(), N) );
      //xg found, xcentroid_new updated

      //termination condition
      double diff=0;          //calculate the difference of the simplex centers
                         //see if the difference is less than the termination criteria
      for(int i=0; i<N; ++i)     
	diff += fabs(xcentroid_old[i]-xcentroid_new[i]);

      if (diff/N < tol) break;              //terminate the optimizer
      else xcentroid_old.swap(xcentroid_new); //update simplex center
      
      //reflection:
      std::vector<double> xr(N,0); 
      for( int i=0; i<N; ++i)
	xr[i]=xg[i]+a*(xg[i]-x[xnp1][i]);
      //reflection, xr found
      
      double fxr;
      std::tie(fxr,out)=testSimplex(xr,input);//record function at xr
      
      if(vf[x1]<=fxr && fxr<=vf[xn])
	std::copy(xr.begin(), xr.end(), x[xnp1].begin() );
      
      //expansion:
      else if(fxr<vf[x1]){
	std::vector<double> xe(N,0);
	for( int i=0; i<N; ++i)
	  xe[i]=xr[i]+b*(xr[i]-xg[i]);
  double tmpDouble;
  std::tie(tmpDouble,out) = testSimplex(xe,input);
	if( tmpDouble < fxr )
	  std::copy(xe.begin(), xe.end(), x[xnp1].begin() );
	else
	  std::copy(xr.begin(), xr.end(), x[xnp1].begin() );
      }//expansion finished,  xe is not used outside the scope
      
      //contraction:
      else if( fxr > vf[xn] ){
	std::vector<double> xc(N,0);
	for( int i=0; i<N; ++i)
	  xc[i]=xg[i]+g*(x[xnp1][i]-xg[i]);
  double tmpDouble;
  std::tie(tmpDouble,out) = testSimplex(xc,input);
	if( tmpDouble  < vf[xnp1] )
	  std::copy(xc.begin(), xc.end(), x[xnp1].begin() );
	else{
	  
	  for( unsigned int i=0; i<x.size(); ++i ){
	    if( (int)i!=x1 ){ 
	      for(int j=0; j<N; ++j) 
		x[i][j] = x[x1][j] + h * ( x[i][j]-x[x1][j] );
	    }
	  }
	  
	}
      }//contraction finished, xc is not used outside the scope

    }//optimization is finished

    if(cnt==iterations){//max number of iteration achieves before tol is satisfied
      std::cout<<"Iteration limit achieves, result may not be optimal"<<std::endl;
    }
    
    double tmpDouble;
    std::tie(tmpDouble,out) = testSimplex(x[x1],input);
    // DEBUG
    /*
    for(auto v : x[x1])
      std::cout << "valore di alpha = " << 1./sqrt(v) << std::endl;
    std::cout << tmpDouble << std::endl;
    */
    return std::make_pair(x[x1],out);
}
//
//-----------------------------------------------------------------------------
void 
Basis::add_hamiltonian_(Eigen::MatrixXd &a, Eigen::VectorXd &u,
Eigen::VectorXd &lS, Eigen::VectorXd &lI,int sign)
{
  /*
   * Calculation of the hamiltonian in the
   * gaussian basis
   */
  int indx_new = k_;

  DoubleFactorial dfact;
  double scalConst = pow(2*M_PI,1.5*(np_-1)) * dfact(2*L_+1)/4./M_PI;
  int nofpairs = coo_.pairs.size();

  /*
   * start the loop over the old gaussian 
   */
  int indx_i=0;
  auto lpS=spin_.begin();
  auto lpI=isospin_.begin();
  for(auto aui : Au_)
  {
    /*
     * Gij - The Gaussian metric
     * General formulas (A7) p.249 SV
     */ 
    

    // Definitions from (A2) p.248 SV
    double det_b = (aui.first+a).determinant();
    Eigen::MatrixXd binv = (aui.first+a).inverse();
    double rho = 1; // if L=0 is useless, just for numerical stability
    if(L_>0)
      rho = aui.second.transpose()*binv*u;
    double scalarProduct = scalConst/pow(det_b,1.5)*pow(rho,L_);
    // I need also spin/isospin scalar product.... 
    // this is something to do better....
    double normST = 1;
    if(T_>=0)
      normST *= lpI->transpose()*lI;
    if(S_>=0)
      normST *= lpS->transpose()*lS;
    

    scalarProduct *= normST;
    gij_(indx_i,indx_new) += sign*scalarProduct;
    gij_(indx_new,indx_i) = gij_(indx_i,indx_new);
    // kinetic energy

    // additional numbers - (A8) p.249 SV
    double Q = 0; // if L=0 useless
    if(L_>0)
      Q = 2*aui.second.transpose()*binv*a*coo_.Lambda*aui.first*binv*u;
    double R = 3.*(a*binv*aui.first*coo_.Lambda).trace();
 
    // (A11)
    haij_(indx_i,indx_new) += sign*1/2.*scalarProduct *(R+L_*Q/rho);
    haij_(indx_new,indx_i) = haij_(indx_i,indx_new) ;

    /*
     * Harmonic Trap energy
     */
    if(indata_.ifHarmonicTrap)
    {
      double vtrap = 3. /indata_.harmonicLength /indata_.harmonicLength *
        scalarProduct * (binv*coo_.Omega).trace();

      trapij_(indx_i,indx_new) += sign*vtrap;
      trapij_(indx_new,indx_i) = trapij_(indx_i,indx_new) ;
    }// end ifHarmonicTrap

    /*
     * Square Radius2 \sum_i r_i^2/N
     */
    double vtrap = 3.*
      scalarProduct * (binv*coo_.Radius2).trace();

    raij_(indx_i,indx_new) += sign*vtrap;
    raij_(indx_new,indx_i) = raij_(indx_i,indx_new) ;
    /*
     *---- Potential energy
     */
    double pot = 0;
    double qpot = 0;
    for(int i=0; i<nofpairs ; i++)
    {
      // additional coefficient which depend on the pair
      // (A18) p.252 SV
      double cij = coo_.W.row(i)*binv*coo_.W.row(i).transpose();
      cij = 1/cij;
      double gamma = 0;
      double gammap =0;
      if(L_>0)
      {
        gamma = cij*coo_.W.row(i)*binv*u;
        gammap= cij*coo_.W.row(i)*binv*aui.second;
      }

      
      // Here the matrix elements come from (A.130) p.282 SV
     // see also the special case (7.6) p.127 SV
      pot += evaluatePotential_(i,
          scalarProduct,cij,rho,gamma ,gammap,
          *lpS,lS, *lpI,lI,
                       indata_.mapPotential[indata_.kindPotential]);

      if(indata_.ifCoulomb)
        qpot += evaluatePotential_(i,
            scalarProduct,cij,rho,gamma ,gammap,
            *lpS,lS, *lpI,lI,
                         indata_.mapPotential["COULOMB"]);


    } // sum over the pairs 


    // questo da mettere dentro il potenziale
    //pot *= gij_(indx_new,indx_i);
      
    // three-body potential
    if(indata_.ifThreeBody)
    {
      double tmp3Pot=0;
      double V0 = indata_.threeBodyParameters[0]/indata_.energyUnits;
      double r02 = indata_.threeBodyParameters[1]*indata_.threeBodyParameters[1]
        /indata_.lengthUnits/indata_.lengthUnits;
      //sum over the triplets
      for(auto t : coo_.triplets)
      {
        // define the matrix Q 
        double q11 = 2./r02 * coo_.W.row(std::get<0>(t))*binv*coo_.W.row(std::get<0>(t)).transpose();
        double q12 = 2./r02 * coo_.W.row(std::get<0>(t))*binv*coo_.W.row(std::get<1>(t)).transpose();
        double q13 = 2./r02 * coo_.W.row(std::get<0>(t))*binv*coo_.W.row(std::get<2>(t)).transpose();

        double q22 = 2./r02 * coo_.W.row(std::get<1>(t))*binv*coo_.W.row(std::get<1>(t)).transpose();
        double q23 = 2./r02 * coo_.W.row(std::get<1>(t))*binv*coo_.W.row(std::get<2>(t)).transpose();

        double q33 = 2./r02 * coo_.W.row(std::get<2>(t))*binv*coo_.W.row(std::get<2>(t)).transpose();


        double det=0;
        int kind = indata_.mapThreeBody[indata_.kindThreeBody];
        switch(kind)
        {
          case InputData::HYPERRADIUS:
          {
            det = (1+q11)*((1+q22)*(1+q33)-q23*q23) - q12*(q12*(1+q33)-q23*q13)
            + q13*(q12*q23-(1+q22)*q13);
            det = pow(det,-1.5);
          }
          break;
          case InputData::TWOSIDES:
          {
            det = pow((1+q11)*(1+q22)-q12*q12,-1.5) + 
                  pow((1+q11)*(1+q33)-q13*q13,-1.5) +
                  pow((1+q22)*(1+q33)-q23*q23,-1.5) ; 
          }
          break;
          default : 
            std::cout << "DEFAULT in 3body" << std::endl;
            abort();
        }

        tmp3Pot += det;
        //tmp3Pot += pow(det,-1.5);
      }//sum over the triplets
      double value = V0*scalarProduct * tmp3Pot;
      // DEBUG
      //pot += value;
      taij_(indx_i,indx_new) += sign*value;
      taij_(indx_new,indx_i) = taij_(indx_i,indx_new);
    }//end if3body

    // three-body potential spin/isospin dependent
    if(indata_.ifThreeBodySpin)
    {
      double tmp3Pot=0;
      double V0 = indata_.threeBodyParametersSpin[0]/indata_.energyUnits;
      double r02 = indata_.threeBodyParametersSpin[1]*indata_.threeBodyParametersSpin[1]
        /indata_.lengthUnits/indata_.lengthUnits;
      //sum over the triplets
      for(auto t : coo_.triplets)
      {
        // define the matrix Q 
        double q11 = 2./r02 * coo_.W.row(std::get<0>(t))*binv*coo_.W.row(std::get<0>(t)).transpose();
        double q12 = 2./r02 * coo_.W.row(std::get<0>(t))*binv*coo_.W.row(std::get<1>(t)).transpose();
        double q13 = 2./r02 * coo_.W.row(std::get<0>(t))*binv*coo_.W.row(std::get<2>(t)).transpose();

        double q22 = 2./r02 * coo_.W.row(std::get<1>(t))*binv*coo_.W.row(std::get<1>(t)).transpose();
        double q23 = 2./r02 * coo_.W.row(std::get<1>(t))*binv*coo_.W.row(std::get<2>(t)).transpose();

        double q33 = 2./r02 * coo_.W.row(std::get<2>(t))*binv*coo_.W.row(std::get<2>(t)).transpose();


        double det=0;
        int kind = indata_.mapThreeBody[indata_.kindThreeBody];
      // projectors on the opposite/empty side
      // size 0
      double ps0_1,ps1_1,pt0_1,pt1_1;
      std::tie(ps0_1,ps1_1) = spinProjector(*lpS,lS,std::get<0>(t));
      std::tie(pt0_1,pt1_1) = isospinProjector(*lpI,lI,std::get<0>(t));
      // size 1
      double ps0_2,ps1_2,pt0_2,pt1_2;
      std::tie(ps0_2,ps1_2) = spinProjector(*lpS,lS,std::get<1>(t));
      std::tie(pt0_2,pt1_2) = isospinProjector(*lpI,lI,std::get<1>(t));
      // size 2
      double ps0_3,ps1_3,pt0_3,pt1_3;
      std::tie(ps0_3,ps1_3) = spinProjector(*lpS,lS,std::get<2>(t));
      std::tie(pt0_3,pt1_3) = isospinProjector(*lpI,lI,std::get<2>(t));

        switch(kind)
        {
          case InputData::TWOSIDES:
          {
            // 0 0
            det = pow((1+q11)*(1+q22)-q12*q12,-1.5) * ps0_3*pt0_3 + 
                  pow((1+q11)*(1+q33)-q13*q13,-1.5) * ps0_2*pt0_2 +
                  pow((1+q22)*(1+q33)-q23*q23,-1.5) * ps0_1*pt0_1; 
            /*
            // 0 1
            det = pow((1+q11)*(1+q22)-q12*q12,-1.5) * ps0_3*pt1_3 + 
                  pow((1+q11)*(1+q33)-q13*q13,-1.5) * ps0_2*pt1_2 +
                  pow((1+q22)*(1+q33)-q23*q23,-1.5) * ps0_1*pt1_1; 

            */
            // 1 0
            det = pow((1+q11)*(1+q22)-q12*q12,-1.5) * ps1_3*pt0_3 + 
                  pow((1+q11)*(1+q33)-q13*q13,-1.5) * ps1_2*pt0_2 +
                  pow((1+q22)*(1+q33)-q23*q23,-1.5) * ps1_1*pt0_1; 

            //1 1 
            /*
            det = pow((1+q11)*(1+q22)-q12*q12,-1.5) * ps1_3*pt1_3 + 
                  pow((1+q11)*(1+q33)-q13*q13,-1.5) * ps1_2*pt1_2 +
                  pow((1+q22)*(1+q33)-q23*q23,-1.5) * ps1_1*pt1_1; 
             */

            /*
            det = pow((1+q11)*(1+q22)-q12*q12,-1.5) * 0.5*(ps1_1*pt1_1 +  ps1_2*pt1_2) +
                  pow((1+q11)*(1+q33)-q13*q13,-1.5) * 0.5*(ps1_1*pt1_1 +  ps1_3*pt1_3) +
                  pow((1+q22)*(1+q33)-q23*q23,-1.5) * 0.5*(ps1_2*pt1_2 +  ps1_3*pt1_3);
            */

            // on sides 
            /*
            det = pow((1+q11)*(1+q22)-q12*q12,-1.5) * ps1_3*pt1_3  +
                  pow((1+q11)*(1+q33)-q13*q13,-1.5) * ps1_2*pt1_2  +
                  pow((1+q22)*(1+q33)-q23*q23,-1.5) * ps1_1*pt1_1 ;
            */

          }
          break;
          default : 
            std::cout << "DEFAULT in 3body" << std::endl;
            abort();
        }

        tmp3Pot += det;
        //tmp3Pot += pow(det,-1.5);
      }//sum over the triplets
      double value = V0*scalarProduct * tmp3Pot;
      // DEBUG
      //pot += value;
      tapij_(indx_i,indx_new) += sign*value;
      tapij_(indx_new,indx_i) = tapij_(indx_i,indx_new);
    }//end if3body


    // four-body potential
    if(indata_.ifFourBody)
    {
      double tmp4Pot=0;
      double V0 = indata_.fourBodyParameters[0]/indata_.energyUnits;
      double r02 = indata_.fourBodyParameters[1]*indata_.fourBodyParameters[1]
        /indata_.lengthUnits/indata_.lengthUnits;
      //sum over the triplets
      for(auto q : coo_.quadruplets)
      {
        // define the matrix Q 
        double q11 = 2./r02 * coo_.W.row(std::get<0>(q))*binv*coo_.W.row(std::get<0>(q)).transpose();
        double q12 = 2./r02 * coo_.W.row(std::get<0>(q))*binv*coo_.W.row(std::get<1>(q)).transpose();
        double q13 = 2./r02 * coo_.W.row(std::get<0>(q))*binv*coo_.W.row(std::get<2>(q)).transpose();
        double q14 = 2./r02 * coo_.W.row(std::get<0>(q))*binv*coo_.W.row(std::get<3>(q)).transpose();
        double q15 = 2./r02 * coo_.W.row(std::get<0>(q))*binv*coo_.W.row(std::get<4>(q)).transpose();
        double q16 = 2./r02 * coo_.W.row(std::get<0>(q))*binv*coo_.W.row(std::get<5>(q)).transpose();

        double q22 = 2./r02 * coo_.W.row(std::get<1>(q))*binv*coo_.W.row(std::get<1>(q)).transpose();
        double q23 = 2./r02 * coo_.W.row(std::get<1>(q))*binv*coo_.W.row(std::get<2>(q)).transpose();
        double q24 = 2./r02 * coo_.W.row(std::get<1>(q))*binv*coo_.W.row(std::get<3>(q)).transpose();
        double q25 = 2./r02 * coo_.W.row(std::get<1>(q))*binv*coo_.W.row(std::get<4>(q)).transpose();
        double q26 = 2./r02 * coo_.W.row(std::get<1>(q))*binv*coo_.W.row(std::get<5>(q)).transpose();

        double q33 = 2./r02 * coo_.W.row(std::get<2>(q))*binv*coo_.W.row(std::get<2>(q)).transpose();
        double q34 = 2./r02 * coo_.W.row(std::get<2>(q))*binv*coo_.W.row(std::get<3>(q)).transpose();
        double q35 = 2./r02 * coo_.W.row(std::get<2>(q))*binv*coo_.W.row(std::get<4>(q)).transpose();
        double q36 = 2./r02 * coo_.W.row(std::get<2>(q))*binv*coo_.W.row(std::get<5>(q)).transpose();

        double q44 = 2./r02 * coo_.W.row(std::get<3>(q))*binv*coo_.W.row(std::get<3>(q)).transpose();
        double q45 = 2./r02 * coo_.W.row(std::get<3>(q))*binv*coo_.W.row(std::get<4>(q)).transpose();
        double q46 = 2./r02 * coo_.W.row(std::get<3>(q))*binv*coo_.W.row(std::get<5>(q)).transpose();

        double q55 = 2./r02 * coo_.W.row(std::get<4>(q))*binv*coo_.W.row(std::get<4>(q)).transpose();
        double q56 = 2./r02 * coo_.W.row(std::get<4>(q))*binv*coo_.W.row(std::get<5>(q)).transpose();

        double q66 = 2./r02 * coo_.W.row(std::get<5>(q))*binv*coo_.W.row(std::get<5>(q)).transpose();

        /*
        Eigen::MatrixXd qmatrix (6,6);
        qmatrix << q11,q12,q13,q14,q15,q16,
                   q12,q22,q23,q24,q25,q26,
                   q13,q23,q33,q34,q35,q36,
                   q14,q24,q34,q44,q45,q46,
                   q15,q25,q35,q45,q55,q56,
                   q16,q26,q36,q46,q56,q66;
        std::cout << qmatrix << std::endl;
        abort();
        */
        //std::cout << "determinante =  " << qmatrix.determinant() << std::endl;

        // Three sides 4body force
        /*
        double det = (1+q11)*((1+q22)*(1+q33)-q23*q23) - q12*(q12*(1+q33)-q23*q13) + q13*(q12*q23-(1+q22)*q13) +
        (1+q11)*((1+q44)*(1+q55)-q45*q45) - q14*(q14*(1+q55)-q45*q15) + q15*(q14*q45-(1+q44)*q15) +
        (1+q22)*((1+q44)*(1+q66)-q46*q46) - q24*(q24*(1+q66)-q46*q26) + q26*(q24*q46-(1+q44)*q26) +
        (1+q33)*((1+q55)*(1+q66)-q56*q56) - q35*(q35*(1+q66)-q56*q36) + q36*(q35*q56-(1+q55)*q36);
        */

        Eigen::MatrixXd qmatrix (6,6);
        qmatrix << 1.+q11,q12,q13,q14,q15,q16,
                   q12,1.+q22,q23,q24,q25,q26,
                   q13,q23,1.+q33,q34,q35,q36,
                   q14,q24,q34,1.+q44,q45,q46,
                   q15,q25,q35,q45,1.+q55,q56,
                   q16,q26,q36,q46,q56,1.+q66;

        // hyperrradial 4body force
        double det = qmatrix.determinant();

			det = pow(det,-1.5);

        tmp4Pot += det;
      }//sum over the quartets
      double value = V0*scalarProduct * tmp4Pot;
      ta4ij_(indx_i,indx_new) += sign*value;
      ta4ij_(indx_new,indx_i) = ta4ij_(indx_i,indx_new);
    }//end if4body

    vaij_(indx_i,indx_new) += sign*pot;
    vaij_(indx_new,indx_i) = vaij_(indx_i,indx_new);

    // Coulomb
    qaij_(indx_i,indx_new) += sign*qpot;
    qaij_(indx_new,indx_i) = qaij_(indx_i,indx_new) ;

    indx_i++;// the next gaussian
    lpS++;
    lpI++;
  } // end of the loop over the old gaussians 
}

//---------------------------------------------------------------------------------------
void
Basis::add_casimir_(Eigen::MatrixXd &a, Eigen::VectorXd &u, Eigen::VectorXd
    &lS, Eigen::VectorXd &lI, int sign)
{
   /*
    * Generation of the Casimir on the 
    * gaussian basis 
    */
  int indx_new = k_;
    DoubleFactorial dfact;
    double scalConst = pow(2*M_PI,1.5*(np_-1)) * dfact(2*L_+1)/4./M_PI;
    // the other components
    int indx_i = 0;
    auto lpI = isospin_.begin();
    auto lpS = spin_.begin();
    for(auto aui : Au_)
    {  
      double tmp=0;
      double tmpSpace = 0;
      double tmpSpin = 0;
      double tmpIso = 0;
      double tmpSpinIso =0;
      double normaSigma = lpS->transpose()*lS;
      double normaTau = lpI->transpose()*lI;
      int pair=0;
      for(auto p : coo_.transpositionJacobi)
      { 

        Eigen::MatrixXd as = p.transpose() * a * p;
        Eigen::VectorXd us = u;
        if(L_>0)
          us = p.transpose() * u;

        Eigen::MatrixXd binv = (aui.first+as).inverse();
        double bdet = (aui.first+as).determinant();
        // definition of the numbers
        double rho=0;
        if(L_>0)
          rho = aui.second.transpose()*binv*us;

        // spin exchange
        double psigma = spinExange(*lpS,lS,pair);
        // isospin exchange
        double  ptau = isospinExange(*lpI,lI,pair);

        tmp += scalConst/pow(bdet,1.5) * pow(rho,L_)
               * psigma*ptau;
        tmpSpace += scalConst/pow(bdet,1.5) * pow(rho,L_);
        tmpSpin += psigma;
        tmpIso  += ptau;
        tmpSpinIso  += psigma*ptau;
        pair++; // I count the number of pairs/transpositions
      }
      casimir_(indx_i,indx_new) += tmp;
      casimir_(indx_new,indx_i) = casimir_(indx_i,indx_new);

      casimirSpace_(indx_i,indx_new) += tmpSpace*normaTau*normaSigma;
      casimirSpace_(indx_new,indx_i) = casimirSpace_(indx_i,indx_new);

      casimirSpin_(indx_i,indx_new) += tmpSpin*gij_(indx_i,indx_new)/normaSigma;
      casimirSpin_(indx_new,indx_i) = casimirSpin_(indx_i,indx_new);

      casimirIso_(indx_i,indx_new) += tmpIso*gij_(indx_i,indx_new)/normaTau;
      casimirIso_(indx_new,indx_i) = casimirIso_(indx_i,indx_new) ;

      casimirSpinIso_(indx_i,indx_new) += tmpSpinIso*gij_(indx_i,indx_new)/normaTau/normaSigma;
      casimirSpinIso_(indx_new,indx_i) = casimirSpinIso_(indx_i,indx_new) ;

      indx_i++;
      lpI++;
      lpS++;
    }
}

/*
 * Save all the the data in a HDF5 file
 */
int 
Basis::saveAll(std::string fileName)
{
  // we open the hdf5 file 
  hid_t       file_id, group_id, dataset_id, dataspace_id, attribute_id;  /* identifiers */
  herr_t      status;

  /* Open a file. */
  file_id = H5Fcreate (fileName.c_str(), 
      H5F_ACC_TRUNC, H5P_DEFAULT,H5P_DEFAULT);

  // Attributes to the root group
  hsize_t dim_e[1];
  dim_e[0] = 1;
  dataspace_id = H5Screate_simple(1, dim_e, NULL);
  attribute_id = H5Acreate(file_id, "Energy Units", 
      H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attribute_id,H5T_NATIVE_DOUBLE,&indata_.energyUnits);

  status = H5Aclose(attribute_id);
  status = H5Sclose(dataspace_id);

  // I want to save the matrices
  hsize_t     dims[2]; // it is a matrix, thus two dimensions
  dims[0] = k_;
  dims[1] = k_;
  dataspace_id = H5Screate_simple(2, dims, NULL);

  /*
   * Save the overlaps gij_
   */

  /* Create the dataset. */
	dataset_id = H5Dcreate (file_id, "/overlaps", H5T_NATIVE_DOUBLE,
	dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT); 

  Eigen::MatrixXd matrix = gij_.block(0,0,k_,k_);
  /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     matrix.data());
  /* Close the first dataset. */
   status = H5Dclose(dataset_id);

  /*
   * Save the kinetic energy haij_
   */

  /* Create the dataset. */
	dataset_id = H5Dcreate (file_id, "/kinetic", H5T_NATIVE_DOUBLE,
	dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT); 

  matrix = haij_.block(0,0,k_,k_);
  /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     matrix.data());
  /* Close the first dataset. */
   status = H5Dclose(dataset_id);

  /*
   * Save the two-body potential energy
   */

  /* Create the dataset. */
	dataset_id = H5Dcreate (file_id, "/twoBodyPotential", H5T_NATIVE_DOUBLE,
	dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT); 

  matrix = vaij_.block(0,0,k_,k_);
  /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     matrix.data());
  /* Close the first dataset. */
   status = H5Dclose(dataset_id);


  /*
   * Save the three-body potential energy
   */

  /* Create the dataset. */
	dataset_id = H5Dcreate (file_id, "/threeBodyPotential", H5T_NATIVE_DOUBLE,
	dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT); 

  matrix = taij_.block(0,0,k_,k_);
  /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     matrix.data());
  /* Close the first dataset. */
   status = H5Dclose(dataset_id);
  /*
   * Save the radius matrix energy raij_
   */

  /* Create the dataset. */
	dataset_id = H5Dcreate (file_id, "/radiusSquare", H5T_NATIVE_DOUBLE,
	dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT); 

  matrix = raij_.block(0,0,k_,k_);
  /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     matrix.data());
  /* Close the first dataset. */
   status = H5Dclose(dataset_id);

	/*
	 *  Save the harmonic trap energy
   */ 

  /* Create the dataset. */
	dataset_id = H5Dcreate (file_id, "/harmonicTrap", H5T_NATIVE_DOUBLE,
	dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT); 

  matrix = trapij_.block(0,0,k_,k_);
  /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     matrix.data());
  /* Close the first dataset. */
   status = H5Dclose(dataset_id);

	/*
	 *  Save the four-body potential energy
   */ 

  /* Create the dataset. */
	dataset_id = H5Dcreate (file_id, "/fourBodyPotential", H5T_NATIVE_DOUBLE,
	dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT); 

  matrix = ta4ij_.block(0,0,k_,k_);
  /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     matrix.data());
  /* Close the first dataset. */
   status = H5Dclose(dataset_id);


  /*
   * Save the three-body potential spin energy
   */

  /* Create the dataset. */
	dataset_id = H5Dcreate (file_id, "/threeBodyPotentialSpin", H5T_NATIVE_DOUBLE,
	dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT); 

  matrix = tapij_.block(0,0,k_,k_);
  /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     matrix.data());
  /* Close the first dataset. */
   status = H5Dclose(dataset_id);

  /*
   * Save the coulomb potential energy
   */

  /* Create the dataset. */
	dataset_id = H5Dcreate (file_id, "/coulombPotential", H5T_NATIVE_DOUBLE,
	dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT); 

  matrix = qaij_.block(0,0,k_,k_);
  /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     matrix.data());
  /* Close the first dataset. */
   status = H5Dclose(dataset_id);

  /*
   * Save the \phi_ij
   */

  /* Create the dataset. */
	dataset_id = H5Dcreate (file_id, "/phiij", H5T_NATIVE_DOUBLE,
	dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT); 

  matrix = phiij_.block(0,0,k_,k_);
  /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     matrix.data());
  /* Close the first dataset. */
   status = H5Dclose(dataset_id);

/*
   * Save the eigenvalues
   */

  /* Create the dataset. */
        dataset_id = H5Dcreate (file_id, "/eigs", H5T_NATIVE_DOUBLE,
        dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT);

  matrix = energy_.block(0,0,k_,1);
  /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     matrix.data());
  /* Close the first dataset. */
   status = H5Dclose(dataset_id);

  /*
   * Save the cij
   */

  /* Create the dataset. */
	dataset_id = H5Dcreate (file_id, "/cij", H5T_NATIVE_DOUBLE,
	dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT); 

  matrix = cij_.block(0,0,k_,k_);
  /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     matrix.data());
  /* Close the first dataset. */
   status = H5Dclose(dataset_id);
  
   // FINISHED WITH MATRICES  *****************************
   
  /* Close the dataspace */
  status = H5Sclose(dataspace_id);

  //*****************************************************************

  // now I've to store the alphas' values
  // the dataspace is 2 dimensional, one dimension
  // for the list, the other for the alphas'
  /*
   * Save the alphas
   */
  dims[0] = k_;
  dims[1] = coo_.pairs.size();
  dataspace_id = H5Screate_simple(2, dims, NULL);
  matrix.resize(dims[0],dims[1]);
  int i = 0;
  for(auto alpha : alpha_)
  {
    matrix.row(i) = alpha;
    i++;
  }

  /* Create the dataset. */
	dataset_id = H5Dcreate (file_id, "/alphas", H5T_NATIVE_DOUBLE,
	dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT); 

  /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     matrix.data());
  /* Close the first dataset. */
   status = H5Dclose(dataset_id);
  /* Close the dataspace */
  status = H5Sclose(dataspace_id);

   /*
    * Save the spin components
    */
  dims[0] = k_;
  dims[1] = coo_.primitiveS.size();
  dataspace_id = H5Screate_simple(2, dims, NULL);
  // Difference of row/cols major ordering between eigen and hdf5
  matrix.resize(dims[1],dims[0]);
  i = 0;
  for(auto spin  : spin_)
  {
    matrix.col(i) = spin;
    i++;
  }


  /* Create the dataset. */
	dataset_id = H5Dcreate (file_id, "/spins", H5T_NATIVE_DOUBLE,
	dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT); 

  /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     matrix.data());
  /* Close the first dataset. */
   status = H5Dclose(dataset_id);
  /* Close the dataspace */
  status = H5Sclose(dataspace_id);

   /*
    * Save the isospin components
    */
  dims[0] = k_;
  dims[1] = coo_.primitiveI.size();
  dataspace_id = H5Screate_simple(2, dims, NULL);
  // Difference of row/cols major ordering between eigen and hdf5
  matrix.resize(dims[1],dims[0]);
  i = 0;
  for(auto isospin  : isospin_)
  {
    matrix.col(i) = isospin;
    i++;
  }


  /* Create the dataset. */
	dataset_id = H5Dcreate (file_id, "/isospins", H5T_NATIVE_DOUBLE,
	dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT); 

  /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     matrix.data());
  /* Close the first dataset. */
   status = H5Dclose(dataset_id);
  /* Close the dataspace */
  status = H5Sclose(dataspace_id);

  

 /*
  * Save the u components
  */
  dims[0] = k_;
  dims[1] = Au_.front().second.size(); // the dimension depends if L=0
  dataspace_id = H5Screate_simple(2, dims, NULL);
  // Difference of row/cols major ordering between eigen and hdf5
  matrix.resize(dims[1],dims[0]);
  i = 0;
  for(auto au : Au_)
  {
    matrix.col(i) = au.second;
    i++;
  }

  /* Create the dataset. */
	dataset_id = H5Dcreate (file_id, "/u", H5T_NATIVE_DOUBLE,
	dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT); 

  /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     matrix.data());
  /* Close the first dataset. */
   status = H5Dclose(dataset_id);
  /* Close the dataspace */
  status = H5Sclose(dataspace_id);


 /*
  * Save the A matrices 
  */
  hsize_t     a_dims[3]; // it is list of  matrices, thus three dimensions
  a_dims[0] = k_;
  a_dims[1] = np_-1;  // number of Jacobi
  a_dims[2] = np_-1;
  dataspace_id = H5Screate_simple(3, a_dims, NULL);
  // Difference of row/cols major ordering between eigen and hdf5
  Eigen::VectorXd vector(a_dims[0]*a_dims[1]*a_dims[2]);
  i = 0;
  for(auto au : Au_)
  {
    vector.segment(i,au.first.size()) = Eigen::Map<Eigen::VectorXd>
      (au.first.data(),au.first.size());
    i+=au.first.size();
  }

  /* Create the dataset. */
	dataset_id = H5Dcreate (file_id, "/A", H5T_NATIVE_DOUBLE,
	dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT); 

  /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     vector.transpose().data());
  /* Close the first dataset. */
   status = H5Dclose(dataset_id);
  /* Close the dataspace */
  status = H5Sclose(dataspace_id);



  /* Close the file. */
  status = H5Fclose(file_id);



  return 0;
}

