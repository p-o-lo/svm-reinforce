/**
 * @file coordinates.h
 */
#include"coordinates.h"
#include<iostream>


Coordinates::Coordinates(int np, std::vector<double> masses, double spin, double isospin, bool ifCoulomb)
{
  // pairs 
  for(int i=0; i< np; i++)
    for(int j=i+1; j< np; j++)
       pairs.push_back(std::make_pair(i,j));

  // triplest
  for(int i=0; i< np; i++)
    for(int j=i+1; j< np; j++)
      for(int k=j+1; k<np; k++)
      {
        int p1 = std::distance(pairs.begin(), std::find(pairs.begin(), pairs.end(), std::make_pair(i,j)));
        int p2 = std::distance(pairs.begin(), std::find(pairs.begin(), pairs.end(), std::make_pair(i,k)));
  
        int p3 = std::distance(pairs.begin(), std::find(pairs.begin(), pairs.end(), std::make_pair(j,k)));

        triplets.push_back(std::make_tuple(p1,p2,p3));
      }

  // quadruplets
  for(int i=0; i< np; i++)
    for(int j=i+1; j< np; j++)
      for(int k=j+1; k< np; k++)
        for(int l=k+1; l< np; l++)
        {
          int p1 = std::distance(pairs.begin(), std::find(pairs.begin(), pairs.end(), std::make_pair(i,j)));
          int p2 = std::distance(pairs.begin(), std::find(pairs.begin(), pairs.end(), std::make_pair(i,k)));
          int p3 = std::distance(pairs.begin(), std::find(pairs.begin(), pairs.end(), std::make_pair(i,l)));
          int p4 = std::distance(pairs.begin(), std::find(pairs.begin(), pairs.end(), std::make_pair(j,k)));
          int p5 = std::distance(pairs.begin(), std::find(pairs.begin(), pairs.end(), std::make_pair(j,l)));
          int p6 = std::distance(pairs.begin(), std::find(pairs.begin(), pairs.end(), std::make_pair(k,l)));

          quadruplets.push_back(std::make_tuple(p1,p2,p3,p4,p5,p6));
        }


  // permutations 
  Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(np);
  perm.setIdentity();
  do{
      //std::cout << perm.toDenseMatrix() << std::endl;
      permutations.push_back(perm);
      signPermutations.push_back(perm.determinant());
  } while (std::next_permutation(perm.indices().data(),
               perm.indices().data()+perm.indices().size()));

  // transpositions 
  for(auto i=pairs.begin(); i<pairs.end(); i++)
  {
    perm.setIdentity();
    transpositions.push_back(perm.applyTranspositionOnTheRight(i->first,i->second));
  }


  // intemediate masses
  std::vector<double> intM; 
  double m=0;
  for(size_t i=0; i<masses.size(); i++)
  {
    m += masses[i];
    intM.push_back(m);
  }
  // Setup of the U matrix eq. (2.5) Suzuki&Varga 
  U.resize(np,np);
  U = Eigen::MatrixXd::Zero(np,np);
  for(size_t i=0; i<(size_t) np; i++)
    for(size_t j=0; j<=i; j++)
      U(i,j) = masses[j]/intM[i];
  for(size_t i=0; i<(size_t) np-1; i++)
    U(i,i+1) = -1.0;


  // Setup of the inverse U matrix eq. (2.6) Suzuki&Varga 
  UINV.resize(np,np);
  UINV = Eigen::MatrixXd::Zero(np,np);
  for(size_t i=0; i<(size_t) np-1; i++)
    for(size_t j=i; j<(size_t)np-1; j++)
      UINV(i,j) = masses[j+1]/intM[j+1];

  for(size_t i=0; i<(size_t) np; i++)
    UINV(i,np-1) = 1;

  for(size_t i=1; i<(size_t) np; i++)
    UINV(i,i-1) = - intM[i-1]/intM[i];


  // Setup of the matrix Omega to use in the case
  // of Harmonic Trap
  Omega.resize(np-1,np-1);
  Omega = Eigen::MatrixXd::Zero(np-1,np-1);
  for(size_t i=0; i<(size_t) np-1; i++)
    for(size_t j=0; j<(size_t) np-1; j++)
    {
      double tmp = 0;
      for(size_t k=0; k<(size_t) np; k++)
        tmp += masses[k]*UINV(k,i)*UINV(k,j);
      Omega(i,j) = tmp;
    }

    
  
  Eigen::MatrixXd p(np,np);
  for(auto i : transpositions)
  {
     p = U * i *UINV; 
     //p = Eigen::MatrixXd::Identity(np-1,np-1);
     transpositionJacobi.push_back(p.block(0,0,np-1,np-1));
  }
  for(auto i : permutations)
  {
     p = U * i *UINV; 
     //p = Eigen::MatrixXd::Identity(np-1,np-1);
     permutationsJacobi.push_back(p.block(0,0,np-1,np-1));
  }

  
  // calculation of the transpositions on the Jacobi

  // Setup of the W
  size_t nc = np*(np-1)/2;
  W.resize(nc, np-1);
  for(size_t i=0; i<nc; i++)
    for(size_t j=0; j<(size_t) np-1; j++)
      W(i,j) = UINV(pairs[i].first,j) - UINV(pairs[i].second,j);
  // Setup of Z
  Z.resize(nc, np-1);
  for(size_t i=0; i<nc; i++)
    for(size_t j=0; j<(size_t) np-1; j++)
      Z(i,j) = U(j,pairs[i].first) - U(j,pairs[i].second);

  // Setup of the matrix Radius2 to use in the 
  // calculation of the square radius
  Radius2.resize(np-1,np-1);
  Radius2= Eigen::MatrixXd::Zero(np-1,np-1);
  for(size_t i=0; i<(size_t) np-1; i++)
    for(size_t j=0; j<(size_t) np-1; j++)
    {
      double tmp = 0;
  //    for(size_t p=0; p<(size_t) nc; p++)
  //      tmp += W(p,i)*W(p,j);
      for(size_t k=0; k<(size_t) np; k++)
        tmp += UINV(k,i)*UINV(k,j);
      Radius2(i,j) = tmp/np;
    }

  // Setup of the Lambda
  Lambda.resize(np-1,np-1);
  Lambda *= 0;
  for(size_t i=0; i<(size_t) np-1; i++)
    for(size_t j=0; j<(size_t)np-1; j++)
    {
      double sum = 0;
      for(size_t k=0; k<(size_t) np; k++)
        sum += U(i,k)*U(j,k)/masses[k];
      Lambda(i,j) = sum;
    }


  /**
   * Our choice of Jacobi coordinates
   * but indeces start from zero
   */
  V.resize(np,np);
  V = Eigen::MatrixXd::Zero(np,np);
  // center of mass 
  for(size_t j=0; j<(size_t) np; j++)
    V(np-1,j) = masses[j]/intM[np-1];
  // and the rest
  for(size_t j=0; j<(size_t) np-1; j++)
  {
    size_t i = np-2-j;
    double factor = sqrt(2*masses[j+1]*intM[j]/(masses[j+1]+intM[j]));
    V(i,j+1) = factor;
    for(int k=j ; k>=0; --k)
      V(i,k) = - factor/intM[j] * masses[k] ;
  }
  VINV.resize(np,np);
  VINV = V.inverse();
  
  // Setup of the WV
  WV.resize(nc, np-1);
  for(size_t i=0; i<nc; i++)
    for(size_t j=0; j<(size_t) np-1; j++)
      WV(i,j) = VINV(pairs[i].first,j) - VINV(pairs[i].second,j);

  // Setup of the kinetic energy
  // It must be KIN=2*I
  KIN.resize(np-1,np-1);
  KIN = Eigen::MatrixXd::Zero(np-1,np-1);
  for(size_t i=0; i<(size_t) np-1; i++)
    for(size_t j=0; j<(size_t)np-1; j++)
    {
      double sum = 0;
      for(size_t k=0; k<(size_t) np; k++)
        sum += V(i,k)*V(j,k)/masses[k];
      KIN(i,j) = sum;
    }

  //Definitions of zita vector for non-local potential  from (C.27) and (C.25)
  //of Few Body System (2008) 42: 33-72 Y. Suzuki, W. Horiuchi
  Eigen::VectorXd zita_i(np-1);
  Eigen::VectorXd bParam(np-1);
  bParam =  Eigen::VectorXd::Unit(np-1,0);                
  for(int  i = 0; i < pairs.size(); i++) {
	  Eigen::MatrixXd firstRow(1,np-1);
	  firstRow = W.row(i);
	  Eigen::MatrixXd constTerm(1,np-1);
	  constTerm = masses[pairs[i].first]*UINV.row(pairs[i].first)/(masses[pairs[i].first] + masses[pairs[i].second]) + 
		  masses[pairs[i].second]*UINV.row(pairs[i].second)/(masses[pairs[i].first] + masses[pairs[i].second]);
	  Eigen::MatrixXd tmp(np-1,np-1);
	  tmp.row(0) << firstRow;
	  int k = 1;
	  for (int j = 0; j <= np-1; j++) {
		  if (j != pairs[i].first && j != pairs[i].second) {
			  tmp.row(np-1-k) = UINV.row(j) - constTerm;
			  k++;
	  	  }
	  }
          Tk.push_back(tmp);
	  zita_i = tmp.colPivHouseholderQr().solve(bParam);
	  zita.push_back(zita_i);

  }  


  /*
   * Spin and Isospin basis
   */
  if(spin>=0)
  {
    // creation of the primitive basis up=1 down=2
    std::vector<int> initial;
    int mu = spin+np/2.;
    // check if the spin is compatible with the number of particles
    if(fabs(mu-spin-np/2.) > 1e-6 || spin > np/2.0)
    {
      std::cerr << "Problem with the value of the spin" << std::endl;
      std::cerr << mu <<  "   "  << spin << "  " << np/2. << std::endl;
      abort();
    }
    for(int i=0; i<mu; i++)
      initial.push_back(1);
    for(int i=mu; i<np; i++)
      initial.push_back(2);
    do {
      primitiveS.push_back(initial);
    } while (next_permutation(initial.begin(), initial.end()));
    
    // standard ordering of the primitive basis
    std::sort(primitiveS.begin(),primitiveS.end(),test_);

    // calculate the transpositions and projectors
    Eigen::MatrixXd uno = Eigen::MatrixXd::Identity(primitiveS.size(),primitiveS.size()); 
    for(auto p : pairs)
    {
      Eigen::MatrixXd t = Eigen::MatrixXd::Zero(primitiveS.size(),primitiveS.size()); 
      int column=0;
      for(auto v : primitiveS)
      {
        std::swap(v[p.first],v[p.second]);
        auto pos = std::distance(primitiveS.begin(), std::find(primitiveS.begin(), primitiveS.end(), v));
        t(pos,column) = 1;
        column++;
      }
      projectorSpin.push_back(std::make_pair( 0.5*(uno-t), 0.5*(uno+t)));
      traspositionSpin.push_back(t);
    }

    // calculate the permutations for the spin
    for(auto p : permutations)
    {
      Eigen::MatrixXd t = Eigen::MatrixXd::Zero(primitiveS.size(),primitiveS.size()); 
      int column=0;
      for(auto v : primitiveS)
      {
        Eigen::Map<Eigen::VectorXi> r (&v[0],v.size());
        Eigen::VectorXi pr = p*r;
        std::vector<int> vr(pr.data(), pr.data()+pr.size());
        auto pos = std::distance(primitiveS.begin(), 
            std::find(primitiveS.begin(), primitiveS.end(), vr));
        t(pos,column) = 1;
        column++;
      }
      permutationsSpin.push_back(t);
    }

    // calculation of the irreps
    for (auto v : primitiveS)
    {
      bool keep = true;
      double partial=0;
      for(auto e : v)
      {
        partial += pow(-1,e+1);
        if(partial<0)
          keep = false;
      }
      if(keep)
        irrepsS.push_back(v);
    }

    // calculation of the irreps coordinates on the primitive
    lS = Eigen::MatrixXd::Zero(irrepsS.size(),primitiveS.size());
    int i=0;
    for(auto ir : irrepsS)
    {
      int j=0; 
      for(auto pr : primitiveS)
      {
        double prod = 1;
        double pb = 0;
        double pm = 0;
        for(int k=0; k<np; k++)
        {
          pb += ir[k];
          pm += pr[k];
          double sb = 3/2.*(k+1) - pb;
          double sm = 3/2.*(k+1) - pm;
          // DEBUG
          /*
          if(i==7 && j==0)
            std::cout << ir[k] << " " << pr[k] << " " << sb << " " << sm << "- "<< C_(ir[k],pr[k],sb,sm) <<std::endl;
          */
          prod *= C_(ir[k],pr[k],sb,sm);
          if(prod==0) // I'm going out of the shadow....
            break;
        }
        lS(i,j) = prod;
        j++; // next column/primitive
      }
      i++; // next row/irreps
    }

  }// end if spin

  if(isospin>=0)
  {
    // creation of the primitive basis up=1 down=2
    std::vector<int> initial;
    int mu = isospin+np/2.;
    // check if the isospin is compatible with the number of particles
    if(fabs(mu-isospin-np/2.) > 1e-6 || isospin > np/2.)
    {
      std::cerr << "Problem with the value of the isospin" << std::endl;
      std::cout << mu << "  "  << isospin << "  "  << np/2. << std::endl;
      abort();
    }
    for(int i=0; i<mu; i++)
      initial.push_back(1);
    for(int i=mu; i<np; i++)
      initial.push_back(2);
    do {
      primitiveI.push_back(initial);
    } while (next_permutation(initial.begin(), initial.end()));
    
    // standard ordering of the primitive basis
    std::sort(primitiveI.begin(),primitiveI.end(),test_);
    Eigen::MatrixXd uno = Eigen::MatrixXd::Identity(primitiveI.size(),primitiveI.size()); 

    // calculate the transpositions
    for(auto p : pairs)
    {
      Eigen::MatrixXd t = Eigen::MatrixXd::Zero(primitiveI.size(),primitiveI.size()); 
      int column=0;
      for(auto v : primitiveI)
      {
        std::swap(v[p.first],v[p.second]);
        auto pos = std::distance(primitiveI.begin(), std::find(primitiveI.begin(), primitiveI.end(), v));
        t(pos,column) = 1;
        column++;
      }
      projectorIsospin.push_back(std::make_pair( 0.5*(uno-t), 0.5*(uno+t)));
      traspositionIsospin.push_back(t);
    }
    
    // calculate the permutations for the isospin
    for(auto p : permutations)
    {
      Eigen::MatrixXd t = Eigen::MatrixXd::Zero(primitiveI.size(),primitiveI.size()); 
      int column=0;
      for(auto v : primitiveI)
      {
        Eigen::Map<Eigen::VectorXi> r (&v[0],v.size());
        Eigen::VectorXi pr = p*r;
        std::vector<int> vr(pr.data(), pr.data()+pr.size());
        auto pos = std::distance(primitiveI.begin(), 
            std::find(primitiveI.begin(), primitiveI.end(), vr));
        t(pos,column) = 1;
        column++;
      }
      permutationsIsospin.push_back(t);
    }

    /*
     * calculation of the irreps
     */
    // in case of the coulomb I've all the irreps compatible with Tz
    if(ifCoulomb)
    {
      // create the primitive with T>Tz (the isospin input)
      for(double tmpiso=np/2.0; tmpiso>isospin; tmpiso-=1.0) 
      {
        // creation of the primitive basis up=1 down=2
        std::vector<std::vector<int> > tmpPrimitive;
        std::vector<int> initial;
        int mu = tmpiso+np/2.;
        // check if the isospin is compatible with the number of particles
        if(fabs(mu-tmpiso-np/2.) > 1e-6 || tmpiso > np/2.)
        {
          std::cerr << "Problem with the value of the isospin" << std::endl;
          std::cout << mu << "  "  << isospin << "  "  << np/2. << std::endl;
          abort();
        }
        for(int i=0; i<mu; i++)
          initial.push_back(1);
        for(int i=mu; i<np; i++)
          initial.push_back(2);
        do {
          tmpPrimitive.push_back(initial);
        } while (next_permutation(initial.begin(), initial.end()));
        
        // standard ordering of the primitive basis
        std::sort(tmpPrimitive.begin(),tmpPrimitive.end(),test_);
        // now I can sort out the irreps
        std::vector<std::vector<int> > tmpIrrepsI;
        for (auto v : tmpPrimitive)
        {
          bool keep = true;
          double partial=0;
          for(auto e : v)
          {
            partial += pow(-1,e+1);
            if(partial<0)
              keep = false;
          }
          if(keep)
            tmpIrrepsI.push_back(v);
        }
        irrepsI.push_back(tmpIrrepsI);
      } // end of the cycle over irreps
    } // end ifcoulomb with the calculation of the other irreps

    std::vector<std::vector<int> > tmpIrrepsI;
    for (auto v : primitiveI)
    {
      bool keep = true;
      double partial=0;
      for(auto e : v)
      {
        partial += pow(-1,e+1);
        if(partial<0)
          keep = false;
      }
      if(keep)
        tmpIrrepsI.push_back(v);
    }
    irrepsI.push_back(tmpIrrepsI);
    
    // calculation of the irreps coordinates on the primitive
    for( auto irrep : irrepsI)
    {
      Eigen::MatrixXd tmpLI = Eigen::MatrixXd::Zero(irrep.size(),primitiveI.size());
      int i=0;
      for(auto ir : irrep)
      {
        int j=0; 
        for(auto pr : primitiveI)
        {
          double prod = 1;
          double pb = 0;
          double pm = 0;
          for(int k=0; k<np; k++)
          {
            pb += ir[k];
            pm += pr[k];
            double sb = 3/2.*(k+1) - pb;
            double sm = 3/2.*(k+1) - pm;
            prod *= C_(ir[k],pr[k],sb,sm);
            if(prod==0) // I'm going out of the shadow....
              break;
          }
          tmpLI(i,j) = prod;
          j++; // next column/primitive
        }
        i++; // next row/irreps
      }
      lI.push_back(tmpLI);
    }
 

    // calculation of the charge operator
    for(auto p : pairs)
    {
      Eigen::MatrixXd q = Eigen::MatrixXd::Zero(primitiveI.size(),primitiveI.size());
      int i=0;
      for(auto pr : primitiveI)
      {
        double v = pr[p.first] * pr[p.second];
        //if(v==4) // this is if protons are = 2 in primitive
        if(v==1)
          q(i,i) = 1.;
        i++;
      }
      chargeOperator.push_back(q);
     }
  }// end if isospin 



}

bool
Coordinates::test_(std::vector<int> v1, std::vector<int> v2)
{
  int e1;
  int e2;

  do
  {
    e1 = v1.back();
    e2 = v2.back();
    v1.pop_back();
    v2.pop_back();
  } while (e1==e2);

  return (e1>e2);
}


double 
Coordinates::C_(int ir, int pr, double s, double m)
{
  double value=0;
  switch(ir)
  {
    case 1 :
      switch(pr) 
      {
        case 1:
          value = sqrt((s+m)/2./s);
        break;
        case 2:
          value = sqrt( (s-m)/2./s );
        break;
      }
    break;

    case 2:
      switch(pr)
      {
        case 1: 
          value = - sqrt( (s-m+1)/(2.*s+2.) );
        break;

        case 2:
          value = sqrt( (s+m+1)/(2.*s+2.) );
        break;
      }
    break;
  }
  return value;
}
