#include<iostream>
#include <sys/stat.h>
#include"inputData.h"
#include"coordinates.h"
#include"basis.h"
#include"print_tuple.h"

int
main(int argc, char ** argv)
{
  if(argc!=2)
  {
    std::cerr << "Usage:\n" << argv[0]<< " <input_file> " << std::endl;
    exit(-1);
  }

  std::string inputBase = argv[1];
  std::string inputFile = inputBase + ".input";
  InputData inData(inputFile) ;

  std::string baseName = inData.dataFile;
  std::string inputName = baseName + ".input";
  struct stat buffer;
  int i=0;
  while(stat(inputName.c_str(), &buffer) == 0)
  {
    baseName = inData.dataFile + "_" + std::to_string(i);
    inputName = baseName + ".input";
    i++;
  } 

  Coordinates cc(inData.npart,inData.masses,inData.totalSpin,inData.totalIsospin,inData.ifCoulomb);
  Basis base(inData,cc);

  base.loadAll(inputBase + ".h5" );

  Eigen::MatrixXd g = base.getGIJ()->block(0,0,base.getDimension(),base.getDimension());
  Eigen::MatrixXd t = base.getHAIJ()->block(0,0,base.getDimension(),base.getDimension());
  Eigen::MatrixXd v = base.getVAIJ()->block(0,0,base.getDimension(),base.getDimension());
  Eigen::MatrixXd v3 = base.getTAIJ()->block(0,0,base.getDimension(),base.getDimension());
  Eigen::MatrixXd vc = base.getQAIJ()->block(0,0,base.getDimension(),base.getDimension());
  Eigen::MatrixXd v4 = base.getTA4IJ()->block(0,0,base.getDimension(),base.getDimension());
  Eigen::MatrixXd r2 = base.getRAIJ()->block(0,0,base.getDimension(),base.getDimension());
  ofstream file("risonanza.dat");
  
  double lmax = 3;
  double lmin = 0;
  double mumax = lmax/2; 
  double mumin = lmin/2;

  double step = 0.01; 
  int n = ceil((mumax-mumin)/step)+1;
  std::cout << "numero di giri " << n << std::endl;


  int maxLambda = 13;
  int statoInR = 1;

  for(int j=0;j<n;j++)
  {
    double mu = step*j;
    Eigen::MatrixXd h = t+v + mu*v3;
    Eigen::GeneralizedSelfAdjointEigenSolver <Eigen::MatrixXd>
            es(h,g);
    Eigen::MatrixXd r = es.eigenvectors().transpose()*r2*es.eigenvectors() ;


    Eigen::SelfAdjointEigenSolver <Eigen::MatrixXd> er(r.block(0,0,maxLambda,maxLambda));
    double soglia=0.05;
    int quanti=0;
    bool non_trovato= true ;
    int starting=0;
    std::cout << 2*mu << "  " ; 
    for(int l=0; l<maxLambda; l++)
    {
      double value =er.eigenvectors().col(statoInR)[l]*er.eigenvectors().col(statoInR)[l]; 
      if(non_trovato)
        starting=l;
      if(value>soglia && l>0)
      { 
        non_trovato = false;
        quanti++; 
      }
      std::cout << value << "  " ;
    }
    Eigen::SelfAdjointEigenSolver <Eigen::MatrixXd> newEr(r.block(starting,starting,quanti,quanti));
    std::cout << "  " << starting << "  " << quanti <<"  ";
    for(int i=0; i<quanti; i++)
      std::cout << er.eigenvectors().col(statoInR)[starting+i]*er.eigenvectors().col(statoInR)[starting+i] << "   " << newEr.eigenvectors().col(0)[i]*newEr.eigenvectors().col(0)[i] << "   ";
    std::cout << std::endl;

    double energy=0;
    for(int i=0; i<quanti; i++)
      energy += newEr.eigenvectors().col(0)[i]*newEr.eigenvectors().col(0)[i]*es.eigenvalues()[starting+i];
    energy *= inData.energyUnits ;


    /* DEBUG
    std::cout << 2*mu << "  " ; 
    for(int l=0; l<maxLambda; l++)
      std::cout << es.eigenvalues()[l+1] << "  " ;
    std::cout << std::endl;
    */

    /*
    double energy=0;
    for(int l=0; l<maxLambda; l++)
      energy += er.eigenvectors().col(1)[l]*er.eigenvectors().col(1)[l]*es.eigenvalues()[l];
    energy *= inData.energyUnits ;
    */

    file.precision(8);

    file << 2*mu << "  " ;
    for(int i=0; i<maxLambda+1; i++)
    {
        file << es.eigenvalues()[i]*inData.energyUnits << " ";
        file << sqrt(r(i,i))*inData.lengthUnits << " ";
    }
    file << energy ;
    file << std::endl;
  }

  file.close();

  

  return 0;
}
