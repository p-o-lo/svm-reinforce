#include<iostream>
#include <sys/stat.h>
#include"inputData.h"
#include"coordinates.h"
#include"basis.h"
#include"print_tuple.h"
#include <vector>
#include <fstream>
#include <Eigen/Dense>

using namespace Eigen;

template<typename M>
M load_csv (const std::string & path) {
    std::fstream indata;
    //indata.open(path);
    indata.open(path.c_str(), std::ios::in);
    if(!indata) {
        std::cerr << "Oops, some problem reading the input file " << path ;
        abort();
    }
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

int main(int argc, char ** argv)
{
  if(argc!=3)
  {
    std::cerr << "Usage:\n" << argv[0]<< " <input_file> <alpha file>" << std::endl;
    exit(-1);
  }

  std::string inputFile = argv[1];
  InputData inData(inputFile) ;


  //std::string baseName = inData.dataFile;

  std::string baseName = "test";
  
  std::string inputName = baseName + ".input";
  struct stat buffer;
  int i=0;
  //while(stat(inputName.c_str(), &buffer) == 0)
  //{
  //  baseName = inData.dataFile + "_" + std::to_string(i);
  //  inputName = baseName + ".input";
    //i++;
  //} 

  ofstream file(inputName.c_str()); 
  file << inData << std::endl;
  file.close();

  //file.open(baseName+".output");
  //std::cout << "Writing on " << baseName+ ".output" << std::endl;

    /* Open alpha file */
  std::string alphaFile = argv[2];
  MatrixXd alphas = load_csv<MatrixXd>(alphaFile);
 
  /*
   * Set up of the matrices 
   * to move from Jacobi to/from cartesian
   */

  Coordinates cc(inData.npart,inData.masses,inData.totalSpin,inData.totalIsospin,inData.ifCoulomb);

  Basis base(inData,cc);
  int l = 0;
  for(; l < alphas.rows(); l++) {
      base.setState(l);
      auto output = base.testElementAlpha(alphas.row(l));
      base.newElement(std::get<1>(output),std::get<2>(output),std::get<3>(output),std::get<4>(output));  
      

    }


    Eigen::MatrixXd e = *base.getEnergy();
    double energy = inData.energyUnits * e(0,0);

    std::cout << energy << std::endl;
    std::cout << base.getPrincipalDimension() << std::endl;
    std::cout << base.getDimension() << std::endl;

//   base.saveAll(baseName + ".h5");
   return 0;

}


