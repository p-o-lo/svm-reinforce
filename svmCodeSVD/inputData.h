/*
 * @file inputData.h
 */
#ifndef INPUTDATA_H
#define INPUTDATA_H
#include<string>
#include<map>
#include<vector>
#include<cmath>
#include<cstdlib>
#include<string>
#include <iomanip>
#include <limits>
#include <sstream>  
using namespace std;
/**
 * Class to read the input data from a file 
 */
class InputData
{
  private:
    int testKindPotential(std::string);
    int testKindThreeBody(std::string);
    int testKindSymmetrization(std::string);
  public:
    string dataFile;
    int sizeBasis; 
    int parity;
    int angularMomentum;
    double totalMomentum;
    double totalSpin;
    double totalIsospin;
    bool ifNoLocalAlpha;
    bool ifJJCoupling; 
    bool ifCoulomb;
    bool ifThreeBody;
    bool ifThreeBodySpin;
    bool ifFourBody;
    bool ifHarmonicTrap;
    double energyUnits;
    double lengthUnits;
    double harmonicLength;
    int npart;
    std::vector<double> masses;
    std::string kindPotential;
    std::vector<std::vector<double> > twoBodyParameters;
    std::string kindThreeBody;
    std::vector<double> threeBodyParameters;
    std::vector<double> threeBodyParametersSpin;
    std::vector<double> fourBodyParameters;
    std::string kindSymmetrization;
    std::vector<std::vector<double> > alphaRanges;
    //
    std::map<std::string,int> mapPotential
    {{"GAUSSIAN",1},{"VOLKOV",2},{"MINNESOTA",3},{"REALGAUSSIAN",4},
      {"COULOMB",5},{"TWOGAUSSIAN_BARRIER",6},{"SHIFTED_GAUSSIAN",7},{"COULOMB_REG",8},{"YAMAGUCHI",9},{"ALPHA_ALPHA",10}};
    enum {GAUSSIAN=1, VOLKOV=2, MINNESOTA=3,REALGAUSSIAN=4,COULOMB=5,
          TWOGAUSSIAN_BARRIER=6,SHIFTED_GAUSSIAN=7,COULOMB_REG=8,YAMAGUCHI=9,ALPHA_ALPHA=10};
    //
    std::map<std::string,int> mapThreeBody
    {{"TWOSIDES",1},{"HYPERRADIUS",2}};
    enum {TWOSIDES=1, HYPERRADIUS=2};
    //
    std::map<std::string,int> mapSymmetrization
    {{"NONE",1},{"BOSONS",2},{"FERMIONS",3}};
    enum {NONE=1, BOSONS=2, FERMIONS=3};

    /**
     * Default Constructor
     */
     InputData(void);
    
    /**
     * Copy constructor
     */
    InputData(const InputData & inData);
    /**
     * Generic constructor from a file
     */
    InputData(std::string fileName);
    /**
     * Print the input data on a file
     */
    int print(std::string fileName);
    /**
     * Write inputdata on the output stream
     */
    friend std::ostream& operator<< (std::ostream& o, const InputData & in);
};
#endif //INPUT_DATA_H
