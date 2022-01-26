/**
 * @file inputData.cc
 */
#include"inputData.h"
#include<fstream>
#include<iostream>

// default constructor
InputData::InputData(void) :
  dataFile("dataFile"),
  sizeBasis(200),
  parity(1),
  angularMomentum(0),
  totalMomentum(0),
  totalSpin(-1),
  totalIsospin(-1),
  ifNoLocalAlpha(false),
  ifJJCoupling(false),
  ifCoulomb(false),
  ifThreeBody(false),
  ifThreeBodySpin(false),
  ifFourBody(false),
  ifHarmonicTrap(false),
  energyUnits(.43281307),
  lengthUnits(10),
  harmonicLength(1),
  npart(3),
  masses(vector<double> {1,1,1}),
  kindPotential{"GAUSSIAN"},
  // V0 r0
  twoBodyParameters{{-2.85193929102002,1}},
  kindThreeBody{"TWOSIDES"},
  threeBodyParameters{{-10,1}},
  threeBodyParametersSpin{{-10,1}},
  fourBodyParameters{{-10,1}},
  kindSymmetrization{"NONE"},
  alphaRanges{{0,5}}
{
}

// copy constructor
InputData::InputData(const InputData & in) :
  dataFile(in.dataFile), sizeBasis(in.sizeBasis),
  parity(in.parity), angularMomentum(in.angularMomentum), totalMomentum(in.totalMomentum),
  totalSpin(in.totalSpin),totalIsospin(in.totalIsospin),ifNoLocalAlpha(in.ifNoLocalAlpha),
  ifJJCoupling(in.ifJJCoupling),ifCoulomb(in.ifCoulomb),ifThreeBody(in.ifThreeBody),ifThreeBodySpin(in.ifThreeBodySpin),
  ifFourBody(in.ifFourBody),ifHarmonicTrap(in.ifHarmonicTrap),energyUnits(in.energyUnits),lengthUnits(in.lengthUnits),
  harmonicLength(in.harmonicLength),npart(in.npart),masses(in.masses),kindPotential(in.kindPotential),
  twoBodyParameters(in.twoBodyParameters),kindThreeBody(in.kindThreeBody), threeBodyParameters(in.threeBodyParameters),
  threeBodyParametersSpin(in.threeBodyParametersSpin), fourBodyParameters(in.fourBodyParameters),kindSymmetrization(in.kindSymmetrization),
  alphaRanges(in.alphaRanges)
{}

/******************************************************************************
 *                        ---------------------                               *
 ******************************************************************************/
// generic constructor from a file
InputData::InputData(std::string fileName)
{
  // these variables can be not presents
  // we give them negative values
  parity = 1;
  totalMomentum = 0;
  totalSpin = -1;
  totalIsospin = -1;
  ifNoLocalAlpha = false;
  ifJJCoupling = false;
  ifCoulomb = false;
  ifThreeBody = false;
  ifThreeBodySpin = false;
  ifFourBody = false;
  ifHarmonicTrap = false;
  kindSymmetrization="NONE";


  std::fstream inFile;
  inFile.open(fileName.c_str(), std::ios::in);
  if(!inFile)
  {
    std::cerr << "Oops, some problem reading the input file " << fileName ;
    abort();
  }
  std::string row;
  double doubleValue;
  int intValue;
  string stringValue;
  bool boolValue;
  int line = 0;
  std::vector<double> tmpVector;
  

  while(inFile >> row)
  {
    line++;
    if(row=="npart")
    {
      inFile >> intValue;
      getline(inFile,row);
      npart = intValue;
    } else if(row=="parity")
    {
      inFile >> intValue;
      getline(inFile,row);
      parity = intValue;
    } else if(row=="totalSpin")
    {
      inFile >> doubleValue;
      getline(inFile,row);
      totalSpin = doubleValue;
    } else if(row=="totalIsospin")
    {
      inFile >> doubleValue;
      getline(inFile,row);
      totalIsospin = doubleValue;
    } else if(row=="ifNoLocalAlpha")
   {
      inFile >> boolValue;
      getline(inFile,row);
      ifNoLocalAlpha = boolValue;
    } else if(row=="ifJJCoupling")
    {
      inFile >> boolValue;
      getline(inFile,row);
      ifJJCoupling = boolValue;
    } else if(row=="ifCoulomb")
    {
      inFile >> boolValue;
      getline(inFile,row);
      ifCoulomb = boolValue;
    } else if(row=="ifThreeBodySpin")
    {
      inFile >> boolValue;
      getline(inFile,row);
      ifThreeBodySpin = boolValue;
    } else if(row=="ifThreeBody")
    {
      inFile >> boolValue;
      getline(inFile,row);
      ifThreeBody = boolValue;
    } else if(row=="ifHarmonicTrap")
    {
      inFile >> boolValue;
      getline(inFile,row);
      ifHarmonicTrap = boolValue;
    } else if(row=="ifFourBody")
    {
      inFile >> boolValue;
      getline(inFile,row);
      ifFourBody = boolValue;
    } else if(row=="energyUnits")
    {
      inFile >> doubleValue;
      getline(inFile,row);
      energyUnits = doubleValue;
    } else if(row=="harmonicLength")
    {
      inFile >> doubleValue;
      getline(inFile,row);
      harmonicLength = doubleValue;
    } else if(row=="kindPotential")
    {
      inFile >> stringValue;
      getline(inFile,row);
      kindPotential = stringValue;
      testKindPotential(kindPotential);
    } else if(row=="kindSymmetrization")
    {
      inFile >> stringValue;
      getline(inFile,row);
      kindSymmetrization = stringValue;
      testKindSymmetrization(kindSymmetrization);
    } else if(row=="kindThreeBody")
    {
      inFile >> stringValue;
      getline(inFile,row);
      kindThreeBody = stringValue;
      testKindThreeBody(kindThreeBody);
    } else if(row=="alphaRanges")
    {
      tmpVector.clear();
      getline(inFile,row);
      std::istringstream iss(row);
      while(iss >> doubleValue)
        tmpVector.push_back(doubleValue);
      alphaRanges.push_back(tmpVector);
    } else if(row=="twoBodyParameters")
    {
      tmpVector.clear();
      getline(inFile,row);
      std::istringstream iss(row);
      while(iss >> doubleValue)
        tmpVector.push_back(doubleValue);
      twoBodyParameters.push_back(tmpVector);
    } else if(row=="lengthUnits")
    {
      inFile >> doubleValue;
      getline(inFile,row);
      lengthUnits = doubleValue;
    } else if(row=="totalMomentum")
    {
      inFile >> doubleValue;
      getline(inFile,row);
      totalMomentum = doubleValue;
    } else if(row=="angularMomentum")
    {
      inFile >> intValue;
      getline(inFile,row);
      angularMomentum = intValue;
    } else if(row=="sizeBasis")
    {
      inFile >> intValue;
      getline(inFile,row);
      sizeBasis = intValue;
    } else if(row=="masses")
    {
      tmpVector.clear();
      getline(inFile,row);
      std::istringstream iss(row);
      while(iss >> doubleValue)
        masses.push_back(doubleValue);
    } else if(row=="threeBodyParametersSpin")
    {
      tmpVector.clear();
      getline(inFile,row);
      std::istringstream iss(row);
      while(iss >> doubleValue)
        threeBodyParametersSpin.push_back(doubleValue);
    } else if(row=="threeBodyParameters")
    {
      tmpVector.clear();
      getline(inFile,row);
      std::istringstream iss(row);
      while(iss >> doubleValue)
        threeBodyParameters.push_back(doubleValue);
    } else if(row=="fourBodyParameters")
    {
      tmpVector.clear();
      getline(inFile,row);
      std::istringstream iss(row);
      while(iss >> doubleValue)
        fourBodyParameters.push_back(doubleValue);
    } else if(row=="dataFile")
    {
      inFile >> stringValue;
      getline(inFile,row);
      dataFile = stringValue;
    }  else if(row=="#")
    {
      getline(inFile,row);
    } else if(row=="%")
    {
      break;
    } else 
      {
	std::cerr << "error in reading line " << line << " of input file " 
		  << fileName  << std::endl;
    abort();
      }
  } // the file has been read
  /*
   * here we have some controls. If the number of masses is less
   * than the number of particles, we assume that all the remaining 
   * particles have the mass of the last one
   */
  if(masses.size() > (size_t)npart)
  {
    std::cout << "The number of masses = " 
      << masses.size() << " is bigger of the number of particles = " << npart ;
    std::cout << std::endl;
    abort();
  } 
  if(masses.size() < (size_t)npart)
  {
    double lastmass = masses.back();
    for(int i=masses.size(); i<npart; i++)
      masses.push_back(lastmass);
  }
  /*
   * if the number of parameter is less than the number of pairs
   * we assume that the remaining pairs have the same value of 
   * the parameters
   */
  int nofpairs = npart*(npart-1)/2;
  if(twoBodyParameters.size() > (size_t)nofpairs)
  {
    std::cout << "The number of two-body potential parameters = " 
      << twoBodyParameters.size() << " is bigger of the number of pairs = " << nofpairs;
    std::cout << std::endl;
    abort();
  } 
  if(twoBodyParameters.size() < (size_t)nofpairs)
  {
    std::vector<double> last = twoBodyParameters.back();
    for(int i=twoBodyParameters.size(); i<nofpairs; i++)
      twoBodyParameters.push_back(last);
  }
  // the same as above for the ranges
  if(alphaRanges.size() > (size_t)nofpairs)
  {
    std::cout << "The number of ranges for the alphas = " 
      << alphaRanges.size() << " is bigger of the number of pairs = " << nofpairs;
    std::cout << std::endl;
    abort();
  } 
  if(alphaRanges.size() < (size_t)nofpairs)
  {
    std::vector<double> last = alphaRanges.back();
    for(int i=alphaRanges.size(); i<nofpairs; i++)
      alphaRanges.push_back(last);
  }
  if(!ifJJCoupling)
    parity = pow(-1,angularMomentum);


}
/******************************************************************************
 *                        ---------------------                               *
 ******************************************************************************/
// write data on a file
int
InputData::print(std::string fileName)
{
  std::fstream outFile;
  outFile.open(fileName.c_str(), std::ios::out);
  if(!outFile)
  {
    std::cerr << "Oops, some problem reading the input file " << fileName ;
    abort();
  }
  outFile << std::setprecision(std::numeric_limits<long double>::digits10);
  outFile << "# ------------------------------------------------------- " << std::endl;
  outFile << "# Input data for SVM        " << std::endl;
  outFile << "# ------------------------------------------------------- " << std::endl;
  outFile << "# Output files   "                                << std::endl;
  outFile << "dataFile             " << dataFile                << std::endl;
  outFile << "sizeBasis            " << sizeBasis               << std::endl;
  outFile << "parity               " << parity                  << std::endl;
  outFile << "angularMomentum      " << angularMomentum         << std::endl;
  outFile << "totalMomentum        " << totalMomentum           << std::endl;
  outFile << "totalSpin            " << totalSpin               << std::endl;
  outFile << "totalIsospin         " << totalIsospin            << std::endl;
  outFile << "ifNoLocalAlpha	   " << ifNoLocalAlpha		<< std::endl;
  outFile << "ifJJCoupling         " << ifJJCoupling            << std::endl;
  outFile << "ifCoulomb            " << ifCoulomb               << std::endl;
  outFile << "ifThreeBody          " << ifThreeBody             << std::endl;
  outFile << "ifThreeBodySpin      " << ifThreeBodySpin         << std::endl;
  outFile << "ifFourBody           " << ifFourBody              << std::endl;
  outFile << "ifHarmonicTrap       " << ifHarmonicTrap          << std::endl;
  outFile << "energyUnits          " << energyUnits             << std::endl;
  outFile << "lengthUnits          " << lengthUnits             << std::endl;
  outFile << "harmonicLength    " << harmonicLength       << std::endl;
  outFile << "npart                " << npart                   << std::endl;
  outFile << "masses               " ;
  for(auto i : masses)
    outFile << i << "  ";
  outFile << std::endl;
  outFile << "kindPotential        " << kindPotential           << std::endl;
  for(auto pairs : twoBodyParameters)
  {
    outFile << "twoBodyParameters    " ;
    for(auto i  : pairs)
      outFile << i  << "  ";
    outFile << std::endl;
  }
  outFile << "kindThreeBody        " << kindThreeBody           << std::endl;
  outFile << "threeBodyParameters  " ;
  for(auto i : threeBodyParameters)
    outFile << i << "  ";
  outFile << std::endl;
  outFile << "threeBodyParametersSpin  " ;
  for(auto i : threeBodyParametersSpin)
    outFile << i << "  ";
  outFile << std::endl;
  outFile << "fourBodyParameters  " ;
  for(auto i : fourBodyParameters)
    outFile << i << "  ";
  outFile << std::endl;
  outFile << "kindSymmetrization   " << kindSymmetrization       << std::endl;
  for(auto pairs : alphaRanges)
  {
    outFile << "alphaRanges          " ;
    for(auto i  : pairs)
      outFile << i  << "  ";
    outFile << std::endl;
  }
  outFile << "# ------------------------------------------------------- " << std::endl;
  outFile << "% END OF FILE"                           << std::endl;

  
  return 0;
}
/******************************************************************************
 *                        ---------------------                               *
 ******************************************************************************/
// write to the stream 
std::ostream& operator<< (std::ostream& outFile, const InputData & in)
{
  outFile << std::setprecision(std::numeric_limits<long double>::digits10);
  outFile << "# ------------------------------------------------------- " << std::endl;
  outFile << "# Input data for SVM        " << std::endl;
  outFile << "# ------------------------------------------------------- " << std::endl;
  outFile << "dataFile             " << in.dataFile      << std::endl;
  outFile << "sizeBasis            " << in.sizeBasis     << std::endl;
  outFile << "parity               " << in.parity        << std::endl;
  outFile << "angularMomentum      " << in.angularMomentum         << std::endl;
  outFile << "totalMomentum        " << in.totalMomentum           << std::endl;
  outFile << "totalSpin            " << in.totalSpin               << std::endl;
  outFile << "totalIsospin         " << in.totalIsospin            << std::endl;
  outFile << "ifNoLocalAlpha	   " << in.ifNoLocalAlpha	   << std::endl;
  outFile << "ifJJCoupling         " << in.ifJJCoupling            << std::endl;
  outFile << "ifCoulomb            " << in.ifCoulomb               << std::endl;
  outFile << "ifThreeBody          " << in.ifThreeBody             << std::endl;
  outFile << "ifThreeBodySpin      " << in.ifThreeBodySpin         << std::endl;
  outFile << "ifFourBody           " << in.ifFourBody              << std::endl;
  outFile << "ifHarmonicTrap       " << in.ifHarmonicTrap          << std::endl;
  outFile << "energyUnits          " << in.energyUnits   << std::endl;
  outFile << "lengthUnits          " << in.lengthUnits   << std::endl;
  outFile << "harmonicLength    " << in.harmonicLength  << std::endl;
  outFile << "npart                " << in.npart        << std::endl;
  outFile << "masses               " ;
  for(auto i : in.masses)
    outFile << i << "  ";
  outFile << std::endl;
  outFile << "kindPotential        " << in.kindPotential           << std::endl;
  for(auto pairs : in.twoBodyParameters)
  {
    outFile << "twoBodyParameters    " ;
    for(auto i  : pairs)
      outFile << i  << "  ";
    outFile << std::endl;
  }
  outFile << "kindThreeBody        " << in.kindThreeBody           << std::endl;
  outFile << "threeBodyParameters  " ;
  for(auto i : in.threeBodyParameters)
    outFile << i << "  ";
  outFile << std::endl;
  outFile << "threeBodyParametersSpin  " ;
  for(auto i : in.threeBodyParametersSpin)
    outFile << i << "  ";
  outFile << std::endl;
  outFile << "fourBodyParameters  " ;
  for(auto i : in.fourBodyParameters)
    outFile << i << "  ";
  outFile << std::endl;
  outFile << "kindSymmetrization   " << in.kindSymmetrization       << std::endl;
  for(auto pairs : in.alphaRanges)
  {
    outFile << "alphaRanges          " ;
    for(auto i  : pairs)
      outFile << i  << "  ";
    outFile << std::endl;
  }
  outFile << "# ------------------------------------------------------- " << std::endl;
  outFile << "% END OF FILE"                           << std::endl;

  return outFile;
}

/******************************************************************************
 *                        ---------------------                               *
 ******************************************************************************/
// test if the potential has been implemented
int
InputData::testKindPotential(std::string inPot)
{

  int test=-1;
  if(mapPotential.find(inPot) == mapPotential.end())
  {
    std::cerr << "The potential " << inPot << " has not been implemented" << std::endl;
    abort();
  } else 
  {
    test = 0;

   }
  return test;
}
// test if the 3Body potential has been implemented
int
InputData::testKindThreeBody(std::string inPot)
{

  int test=-1;
  if(mapThreeBody.find(inPot) == mapThreeBody.end())
  {
    std::cerr << "The 3B potential " << inPot << " has not been implemented" << std::endl;
    abort();
  } else 
  {
    test = 0;

   }
  return test;
}
// test if the symmetrization has been implemented
int
InputData::testKindSymmetrization(std::string in)
{

  int test=-1;
  if(mapSymmetrization.find(in) == mapSymmetrization.end())
  {
    std::cerr << "The Symmetrization " << in << " has not been implemented" << std::endl;
    abort();
  } else 
  {
    test = 0;

   }
  return test;
}
/**
 * @file inputData.cc
 */
#include"inputData.h"
