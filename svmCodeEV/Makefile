CPP=g++

LIBHDF5= /usr/include/hdf5/serial/
EIGEN= /home/paolo/PhD_Nice/codiciSVM/eigen/

COPT= -O3 -DNDEBUG -mavx -mfma -std=c++14 -Wall -I$(EIGEN)  -I$(LIBHDF5) -no-pie
COPT= -O3 -DNDEBUG -Wall -std=c++11 -I$(EIGEN) -I$(LIBHDF5) -I/usr/include -no-pie


#COPT= -Wall -ggdb -std=c++11 -I$(EIGEN)-I$(CPPHDF5) -I$(LIBHDF5)
LIBS= -lgsl -llapack -lblas -lhdf5 -lgslcblas  -lm # -larpack -lgfortran -fopenmp 
#LIBS= -llapack -lblas -lhdf5 `gsl-config --cflags --libs` # -larpack -lgfortran -fopenmp 
PHDF5=/usr/lib/x86_64-linux-gnu/hdf5/serial

CPPFLAGS=$(COPT)

OBJECTS= check_energy.o inputData.o coordinates.o basis.o akpot.o


#---------------- Main programs --------------------------------------
all: svmThree 
svmThree: ${OBJECTS} Makefile
	  ${CPP} -o $@ ${OBJECTS} -L${PHDF5} ${LIBS} `gsl-config --cflags --libs`  

#---------------- Objects -------------------------------------------
check_energy.o: check_energy.cc 
	${CPP} ${CPPFLAGS} -c  $<
inputData.o: inputData.cc inputData.h
	${CPP} ${CPPFLAGS} -c  $<
coordinates.o: coordinates.cc coordinates.h
	${CPP} ${CPPFLAGS} -c  $<
basis.o: basis.cc basis.h
	${CPP} ${CPPFLAGS} -c  $<
akpot.o: akpot.f
	  gfortran -c $<
#----------------- Clean --------------------------------------------
clean:
	@rm *.o svmThree svmThreeP
