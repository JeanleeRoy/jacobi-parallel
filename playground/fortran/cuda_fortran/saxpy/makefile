OBJECTS = main.o cu_saxpy.o saxpy.o

GPU_CC = nvcc
GPU_FC = nvfortran

GPU_FLAGS = -cuda
FC_GPU_FLAGS = -lstdc++ -L/usr/local/cuda/lib64/ -lcudart

FC = gfortran

main: $(OBJECTS)
	$(FC) -o saxpy $(OBJECTS) $(FC_GPU_FLAGS)

cu_saxpy.o: cu_saxpy.f90
	$(FC) -c cu_saxpy.f90

cu_saxpy.mod: cu_saxpy.o

main.o: main.f90 cu_saxpy.mod
	$(FC) -c main.f90

saxpy.o: saxpy.cu
	$(GPU_CC) -c saxpy.cu

clean:
	rm -f main saxpy *.o *.mod
