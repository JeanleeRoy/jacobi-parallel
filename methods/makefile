OBJECTS = main.o jacobi_method.o jacobi_cuda.o jacobi_acc.o

GPU_CC = nvcc

PGCC = pgcc -g -acc -gpu=nvidia -Minfo=accel -DPGI

FC_GPU_FLAGS = -lstdc++ -L/usr/local/cuda/lib64/ -lcudart -ldl -lm
FC = gfortran


jacobi: $(OBJECTS)
	$(FC) -o jacobi $(OBJECTS) $(FC_GPU_FLAGS)

gpu_code.o: $(OBJECTS)
	$(GPU_CC) -arch=sm_52 -dlink $(OBJECTS) -lm -o gpu_code.o

jacobi_cuda.o: jacobi_cuda.cu
	$(GPU_CC) -c jacobi_cuda.cu -lm

jacobi_acc.o: jacobi_acc.cu
	$(GPU_CC) -c jacobi_acc.cu -o $@ -lm

jacobi_method.o: jacobi_method.f90
	$(FC) -c jacobi_method.f90

jacobi_method.mod: jacobi_method.o

main.o: main.f90 jacobi_method.mod
	$(FC) -c main.f90

clean:
	rm -f jacobi *.o *.mod
