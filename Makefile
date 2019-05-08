mpi_matrixmul: mpi_matrixmul.c
	mpicc -o mpi_matrixmul mpi_matrixmul.c

gauss: gauss.c
	mpicc -o gauss gauss.c

clean:
	rm -f *.exe 
