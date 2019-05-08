#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <limits.h>
#include <pthread.h>
#include <string.h>
#include "mpi.h"

/* Program Parameters */
#define MAXN 2000  /* Max value of N */
int N;  /* Matrix size */
int procs, rank;  /* Number of processors to use */

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
				* It is this routine that is timed.
				* It is called only on the parent.
				*/


				/* returns a seed for srand based on the time */
unsigned int time_seed() {
	struct timeval t;
	struct timezone tzdummy;

	gettimeofday(&t, &tzdummy);
	return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
	int submit = 0;  /* = 1 if submission parameters should be used */
	int seed = 0;  /* Random seed */
	char uid[8] = "Nicholas"; /*User name */

	/* Read command-line arguments */
	srand(time_seed());  /* Randomize */
	if (argc != 3) {
		if (argc == 2 && !strcmp(argv[1], "submit")) {
			/* Use submission parameters */
			submit = 1;
			N = 4;
			procs = 2;
			printf("\nSubmission run for \"%s\".\n", uid);
			srand(randm());
		}
		else {
			if (argc == 4) {
				seed = atoi(argv[3]);
				srand(seed);
				printf("Random seed = %i\n", seed);
			}
			else {
				printf("Usage: %s <matrix_dimension> <num_procs> [random seed]\n",
					argv[0]);
				printf("       %s submit\n", argv[0]);
				exit(0);
			}
		}
	}
	/* Interpret command-line args */
	if (!submit) {
		N = atoi(argv[1]);
		if (N < 1 || N > MAXN) {
			printf("N = %i is out of range.\n", N);
			exit(0);
		}
		procs = atoi(argv[2]);
		if (procs < 1) {
			printf("Warning: Invalid number of processors = %i.  Using 1.\n", procs);
			procs = 1;
		}
	}

	/* Print parameters */
	printf("\nMatrix dimension N = %i.\n", N);
	printf("Number of processors = %i.\n", procs);

}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
	int row, col;

	printf("\nInitializing...\n");
	for (col = 0; col < N; col++) {
		for (row = 0; row < N; row++) {
			A[row][col] = (float)rand() / 32768.0;
		}
		B[col] = (float)rand() / 32768.0;
		X[col] = 0.0;
	}

}

/* Print input matrices */
void print_inputs() {
	int row, col;

	if (N < 10) {
		printf("\nA =\n\t");
		for (row = 0; row < N; row++) {
			for (col = 0; col < N; col++) {
				printf("%5.2f%s", A[row][col], (col < N - 1) ? ", " : ";\n\t");
			}
		}
		printf("\nB = [");
		for (col = 0; col < N; col++) {
			printf("%5.2f%s", B[col], (col < N - 1) ? "; " : "]\n");
		}
	}
}

void print_X() {
	int row;

	if (N < 10) {
		printf("\nX = [");
		for (row = 0; row < N; row++) {
			printf("%5.2f%s", X[row], (row < N - 1) ? "; " : "]\n");
		}
	}
}

void main(int argc, char **argv) {
	/* Timing variables */
	struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
	struct timezone tzdummy;
	clock_t etstart2, etstop2;  /* Elapsed times using times() */
	unsigned long long usecstart, usecstop;
	struct tms cputstart, cputstop;  /* CPU times for my processes */

	/* Process program parameters */
	parameters(argc, argv);

	/* Initialize A and B */
	initialize_inputs();

	/* Print input matrices */
	print_inputs();

	/* Start Clock */
	printf("\nStarting clock.\n");
	gettimeofday(&etstart, &tzdummy);
	etstart2 = times(&cputstart);

	/* Gaussian Elimination */
	gauss();

	/* Stop Clock */
	gettimeofday(&etstop, &tzdummy);
	etstop2 = times(&cputstop);
	printf("Stopped clock.\n");
	usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
	usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

	/* Display output */
	print_X();

	/* Display timing results */
	printf("\nElapsed time = %g ms.\n",
		(float)(usecstop - usecstart) / (float)1000);
	/*printf("               (%g ms according to times())\n",
	 *       (etstop2 - etstart2) / (float)CLOCKS_PER_SEC * 1000);
	 */
	 /*
		 printf("(CPU times are accurate to the nearest %g ms)\n",
			 1.0 / (float)CLOCKS_PER_SEC * 1000.0);
		 printf("My total CPU time for parent = %g ms.\n",
			 (float)((cputstop.tms_utime + cputstop.tms_stime) -
			 (cputstart.tms_utime + cputstart.tms_stime)) /
				 (float)CLOCKS_PER_SEC * 1000);
		 printf("My system CPU time for parent = %g ms.\n",
			 (float)(cputstop.tms_stime - cputstart.tms_stime) /
			 (float)CLOCKS_PER_SEC * 1000);
		 printf("My total CPU time for child processes = %g ms.\n",
			 (float)((cputstop.tms_cutime + cputstop.tms_cstime) -
			 (cputstart.tms_cutime + cputstart.tms_cstime)) /
				 (float)CLOCKS_PER_SEC * 1000);
		 /* Contrary to the man pages, this appears not to include the parent */
	printf("--------------------------------------------\n");

}

/* ------------------ Above Was Provided --------------------- */

void gauss() {
	MPI_Status status;
	MPI_Request request;
	int row, col, i, norm, rank;
	float mult;

	MPI_Barrier(MPI_COMM_WORLD);

	/* proccessor rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	/* Find out how many processes are being used */
	MPI_Comm_size(MPI_COMM_WORLD, &procs);

	/* Array with the row size and number of rows that each processor will handle */
	int * first_row_A_array = (int*)malloc(procs * sizeof(int));
	int * n_of_rows_A_array = (int*)malloc(procs * sizeof(int));
	int * first_row_B_array = (int*)malloc(procs * sizeof(int));
	int * n_of_rows_B_array = (int*)malloc(procs * sizeof(int));
	for (i = 0; i < procs; i++) {
		first_row_A_array[i] = 0;
		n_of_rows_A_array[i] = 0;
		first_row_B_array[i] = 0;
		n_of_rows_B_array[i] = 0;
	}

	/* Main loop. After every iteration, a new column will have all 0 values down the [norm] index */
	for (norm = 0; norm < N - 1; norm++) {

		MPI_Bcast(&A[N*norm], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&B[norm], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		/* subset of rows of this iteration */
		int subset = N - 1 - norm;
		/* number that indicates the step as a float */
		float step = ((float)subset) / (procs);
		/* First and last rows that this process will work into for this iteration */
		int first_row = norm + 1 + ceil(step * (rank));
		int last_row = norm + 1 + floor(step * (rank + 1));
		if (last_row >= N) last_row = N - 1;
		int number_of_rows = last_row - first_row + 1;

		if (rank == 0) {

			for (i = 1; i < procs; i++) {

				/* We send to each process the amount of data that they are going to handle */
				int first_row_rmte = norm + 1 + ceil(step * (i));
				int last_row_rmte = norm + 1 + floor(step * (i + 1));
				if (last_row_rmte >= N) last_row_rmte = N - 1;
				int number_of_rows_rmte = last_row_rmte - first_row_rmte + 1;

				/* In case this process isn't assigned any task, continue. This happens when there are more processors than rows */
				//if( number_of_rows_rmte < 1 || first_row_rmte >= N ) continue;

				if (number_of_rows_rmte < 0) number_of_rows_rmte = 0;
				if (first_row_rmte >= N) { number_of_rows_rmte = 0; first_row_rmte = N - 1; };

				first_row_A_array[i] = first_row_rmte * N;
				first_row_B_array[i] = first_row_rmte;
				n_of_rows_A_array[i] = number_of_rows_rmte * N;
				n_of_rows_B_array[i] = number_of_rows_rmte;

			}

		}

		MPI_Scatterv(
			&A[0],              // send buffer
			n_of_rows_A_array,  // array with number of elements in each chunk
			first_row_A_array,  // array with pointers to initial element of each chunk
			MPI_DOUBLE,          // type of elements to send
			&A[first_row * N],  // receive buffer
			N * number_of_rows, // number of elements to receive
			MPI_DOUBLE,          // type of elements to receive
			0,					// who sends
			MPI_COMM_WORLD
		);
		MPI_Scatterv(
			&B[0],
			n_of_rows_B_array,
			first_row_B_array,
			MPI_DOUBLE,
			&B[first_row],
			number_of_rows,
			MPI_DOUBLE,
			0,
			MPI_COMM_WORLD
		);

		/*  Gaussian elimination                   */

		if (number_of_rows > 0 && first_row < N) {
			/* Similar code than in the sequential case */
			for (row = first_row; row <= last_row; row++) {

				mult = A[row][norm] / A[norm][norm];
				for (col = norm; col < N; col++) {
					A[row][col] -= A[norm][col] * mult;
				}

				B[row] -= B[norm] * mult;
			}
		}


		/* --------------------------------------- */
		/*  Send back the results                  */
		/*  -------------------------------------- */
		/* Sender side */

		if (rank != 0) {
			if (number_of_rows > 0 && first_row < N) {
				MPI_Isend(&A[first_row * N], N * number_of_rows, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);
				MPI_Isend(&B[first_row], number_of_rows, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);
			}
		}
		/* Receiver side */
		else {

			for (i = 1; i < procs; i++) {

				// In case this process isn't assigned any task, continue. This happens when there are more processors than rows 
				if (n_of_rows_B_array[i] < 1 || first_row_B_array[i] >= N) continue;

				MPI_Recv(&A[first_row_A_array[i]], n_of_rows_A_array[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(&B[first_row_B_array[i]], n_of_rows_B_array[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
			}


		}

	}

	/* Back substitution */
	if (rank == 0) {
		for (row = N - 1; row >= 0; row--)
		{
			X[row] = B[row];
			for (col = N - 1; col > row; col--)
			{
				X[row] -= A[row][col] * X[col];
			}
			X[row] /= A[row][row];
		}
	}


	MPI_Barrier(MPI_COMM_WORLD);

	/* Free memory used for the arrays that we allocated previously */
	free_memory();

	MPI_Finalize();
}

