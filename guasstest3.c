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
float A[MAXN][MAXN], B[MAXN], X[MAXN];
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
	int i, norm, row, col;
	float mult;
	int map[MAXN];

	MPI_Init();

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);   /* get current process id */
	MPI_Comm_size(MPI_COMM_WORLD, &procs); /* get number of processes */

	MPI_Bcast(&A[0][0], N*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&B, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

	for (i = 0; i < N; i++)
	{
		map[i] = i % procs;
	}

	for (norm = 0; norm < N; norm++) {
		MPI_Bcast(&A[norm][norm], N - norm, MPI_FLOAT, map[norm], MPI_COMM_WORLD);
		MPI_Bcast(&B[norm], 1, MPI_FLOAT, map[norm], MPI_COMM_WORLD);
		for (row = norm + 1; row < N; row++) {
			if (map[row] == rank) {
				mult = A[row][norm] / A[norm][norm];
			}
		}
		for (row = norm + 1; row < N; row++) {
			if (map[row] == rank) {
				for (col = 0; col < N; col++) {
					A[row][col] -= mult * A[norm][col];
				}
				B[row] -= (mult * B[norm]);
			}
		}
	}

	/* Back substitution */
	for (row = N - 1; row >= 0; row--) {
		X[row] = B[row];
		for (col = N - 1; col > row; col--) {
			X[row] -= A[row][col] * X[col];
		}
		X[row] /= A[row][row];
	}
}