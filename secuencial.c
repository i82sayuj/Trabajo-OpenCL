#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TAM_MAT 4000

float A[TAM_MAT][TAM_MAT];
float B[TAM_MAT][TAM_MAT];
float C[TAM_MAT][TAM_MAT];

int main(int argc, char** argv) {
	srand(time(NULL));

	clock_t t0, t1;
	double segundos = 0;
	t0 = clock();


	for (int i = 0; i < TAM_MAT; i++) {
		for (int j = 0; i < TAM_MAT; i++) {

			A[i][j] = rand() / (float)RAND_MAX;
			B[i][j] = rand() / (float)RAND_MAX;
			C[i][j] = 0;
		}
	}

	for (int i = 0; i < TAM_MAT; i++) {
		for (int j = 0; j < TAM_MAT; j++) {
			for (int k = 0; k < TAM_MAT; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}

	t1 = clock();
	segundos = (double)(t1 - t0) / CLOCKS_PER_SEC;
	
	printf("El tiempo en secuencial es: %f\n", segundos);
	printf(" C[0][0] = %f", C[0][0]);

}