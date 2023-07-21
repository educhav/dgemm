/* #include <x86intrin.h> */
#include <immintrin.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define SIZE 32

void dgemm_slow(size_t n, double* A, double* B, double* C) {
    for(size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < n; j++) {
            for(size_t k = 0; k < n; k++) {
                C[(i*n)+j] += A[(i*n)+j] * B[(j*n)+k];
            }
        }
    }
}

void dgemm_DLP(size_t n, double* A, double* B, double* C) {
    for(size_t i = 0; i < n; i += 4) {
        for(size_t j = 0; j < n; j += 4) {
            __m256d c0 = _mm256_load_pd(C+i+j*n);
            for(size_t k = 0; k < n; k++) {
                c0 = _mm256_add_pd(c0, 
                        _mm256_mul_pd(
                            _mm256_load_pd(A+i+k*n), 
                            _mm256_broadcast_sd(B+k+j*n)
                        )
                    );
            }
        }
    }
}

int main() {
    double* A = malloc(sizeof(double) * 32);
    double* B = malloc(sizeof(double) * 32);
    double* C = malloc(sizeof(double) * 32);

    for(int i = 0; i < 32; i++) {
        C[i] = 0;
    }


    /* double diff_t; */
    /* time_t st, et; */
    clock_t st, et;
    double total_t;

    st = clock();
    for(int i = 0; i < 100; i++ ) {
        dgemm_slow(32, A, B, C);
    }
    et = clock();

    total_t = ((double)et-st) / CLOCKS_PER_SEC;

    printf("DGEMM with no optimization: Execution time = %f\n", total_t);

    st = clock();
    for(int i = 0; i < 100; i++ ) {
        dgemm_DLP(32, A, B, C);
    }
    et = clock();

    total_t = ((double)et-st) / CLOCKS_PER_SEC;

    printf("DGEMM with DLP: Execution time = %f\n", total_t);


}
