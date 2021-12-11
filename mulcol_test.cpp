#include <iostream>
#include <cassert>
#include <stdlib.h>
#include "matmul.h"

int main() {
    float A[a_row][a_col];
	float B[b_row][b_col];
    int group[b_row];
	int tag[b_row][b_col];
	assert(a_col == b_row * comb);
	float C[a_row][b_col];
    float C_sw[a_row][b_col];

    for (int i = 0; i < a_row; ++i)
        for (int k = 0; k < a_col; ++k)
            A[i][k] = rand() % 100 / 10.0;
    
    for (int k = 0; k < b_row; ++k) {
		group[k] = comb;
		for (int j = 0; j < b_col; ++j) {
            B[k][j] = rand() % 100 / 10.0;
			tag[k][j] = rand() % comb;
		}
	}

    matmul(&A[0][0], &B[0][0], &C[0][0], &group[0], &tag[0][0]);
    int cnt_error = 0;
    for (int i = 0; i < a_row; ++i)
        for (int j = 0; j < b_col; ++j) {
            C_sw[i][j] = 0;
            for (int k = 0; k < a_col; ++k) {
				if (tag[k / comb][j] == k % comb)
					C_sw[i][j] += A[i][k] * B[k / comb][j];
            }
            if (C_sw[i][j] != C[i][j])
                cnt_error++;
        }
    if (cnt_error > 0) {
        std::cout << "error number: " << cnt_error << std::endl;
        std::cout << "test fail" << std::endl;
	}
	else {	
        std::cout << "test pass" << std::endl;
	}
		std::cout << "matrix A:" << std::endl;
		for (int i = 0; i < a_row; i++) {
			for (int k = 0; k < a_col; k++) {
				std::cout << A[i][k];
				if (k < a_col - 1)
					std::cout << ", ";
			}
			std::cout << std::endl;
		}

		std::cout << "matrix B:" << std::endl;
		for (int k = 0; k < b_row; k++) {
			for (int j = 0; j < b_col; j++) {
				std::cout << B[k][j] << "(" << tag[k][j] << "), ";
			}
			std::cout << std::endl;
		}

		std::cout << "matrix C:" << std::endl;
		for (int i = 0; i < a_row; i++) {
			for (int j = 0; j < b_col; j++) {
				std::cout << C_sw[i][j] << ", ";
			}
			std::cout << std::endl;
		}
}
