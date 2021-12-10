#include "matmul.h"

void matmul(int* A, int* B, int* C, int* group, int* tag) {


int localA[MAX_SIZE][MAX_SIZE][comb];
#pragma HLS ARRAY_PARTITION variable=localA dim = 1 complete

int localB[MAX_SIZE][MAX_SIZE];
#pragma HLS ARRAY_PARTITION variable=localB dim = 0 complete

int localC[MAX_SIZE][MAX_SIZE];
#pragma HLS ARRAY_PARTITION variable = localC dim = 2 complete

int localGroup[MAX_SIZE];
#pragma HLS ARRAY_PARTITION variable = localGroup dim = 0 complete

int localTag[MAX_SIZE][MAX_SIZE];
#pragma HLS ARRAY_PARTITION variable = localTag dim = 0 complete

for (int k = 0; k < b_row; ++k)
	localGroup[k] = group[k];

int group_idx = 0;
int row_idx = 0;
for (int k = 0; k < a_col; ++k) {
	if (row_idx == localGroup[group_idx]) {
		row_idx = 0;
		group_idx += 1;
	}
	for (int i = 0; i < a_row; ++i) {
	#pragma HLS UNROLL
        localA[i][group_idx][row_idx] = A[i * a_col + k];
	}
	row_idx += 1;
}

for (int k = 0; k < b_row; ++k)
    for (int j = 0; j < b_col; ++j) {
        localB[k][j] = B[k * b_col + j];
		localTag[k][j] = tag[k * b_col + j];
	}
systolic1:
    for (int i = 0; i < a_row; i++) {
       #pragma HLS LOOP_TRIPCOUNT min=8 max=8
       #pragma HLS PIPELINE II=1
    systolic2:
	int p_sum[MAX_SIZE][MAX_SIZE];
#pragma HLS ARRAY_PARTITION variable = p_sum dim = 0 complete
        for (int j = 0; j < MAX_SIZE; j++) {
        systolic3:
            for (int k = 0; k < MAX_SIZE; k++) {
                int last = (k == 0) ? 0 : p_sum[k - 1][j];
                int a_val;
				switch(localTag[k][j]) {
					case 0:
						a_val = localA[i][k][0];
						break;
					case 1:
						a_val = localA[i][k][1];
						break;
					case 2:
						a_val = localA[i][k][2];
						break;
				}
				a_val = (i < a_row && k < a_col) ? a_val : 0;
                int b_val = (k < b_row && j < b_col) ? localB[k][j] : 0;
                p_sum[k][j] = last + a_val * b_val;
            }
                localC[i][j] = p_sum[MAX_SIZE - 1][j];
	}
    }

for (int i = 0; i < a_row; ++i)
    for (int j = 0; j < b_col; ++j)
        C[i * b_col + j] = localC[i][j];

return;
}
