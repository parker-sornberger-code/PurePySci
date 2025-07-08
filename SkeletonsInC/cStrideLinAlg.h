#ifndef CSTRIDELINALG_H
#define CSTRIDELINALG_H

#include "cStrideArray.h"

typedef struct {
    Array** Arr1;
    Array** Arr2;
} ArrTuple_2;
typedef struct {
    Array** Arr1;
    Array** Arr2;
    Array** Arr3;
} ArrTuple_3;
void FreeDoubleTuple(ArrTuple_2* ret);
void FreeTripleTuple(ArrTuple_3* ret);
Array* proj(Array* v1, Array* v2);
Array* Givens(size_t row1, size_t row2, size_t col, Array* mat);
Array* mult_for_Givens(Array* A, Array* B, size_t g_i);
Array* mult_for_Givens_general(Array* A, Array* B, size_t g_i, size_t g_j);
ArrTuple_2* UpperHess(Array* mat);
ArrTuple_2* QR_g_for_h( Array* mat);
ArrTuple_3* HessQR(Array* mat, int iterations, float tol);



#endif

