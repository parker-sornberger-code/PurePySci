
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "cStrideArray.h"
void pArr(Array* Arr, bool do_free){
    

    printf("Rows: %zu Cols: %zu\n", Arr->rows, Arr->cols);
    size_t elems = Arr->elems;
    size_t rows = Arr->rows;
    size_t cols = Arr->cols;

    
    for(size_t i=0; i<rows; i++){
        for(size_t j=0; j<cols; j++){

            printf("%.2f ", Arr->arr[i*cols + j]);
        };
        printf("\n");
    };
    
    printf("\n");
    if (do_free)
        FreeArr(Arr);

};
void FreeArr(Array* Arr){
    free(Arr->arr);
    free(Arr);
    return;

};



Array* fill(size_t rows, size_t cols, float fill_with){
    size_t elems = rows * cols;
    //float* arr = (float*)malloc(sizeof(float)*elems);
    Array* Arr = (Array*)malloc(sizeof(Array));
    Arr->arr = (float*)malloc(sizeof(float)*elems);
    for(size_t i=0; i<elems; i++){
        Arr->arr[i] = fill_with;
    };
    Arr->rows = rows;
    Arr->cols = cols;
    Arr->elems = elems;
    Arr->transposed = false;
    return Arr;

};
Array* zeros(size_t rows, size_t cols){

    return fill(rows, cols, 0.0f);

};
Array* ones(size_t rows, size_t cols){

    return fill(rows, cols, 1.0f);

};
Array* arange(size_t maxsize){

    Array* Arr = (Array*)malloc(sizeof(Array));
    Arr->arr = (float*)malloc(sizeof(float)*maxsize);
    Arr->elems = maxsize;
    Arr->rows = maxsize;
    size_t cols = 1;
    Arr->cols = cols;
    Arr->transposed = false;
    float count = 0.0f;
    for(size_t i=0; i<maxsize;i++){
        Arr->arr[i] = count;
        count++;

    };
    return Arr;

};

Array* shape_arange(size_t maxsize, size_t rows, size_t cols){
    Array* Arr = arange(maxsize);
    reshape(Arr, rows, cols);
    return Arr;

};
Array* copy_transpose(Array* Arr){ //creates a new array
    Array* tArr = (Array*)malloc(sizeof(Array));
    size_t elems = Arr->elems;
    tArr->arr = (float*)malloc(sizeof(float)*elems);
    size_t rows = Arr-> rows;
    size_t cols = Arr->cols;
    tArr->rows = cols;
    tArr->cols = rows;
    tArr->elems = elems;
    tArr->transposed = ~Arr->transposed;

    for(size_t i=0; i<rows; i++){
        for(size_t j=0; j<cols;j++){
            tArr->arr[i + rows*j] = Arr->arr[rows*i + j];
        };

    };
    return tArr;
};
int shallow_transpose(Array* Arr){

    if (Arr->rows != Arr->cols){
        Arr->transposed = true;
        return reshape(Arr, Arr->cols, Arr->rows);

    };

    Arr->transposed=true;
    return 0;
};


int reshape(Array* arr, size_t new_rows, size_t new_cols) {
    if (new_rows * new_cols != arr->elems) {
        return -1; // Invalid reshape
    }
    arr->rows = new_rows;
    arr->cols = new_cols;
    return 0;
}

Array* get_row(Array* arr, size_t which_row, bool transposed){

    size_t rows = arr->rows;
    size_t cols = arr->cols;
    size_t elems = 0;
    if (transposed==true)
        elems = rows;
    
    else
         elems = cols;
    Array* Row = (Array*)malloc(sizeof(Array));
    Row->arr = (float*)malloc(sizeof(float)*elems);
    Row->rows = 1;
    Row->cols = elems;
    Row->elems = elems;
    Row->transposed = transposed; // Row->transposed = arr->transposed;
    size_t begin = which_row * cols;

    for(size_t i = 0; i<elems;i++){
        Row->arr[i] = arr->arr[begin+i];

    };

    return Row;

};
Array* get_col(Array* arr, size_t which_col, bool transposed){

    size_t rows = arr->rows;
    size_t cols = arr->cols;
    size_t elems = 0;
    if (transposed==true)
        elems = cols;
    
    else
         elems = rows;
    Array* Col = (Array*)malloc(sizeof(Array));
    Col->arr = (float*)malloc(sizeof(float)*elems);
    Col->rows = elems;
    Col->cols = 1;
    Col->elems = elems;
    Col->transposed = transposed; //Row->transposed = arr->transposed;

    size_t index = which_col;

    for(size_t i = 0;i<rows; i++){
        Col->arr[i] = arr->arr[index];

        index += cols;
        
    };

    return Col;


};
float dot(Array* row, Array* col){

    float val=0.0f;
    size_t elems = row->elems;

    for(size_t i = 0; i < elems; i++){

        val = val +  (row->arr[i]*col->arr[i]);
    };
    return val;
};
Array* matmul(Array* arr1, Array* arr2){
    size_t rows = arr1->rows;
    size_t cols = arr2->cols;
    Array* result = zeros(arr1->rows, arr2->cols);

    Array* row;
    Array* col;
    float val;
    float pval;
    size_t index;
    for(size_t i = 0; i<rows; i++){
        row = get_row(arr1, i, false);
        for(size_t j = 0; j < cols; j++){
            col = get_col(arr2, j, false);
            val = dot(row, col);
            index = i * cols + j;
            pval = result->arr[index];
            result->arr[index] = pval + val;
        };
    };

    return result;
};
Array* eye(size_t dim){

    Array* iden = zeros(dim, dim);
    for(size_t i = 0; i < dim; i++){
        iden->arr[i * dim + i] = 1.0;
    };
    return iden;
};
Array* get_diag(Array* arr){
    Array* diag = (Array*)malloc(sizeof(Array));
    size_t cols = arr->cols;
    diag->cols = cols;
    diag->rows = 1;
    diag->transposed = false;
    diag->elems = cols;
    diag->arr = (float*)malloc(sizeof(float)*cols);

    for(size_t i = 0; i<cols; i++){
        diag->arr[i] = arr->arr[i*cols + i]; 
    };
    return diag;

};
Array* full_copy(Array* arr){

    Array* acopy = (Array*)malloc(sizeof(Array));
    size_t elems = arr->elems;
    acopy->arr = (float*)malloc(sizeof(float)*elems);
    acopy->arr = arr->arr;
    acopy->elems = elems;
    acopy->rows = arr->rows;
    acopy->cols = arr->cols;
    acopy->transposed = arr->transposed;
    return acopy;
};
Array* make_upper_diag(Array* arr){
    Array* udiag = full_copy(arr);
    size_t rows = arr->rows;
    size_t cols = arr->cols;
    for(size_t i = 0; i<rows; i++){
        for(size_t j=0; j<i; j++){
            udiag->arr[i*cols + j]=0.0;
        };
    };
    return udiag;
};
void make_upper_diag_inplace(Array* arr){
    size_t rows = arr->rows;
    size_t cols = arr->cols;
    for(size_t i = 0; i<rows; i++){
        for(size_t j=0; j<i; j++){
            arr->arr[i*cols + j]=0.0;
        };
    };
    return;
};
float _add_2(float v1, float v2){

    return v1+v2;

};
float _sub_2(float v1, float v2){
    return v1-v2;

};
float _mul_2(float v1, float v2){
    return v1*v2;
};
float _div_2(float v1, float v2){
    return v1/v2;
};
Array* arr_pm_arr(Array* arr1, Array* arr2, bool negate_2nd){
    size_t rows = arr1->rows;
    size_t cols =arr1->cols;
    size_t elems = arr1->elems;
    Array* result = zeros(rows, cols);
    float v1;
    float v2;
    float res;
    float (*mathptr)(float, float);
    if (negate_2nd)
        mathptr = &_sub_2;
    else
        mathptr = &_add_2;

    //switch negate_2nd;
    for (size_t i = 0; i<elems; i++){
        v1 = arr1->arr[i];
        v2 = arr2->arr[i];
        res = mathptr(v1, v2);
        result->arr[i] = res;
    };

    return result;
};
Array* sc_pm_arr(float scalar, Array* arr, bool negate_2nd){
    size_t rows = arr->rows;
    size_t cols =arr->cols;
    Array* result = zeros(arr->rows, arr->cols);
    size_t elems = arr->elems;
    float v1=scalar;
    float v2;
    float res;
    float (*mathptr)(float, float);
    if (negate_2nd)
        mathptr = &_sub_2;
    else
        mathptr = &_add_2;
    for(size_t i=0; i<elems; i++){
        
        v2 = arr->arr[i];
        res = mathptr(v1, v2);
        result->arr[i] = res;

    };
    return result;
};
Array* arr_pm_sc(Array* arr, float scalar, bool negate_2nd){
    size_t rows = arr->rows;
    size_t cols =arr->cols;
    Array* result = zeros(arr->rows, arr->cols);
    size_t elems = arr->elems;
    float v1;
    float v2=scalar;
    float res;
    float (*mathptr)(float, float);
    if (negate_2nd)
        mathptr = &_sub_2;
    else
        mathptr = &_add_2;
    for(size_t i=0; i<elems; i++){
        v1 = arr->arr[i];
        res = mathptr(v1, v2);
        result->arr[i] = res;

    };
    return result;
};
Array* arr_times_arr(Array* arr1, Array* arr2, bool divide_2nd){
    size_t rows = arr1->rows;
    size_t cols =arr1->cols;
    Array* result = zeros(rows, cols);
    size_t elems = arr1->elems;
    float v1;
    float v2;
    float res;
    float (*mathptr)(float, float);
    if (divide_2nd)
        mathptr = &_div_2;
    else
        mathptr = &_mul_2;
    for(size_t i=0; i<elems; i++){
        v1 = arr1->arr[i];
        v2 = arr2->arr[i];
        res = mathptr(v1, v2);
        result->arr[i] = res;

    };
    return result;
};
Array* arr_times_sc(Array* arr, float scalar, bool divide_2nd){
    size_t rows = arr->rows;
    size_t cols =arr->cols;
    Array* result = zeros(arr->rows, arr->cols);
    size_t elems = arr->elems;
    float v1;
    float v2=scalar;
    float res;
    float (*mathptr)(float, float);
    if (divide_2nd)
        mathptr = &_div_2;
    else
        mathptr = &_mul_2;
    for(size_t i=0; i<elems; i++){
        v1 = arr->arr[i];
        res = mathptr(v1, v2);
        result->arr[i] = res;

    };
    return result;
};
Array* sc_times_arr(float scalar, Array* arr, bool divide_2nd){
    size_t rows = arr->rows;
    size_t cols =arr->cols;
    Array* result = zeros(arr->rows, arr->cols);
    size_t elems = arr->elems;
    float v2;
    float v1=scalar;
    float res;
    float (*mathptr)(float, float);
    if (divide_2nd)
        mathptr = &_div_2;
    else
        mathptr = &_mul_2;
    for(size_t i=0; i<elems; i++){
        v2 = arr->arr[i];
        res = mathptr(v1, v2);
        result->arr[i] = res;

    };
    return result;
};
void set_row_to_row(Array* to_change, Array* set_with, size_t starting_at, size_t index_skip){

    for(size_t i=0;i<index_skip;i++){
        to_change->arr[starting_at+i] = set_with->arr[i];

    };
    return;
};
Array* udiag_left(Array* upper_diag, Array* arr){
    size_t rows = upper_diag->rows;
    size_t cols = upper_diag->cols;
    size_t elems = upper_diag->elems;
    Array* result = zeros(rows, cols);
    float aij;
    //float pval;
    Array* brow;
    Array* crow;
    Array* rowres;
    Array* rowsum;
    size_t index;
    size_t row_index;
    size_t index_skip = cols;
    for(size_t i=0; i<rows; i++){
        row_index = i*cols;
        for(size_t j=i; j<cols; j++){
            index = i*cols + j;
            //pval = result->arr[index];
            aij = upper_diag->arr[index];
            brow = get_row(arr, j, false);
            crow = get_row(result, i, false);
            rowres = arr_times_sc(brow, aij, false);
            rowsum = arr_pm_arr(crow, rowres, false);
            //pArr(rowsum, false);
            set_row_to_row(result, rowsum, row_index, index_skip);


        };
    };
    return result;
};
Array* udiag_right(Array* arr, Array* upper_diag){
    size_t rows = upper_diag->rows;
    size_t cols = upper_diag->cols;
    size_t elems = upper_diag->elems;
    Array* result = zeros(rows, cols);
    float bij;
    Array* ccol;
    Array* acol;

    for(size_t i=0; i<rows; i++){
        
        for(size_t j=i; j<cols; j++){

        };
    };
    return result;
};
void set_col_to_col(Array* to_change, Array* set_with, size_t starting_at, size_t col_height, size_t col_skip){
    for(size_t i=0;i<col_height;i++){
        to_change->arr[starting_at+i*col_skip] = set_with->arr[i];

    };
    return;
};
//https://github.com/numpy/numpy/blob/main/numpy/_core/src/multiarray/arrayobject.c

