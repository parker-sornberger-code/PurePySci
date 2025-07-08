#ifndef CSTRIDEARRAY_H
# define CSTRIDEARRAY_H
typedef struct Array{
    float* arr;
    size_t rows;
    size_t cols;
    size_t elems;
    bool transposed;


}Array;
Array* zeros(size_t rows, size_t cols);
Array* fill(size_t rows, size_t cols, float fill_with);
Array* zeros(size_t rows, size_t cols);
Array* ones(size_t rows, size_t cols);
Array* arange(size_t maxsize);
Array* shape_arange(size_t maxsize, size_t rows, size_t cols);
Array* copy_transpose(Array* Arr);
int shallow_transpose(Array* Arr);
int reshape(Array* arr, size_t new_rows, size_t new_cols) ;
Array* slice(const Array* arr, size_t row_start, size_t row_end, size_t col_start, size_t col_end) ;
void pArr(Array* Arr, bool do_free);
void FreeArr(Array* Arr);
Array* get_row(Array* arr, size_t which_row, bool transposed);
Array* get_col(Array* arr, size_t which_col, bool transposed);
float dot(Array* row, Array* col);
Array* matmul(Array* arr1, Array* arr2);
Array* eye(size_t dims);
Array* get_diag(Array* arr);
Array* make_upper_diag(Array* arr);
void make_upper_diag_inplace(Array* arr);
Array* full_copy(Array* arr);
//Array* arr_pm_arr(Array* arr1, Array* arr2, bool negate_2nd);
float _add_2(float v1, float v2);
float _sub_2(float v1, float v2);
float _mul_2(float v1, float v2);
float _div_2(float v1, float v2);
Array* arr_pm_arr(Array* arr1, Array* arr2, bool negate_2nd);
Array* sc_pm_arr(float scalar, Array* arr, bool negate_2nd);
Array* arr_pm_sc(Array* arr, float scalar, bool negate_2nd);

Array* udiag_left(Array* upper_diag, Array* arr);
Array* udiag_right(Array* arr, Array* upper_diag);

Array* sc_times_arr(float scalar,Array* arr,  bool divide_2nd);
Array* arr_times_arr(Array* arr1, Array* arr2, bool divide_2nd);
Array* arr_times_sc(Array* arr, float scalar, bool divide_2nd);

void set_row_to_row(Array* to_change, Array* set_with, size_t starting_at, size_t index_skip);
void set_col_to_col(Array* to_change, Array* set_with, size_t starting_at, size_t col_height, size_t col_skip );


#endif
