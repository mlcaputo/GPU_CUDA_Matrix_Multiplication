# GPU_CUDA_Matrix_Multiplication

Implementation of a general matrix multiplication (gemm) subroutine using the function in cuBLAS library (SGEMM). The matrix file contains two 32-bit unsigned integers 
representing the matrix size, followed by the matrix data as 32-bit floating point values in column-major order. Accepts arbitrart matrix sizes and assumes that both matrices 
will fit in global memory. 
