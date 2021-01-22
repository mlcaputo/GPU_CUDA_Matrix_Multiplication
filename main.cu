#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))


int main (int argc, char *argv[])
{

  FILE *inA;
  FILE *inB;
  int bufferINT;
  float bufferFLOAT;

  int m; //6
  int n; //4
  int k; //5

  inA = fopen(argv[1], "r");
  inB = fopen(argv[2], "r");

  if (inA == NULL || inB == NULL) {
    printf("Couldn’t open file for reading. \n");
    return 0;
  }

  fread(&bufferINT, sizeof(unsigned int), 1, inA);
  int a_row = bufferINT;
  fread(&bufferINT, sizeof(unsigned int), 1, inA);
  int a_col = bufferINT;

  fread(&bufferINT, sizeof(unsigned int), 1, inB);
  int b_row = bufferINT;
  fread(&bufferINT, sizeof(unsigned int), 1, inB);
  int b_col = bufferINT;

  if(a_col != b_row)
  {
    printf("Matrices can't be multiplied");
    return 0;
  }
  else
  {
    m = a_row;
    n = b_col;
    k = a_col;
  }


  cudaError_t cudaStat;                                        // cudaMalloc status
  cublasStatus_t stat;                                         // CUBLAS functions status
  cublasHandle_t handle;                                       // CUBLAS context
  int i,j;                                                      // i-row index ,j- column index
  float * a;                                                    // mxk matrix a on the host
  float * b;                                                    // kxn matrix b on the host
  float * c;                                                    // mxn matrix c on the host
  a=(float*) malloc (m*k* sizeof(float));                  // host memory for a
  b=(float*) malloc (k*n* sizeof(float));                  // host memory for b
  c=(float*) malloc (m*n* sizeof(float));                  // host memory for c


  // define an mxk matrix a column by column
   // a:
  for(j=0;j<k;j++){
    for(i=0;i<m;i++){
      fread(&bufferFLOAT, sizeof(float), 1, inA);
      a[IDX2C(i,j,m)]=bufferFLOAT;
    }
  }


/*  // print a row by row
  printf ("a:\n");
  for (i=0;i<m;i++){
    for (j=0;j<k;j++){
      printf ("%5.0f",a[IDX2C(i,j,m )]);
    }
    printf ("\n");
  }*/


  // define a kxn matrix b column by column
  for(j=0;j<n;j++){
    for(i=0;i<k;i++){
      fread(&bufferFLOAT, sizeof(float), 1, inB);
      b[IDX2C(i,j,k)]=bufferFLOAT;
    }
  }


/*  // print b row by row
  printf ("b:\n");
  for (i=0;i<k;i++){
    for (j=0;j<n;j++){
      printf ("%5.0f",b[IDX2C(i,j,k )]);
    }
    printf ("\n");
  }*/


  // define an mxn matrix c column by column
  int ind =0;
  for(j=0;j<n;j++){
    for(i=0;i<m;i++){
      c[IDX2C(i,j,m)]=(float)ind;
    }
  }


/*  // print c row by row
  printf ("c:\n");
    for (i=0;i<m;i++){
      for (j=0;j<n;j++){
        printf ("%5.0f",c[IDX2C(i,j,m)]);
      }
      printf ("\n");
  }*/

  fclose(inA);
  fclose(inB);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  // on the device
  float* d_a; // d_a - a on the device
  float* d_b; // d_b - b on the device
  float* d_c; // d_c - c on the device

  cudaStat = cudaMalloc((void**)&d_a,m*k*sizeof(*a));
  // memory alloc for a

  cudaStat = cudaMalloc((void**)&d_b,k*n*sizeof(*b));
  // memory alloc for b

  cudaStat = cudaMalloc((void**)&d_c,m*n*sizeof(*c));
  // memory alloc for c

  stat = cublasCreate(&handle); // initialize CUBLAS context


  // copy matrices from the host to the device
  stat = cublasSetMatrix(m,k,sizeof(*a),a,m,d_a,m); //a -> d_a
  stat = cublasSetMatrix(k,n,sizeof(*b),b,k,d_b,k); //b -> d_b
  stat = cublasSetMatrix(m,n,sizeof(*c),c,m,d_c,m); //c -> d_c
  float al =1.0f; // al =1
  float bet =1.0f; // bet =1


  // matrix - matrix multiplication : d_c = al*d_a *d_b + bet *d_c
  // d_a -mxk matrix , d_b -kxn matrix , d_c -mxn matrix ;
  // al ,bet -scalars
  cudaEventRecord(start);

  stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,d_a,
  m,d_b,k,&bet,d_c,m);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float seconds = 0;
  float gflops = 0;
  float calc = n;
  cudaEventElapsedTime(&seconds, start, stop);
  gflops = 2 * pow(calc, 3);
  gflops = gflops / seconds;

  printf("Time: %.6f \n", seconds);
  printf("GFLOP/s: %.5f \n", gflops);



  stat = cublasGetMatrix (m,n,sizeof(*c),d_c ,m,c,m); // cp d_c - >c

  /*printf ("c after Sgemm :\n");
  for(i=0;i<m;i ++){
    for(j=0;j<n;j ++){
      printf ("%7.0f",c[ IDX2C (i,j,m )]); // print c after Sgemm
    }
    printf ("\n");
  }*/


  cudaFree (d_a); // free device memory
  cudaFree (d_b); // free device memory
  cudaFree (d_c); // free device memory

  cublasDestroy (handle); // destroy CUBLAS context

  free (a); // free host memory
  free (b); // free host memory
  free (c); // free host memory
  return EXIT_SUCCESS ;
}
