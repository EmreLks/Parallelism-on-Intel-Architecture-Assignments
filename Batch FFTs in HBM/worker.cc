#include <mkl.h>
#include<hbwmalloc.h>

//implement scratch buffer on HBM and compute FFTs, refer instructions on Lab page
void runFFTs( const size_t fft_size,
              const size_t num_fft,
			  MKL_Complex8 *data, 
			  DFTI_DESCRIPTOR_HANDLE *fftHandle)
{
  MKL_Complex8 *buff;
  const long size = fft_size;

  hbw_posix_memalign((void**) &buff, 4096, sizeof(MKL_Complex8)*size);
  
  for(size_t i = 0; i < num_fft; i++)
  {
	#pragma omp parallel for
    for(long j = 0; j < size; j++)
	{
      buff[j].real = data[(i * size) + j].real;
      buff[j].imag = data[(i * size) + j].imag;
    }
    // End of the for loop.
	
    DftiComputeForward (*fftHandle, buff);
	
	#pragma omp parallel for
    for(long j = 0; j < size; j++)
	{
      data[(i * size) + j].real = buff[j].real;
      data[(i * size) + j].imag = buff[j].imag;
    }
    // End of the for loop.
	
  }
  
  hbw_free(buff);
}