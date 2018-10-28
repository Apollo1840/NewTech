#include <iostream>
#include <math.h>

void add(int n, float *x, float *y)
{
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elements
  float *x, *y;
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  add(N, x, y);
  
  delete [] x;
  delete [] y;
  
  return 0;
}