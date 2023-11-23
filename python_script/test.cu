#include <iostream>

// class B {
//   public:
//     int b = 0;
//     __device__ B(int b_) {
//       b = b_;
//       printf("init B!\n");
//     }
// };

// class A {
//   public:
//     int a = 0;
//     __device__ A(int a_) {a = a_;}
//     __device__ void func() {
//       B bi(33);
//     }
// };

// template <typename Operator>
// __global__
// void device_kernel()
// {
//   // Dynamic shared memory base pointer
//   extern __shared__ char smem[];

//   Operator op(99);
//   op.func();
// }


// int main() {
//   device_kernel<A><<<8, 32>>>();
//   cudaDeviceSynchronize();
//   return 0;
// }

class A {
  public:
    int a = 0;
    __device__ A(int a_) {
      a = a_;
      printf("init A!\n");
    }
};

__global__
void device_kernel()
{
  // Dynamic shared memory base pointer
  extern __shared__ char smem[];
  A ai(99);
}


int main() {
  device_kernel<<<8, 32>>>();
  cudaDeviceSynchronize();
  return 0;
}