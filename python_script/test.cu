#include <iostream>

int main() {
  int iter = 3;
  int K_PIPE_MAX = 60;
  int PatternLen = 4;
  printf("%d\n", (iter - K_PIPE_MAX) % PatternLen);
  return 0;
}