#include <iostream>
using namespace std;
enum class Mat {
    A,
    B
  };

int main() {
  int a[4] = {3,4,5,6};
  cout << a[int(Mat::A)] << a[int(Mat::B)] << endl;
  return 0;
}