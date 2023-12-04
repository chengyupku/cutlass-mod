#include <iostream>
#include <cstddef>
#include <climits>

int main() {
    std::cout << "Size of size_t: " << sizeof(size_t) << " bytes\n";
    std::cout << "Maximum value of size_t: " << SIZE_MAX << "\n";
    return 0;
}