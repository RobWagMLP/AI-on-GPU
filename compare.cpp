#include <stdio.h>
#include <iostream>
using namespace std;
const size_t SIZE = 256*256*256;
int main() {
    
    double *a = new double[SIZE];
    double *b = new double[SIZE];
    double *c = new double[SIZE];
    for(size_t i = 0; i < SIZE; i++) {
        a[i] = 1;
        b[i] = 2;
    }
    for(int i = 0; i < 100; i++) {
        for(size_t j = 0; j < SIZE; j++) {
            c[j] = a[j] + b[j];
        }
    }
    int sum = 0;
    for(size_t k = 0; k < SIZE; k++) {
        sum += c[k];
    }
    cout << sum << "\n";
    delete[] a; delete[] b; delete[] c;
    return 0;
}