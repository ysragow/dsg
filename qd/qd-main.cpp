#include <iostream>
#include <cmath>
#include <ctime>

using namespace std;

int main() {
    return 0
}

class Predicate {
    /*
    Class containing the predicate for a node
    - dimension: the number of dimensions in the hypercube
    - upper: the upper bounds for each dimension
    - lower: the lower bounds for each dimension
    */
    private:
        int dimension;
        int *upper[dimension];
        int *lower[dimension];
        bool generated;
    public:
        Predicate(int n) {
            dimension = n;
            generated = false;
        }
        void generate(int upper_in[dimension], int lower_in[dimension]) {
            upper = upper_in;
            lower = lower_in;
            generated = True;
        }
}

class Node {
    private:
        Node *left;
        Node *right;
        bool leaf;
    public:




}