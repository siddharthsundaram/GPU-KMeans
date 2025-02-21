#include <iostream>
#include "arg_parser.h"

using namespace std;

int main(int argc, char **argv) {
    parse_args(argc, argv);
    print_args();

    return 0;
}