#include <iostream>
#include "arg_parser.h"

std::vector<Point> points;

int main(int argc, char **argv) {

    // Parse CLI args, read input file, and set random seed
    parse_args(argc, argv);
    // print_args();

    read_points(points);
    // print_points(points);

    srand(seed);

    

    return 0;
}