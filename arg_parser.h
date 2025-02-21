#ifndef ARG_PARSER_H
#define ARG_PARSER_H

#include <boost/program_options.hpp>
#include <string>
#include <iostream>

namespace bpo = boost::program_options;

extern int num_clusters;
extern int dims;
extern std::string input_file;
extern int max_iter;
extern double thresh;
extern bool output_centroids;
extern int seed;
extern bool use_gpu;
extern bool use_shared_mem;
extern bool use_kpp;

void parse_args(int argc, char **argv);
void print_args();

#endif