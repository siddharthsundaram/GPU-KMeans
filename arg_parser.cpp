#include "arg_parser.h"

int num_clusters;
int dims;
std::string input_file;
int max_iter;
double thresh;
bool output_centroids;
int seed;
bool use_gpu;
bool use_shared_mem;
bool use_kpp;

void parse_args(int argc, char **argv) {
    
    bpo::options_description od("Options");
    od.add_options()
        ("-k", bpo::value<int>()->required(), "Number of clusters")
        ("-d", bpo::value<int>()->required(), "Dimension of the points")
        ("-i", bpo::value<std::string>()->required(), "Input filename")
        ("-m", bpo::value<int>()->default_value(100), "Maximum number of iterations")
        ("-t", bpo::value<double>()->default_value(1e-4), "Threshold for convergence test")
        ("-c", "Output cluster centroids if true, labels of all points otherwise")
        ("-s", bpo::value<int>()->default_value(8675309), "Random generator seed")
        ("-g", "Enable GPU implementation if true, don't otherwise")
        ("-f", "Enable shared memory GPU implementation if true, don'f otherwise")
        ("-p", "Enable Kmeans++ implementation if true, don't otherwise");

    bpo::variables_map var_map;
    try {
        bpo::store(bpo::parse_command_line(argc, argv, od), var_map);
        bpo::notify(var_map);
    } catch (const bpo::error &e) {
        std::cout << "Error: " << e.what() << "\n" << std::endl;
        return;
    }

    if (var_map.empty()) {
        std::cout << "No options were parsed!" << std::endl;
    }

    num_clusters = var_map["-k"].as<int>();
    dims = var_map["-d"].as<int>();
    input_file = var_map["-i"].as<std::string>();
    max_iter = var_map["-m"].as<int>();
    thresh = var_map["-t"].as<double>();
    output_centroids = var_map.count("-c") > 0;
    seed = var_map["-s"].as<int>();
    use_gpu = var_map.count("-g") > 0;
    use_shared_mem = var_map.count("-f") > 0;
    use_kpp = var_map.count("-p") > 0;
}

void print_args() {
    std::cout << "Number of Clusters: " << num_clusters << "\n";
    std::cout << "Dimensions: " << dims << "\n";
    std::cout << "Input File: " << input_file << "\n";
    std::cout << "Max Iterations: " << max_iter << "\n";
    std::cout << "Threshold: " << thresh << "\n";
    std::cout << "Output Mode: " << (output_centroids ? "Centroids" : "Labels") << "\n";
    std::cout << "Random Seed: " << seed << "\n";
    std::cout << "GPU Implementation: " << (use_gpu ? "Enabled" : "Disabled") << "\n";
    std::cout << "Shared-Memory GPU: " << (use_shared_mem ? "Enabled" : "Disabled") << "\n";
    std::cout << "KMeans++: " << (use_kpp ? "Enabled" : "Disabled") << "\n";
}

Point load_point(std::string line, std::string delimiter) {
    int start = 0;
    int end = 0;
    std::vector<float> pos;
    int label;

    for (int i = 0; i <= dims; ++i) {
        end = line.find(delimiter, start);
        std::string num = line.substr(start, end - start);

        if (i == 0) {
            label = std::stoi(num);
        } else {
            pos.push_back(std::stof(num));
        }

        start = end + delimiter.length();
    }

    pos.push_back(std::stof(line.substr(start)));
    Point p = {label, pos};
    return p;
}

void read_points(std::vector<Point> &points) {
    std::ifstream in(input_file);

    if (!in.is_open()) {
        std::cout << "Error: File could not be opened. Path: " << input_file << std::endl;
        return;
    }

    std::string line;
    getline(in, line);
    int num_points = std::stoi(line);

    for (int i = 0; i < num_points; ++i) {
        getline(in, line);
        Point p = load_point(line, " ");
        points.push_back(p);
    }

    in.close();
}

void print_points(std::vector<Point> &points) {
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "Point " << points[i].label << ", Position: [";

        for (size_t j = 0; j < dims; ++j) {
            std::cout << points[i].pos[j];
            if (j < dims - 1) {
                std::cout << ", ";
            }
        }

        std::cout << "]\n";
    }
}