#ifndef RANDOM_WALKS_H
#define RANDOM_WALKS_H

#include <map>
#include <string>
#include <random>

using parameterSet = std::map<std::string, double>;
using doublePair =  std::pair<double, double>;

bool check_GBM_params(const parameterSet&);
std::vector<double> simulate_GBM(const parameterSet&, int, std::mt19937_64&);
doublePair stats_GBM(const parameterSet&);

bool check_VG_params(const parameterSet&);
std::vector<double> simulate_VG(const parameterSet&, int, std::mt19937_64&);
doublePair stats_VG(const parameterSet&);

#endif
