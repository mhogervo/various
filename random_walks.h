#ifndef RANDOM_WALKS_H
#define RANDOM_WALKS_H

#include <map>
#include <random>

/*
Lüscher's ranlux is the gold standard, but it's slower. std::ranlux48 is *really* slow.
// std::ranlux24 rng(seed_seq);
// std::ranlux48 rng(seed_seq);
Mersenne is faster and still acceptable. The _64 version is slightly better.
// std::mt19937 rng(seed_seq);
*/

using rngClass = std::mt19937_64;
using parameterSet = std::map<std::string, double>;
using doublePair =  std::pair<double, double>;

bool check_GBM_params(const parameterSet&);
std::vector<double> simulate_GBM(const parameterSet&, int, rngClass&);
doublePair stats_GBM(const parameterSet&);

bool check_VG_params(const parameterSet&);
std::vector<double> simulate_VG(const parameterSet&, int, rngClass&);
doublePair stats_VG(const parameterSet&);

class MCrun {
public:
    MCrun(std::string, const parameterSet&, int, int, rngClass&);

    std::vector<double> fetchClosingPrices();

    std::vector<double> fetchRun(int j) {
        // Return the j-th run; note that we must have 0<j<num_runs;
        return simulations[j];
    }
    
private:
    std::vector<std::vector<double>> simulations;

    std::string modelName;
    parameterSet modelParameters;
    int timesteps, num_runs;
    rngClass rng;
};


#endif
