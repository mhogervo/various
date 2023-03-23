#ifndef RANDOM_WALKS_H
#define RANDOM_WALKS_H

#include <random>
#include <map>

using rngClass = std::mt19937_64;

// Lüscher's ranlux is the gold standard, but it's slow. std::ranlux48 is *really* slow.
//    std::ranlux24;
//    std::ranlux48;

// Mersenne is faster and still acceptable. The _64 version is slightly better.
//    std::mt19937;
//    std::mt19937_64;


using parameterSet = std::map<std::string, double>;
using doublePair =  std::pair<double, double>;

std::vector<double> simulate_GBM(const parameterSet&, int, rngClass&);
doublePair stats_GBM(const parameterSet&);

std::vector<double> simulate_VG(const parameterSet&, int, rngClass&);
doublePair stats_VG(const parameterSet&);

bool check_parameter_set(std::string, const parameterSet&);

class MCrun {
    /*
      This is a class that that a set of parameters and performs multiple Monte Carlo simulations at once.
      It can return individual runs or the collection of all closing prices.
    */
public:
    MCrun(std::string, const parameterSet&, int, int, rngClass&);

    std::vector<double> fetchClosingPrices();

    std::vector<double> fetchRun(int j) {
        // Return the j-th run; note that we must have 0<j<num_runs;
        return simulations[j];
    }

    void printParameters();
    
    std::vector<std::vector<double>> simulations; // stores the samples
    std::string modelName;
    parameterSet modelParameters;
    int timesteps, num_runs;
private:
    rngClass rng; // contains a pre-defined RNG, passed by reference.
};


#endif
