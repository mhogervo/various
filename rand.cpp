#include <iostream>
#include <math.h>
#include <string>
#include "random_walks.h"

// using doublePair = std::pair<double, double> defined in random_walks.h

doublePair stats_from_sample(const std::vector<double>&);
void print_stats(const doublePair, const doublePair);

// not in use currently:
double gamma_moments(int, double, double);

int main()
{
    int my_seed = 8732;
    int num_samples = 2e4;
    int n_steps = 100; // steps for every single walk; total calls to the RNG is O(1) * num_samples * n_steps
    // where O(1) is the number of calls at every timestep.

    // initiate the RNG with a fixed seed (set above)
    std::seed_seq seed_seq {my_seed};
    std::mt19937_64 rng(seed_seq);
    // std::ranlux is 'better' but slower. std::ranlux48 is *really* slow.
    // std::ranlux24 rng(seed_seq);
    // std::ranlux48 rng(seed_seq);
    // Mersenne is faster and still acceptable. The _64 version is slightly better.
    // std::mt19937 rng(seed_seq);

    // I: Simulate num_samples different geometric Brownian motions with specified parameters.
   
    parameterSet brownianParams {{"sigma", 0.3}, {"mu", 1}, {"S0", 100}, {"T", 0.7}};
    std::vector<double> sample_list = {};
    for (int i=0; i<num_samples; i++) {
        auto sim = simulate_GBM(brownianParams, n_steps, rng); 
        sample_list.push_back(sim.back()); // record only the last value, S_T
    }
    // extract statistics and compare to analytics:
    doublePair sample_stats = stats_from_sample(sample_list);
    doublePair stats_th = stats_GBM(brownianParams);
    
    std::cout << "\nComparing sample mean and standard deviation (GBM), over " << num_samples << " realizations:" << std::endl;
    print_stats(stats_th, sample_stats);

    // -------------------------------------------
    
    // II: Simulate num_samples different VG processes with specified parameters.
    
    parameterSet VGParams {{"theta", -1.2}, {"sigma", 2.4}, {"nu", 0.7}, {"S0", 30.}, {"T", 0.4}};
    sample_list = {};
    for (int i=0; i<num_samples; i++) {
        auto sim = simulate_VG(VGParams, n_steps, rng);
        sample_list.push_back(sim.back());
    }
    sample_stats = stats_from_sample(sample_list);
    stats_th = stats_VG(VGParams);
    
    std::cout << "Comparing sample mean and standard deviation (VG), over " << num_samples << " realizations:" << std::endl;
    print_stats(stats_th, sample_stats);
    
}


doublePair stats_from_sample(const std::vector<double> & v) {
    /**
       Given a vector v of floats, compute the mean and standard deviation.
       
       Input: a std::vector<> of doubles, v[0], v[1], ..., v[N-1].
       
       Output: an std::pair containing the mean and standard deviation.
       To extract, use e.g.
       
           out = stats_from_sample(my_samples);
           double mu = out.first, sigma = out.second;
        
     */
    
    int n = v.size();
    double av(0), std_dev(0);
    
    for (int i=0; i<n; i++) av += v[i];
    av /= n;
    
    for (int i=0; i<n; i++) std_dev += pow(v[i] - av, 2);
    std_dev /= (n-1);
    std_dev = sqrt(std_dev);
    
    auto out = std::make_pair(av, std_dev);
    return out;
}


void print_stats(const doublePair stats_th, const doublePair stats_sample) {
    /* Given two pairs of (mu, sigma) in theory and for a sample, print the comparison. */
    double mu_est = stats_sample.first, sigma_est = stats_sample.second;
    double mu_th = stats_th.first, sigma_th = stats_th.second;
    std::cout << "\ttheory\tsample" << std::endl;
    std::cout << "mu\t" << mu_th << "\t" << mu_est << "\nsigma\t" << sigma_th << "\t" << sigma_est << "\n" << std::endl;
}


double gamma_moments(int n, double alpha=1.0, double beta=1.0) {
    /**
       Return the n-th moment of the gamma distribution:
       
           (alpha)_n * beta^n
       
       where (x)_n = x(x+1)...(x+n-1) is the Pochhammer symbol.
    */
    
    double out = 1;
    for (int i=0; i<n; i++) out *= alpha + i;
    out *= pow(beta,n);
    return out;
}
