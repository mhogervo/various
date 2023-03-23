#include <iostream>
#include "random_walks.h"

/* defined in random_walks.h:
   
using rngClass = std::mt19937_64;
using parameterSet = std::map<std::string, double>;
using doublePair =  std::pair<double, double>;

*/

doublePair stats_from_sample(const std::vector<double>&);
void print_stats(const doublePair, const doublePair);

int main()
{
    int my_seed = 2023;
    int num_samples = 2e4;
    int n_steps = 100;      // steps for every single walk; total calls to the RNG is O(1) * num_samples * n_steps
                            // where O(1) is the number of calls at every timestep.

    // initiate the RNG with a fixed seed (set above)
    std::seed_seq seed_seq {my_seed};
    rngClass rng(seed_seq);

    
    // I: Simulate num_samples different geometric Brownian motions with specified parameters.
    
    parameterSet brownianParams {{"sigma", 0.3}, {"mu", 1}, {"S0", 100}, {"T", 0.7}};
    MCrun GBM_run("GBM", brownianParams, n_steps, num_samples, rng);
    GBM_run.printParameters();
    std::vector<double> sample_list = GBM_run.fetchClosingPrices();
   
    // extract statistics and compare to analytics:
    doublePair sample_stats = stats_from_sample(sample_list);
    doublePair stats_th = stats_GBM(brownianParams);
    
    std::cout << "Comparing sample mean and standard deviation (GBM):" << std::endl;
    print_stats(stats_th, sample_stats);

    std::cout << "-------------\n" << std::endl;
    
    // II: Simulate num_samples different VG processes with specified parameters.
    
    parameterSet VGParams {{"theta", -1.2}, {"sigma", 2.4}, {"nu", 0.7}, {"S0", 30.}, {"T", 8}};
    MCrun VG_run("VG", VGParams, n_steps, num_samples, rng);
    VG_run.printParameters();
    sample_list = VG_run.fetchClosingPrices();
    
    sample_stats = stats_from_sample(sample_list);
    stats_th = stats_VG(VGParams);
    
    std::cout << "Comparing sample mean and standard deviation (VG):" << std::endl;
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
    std::cout << "\t\t(theory, sample)" << std::endl;
    std::cout << "mu\t(" << mu_th << ", " << mu_est << ")" << std::endl;
    std::cout << "sigma\t(" << sigma_th << ", " << sigma_est << ").\n" << std::endl;
}
