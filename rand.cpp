#include <random>
#include <cstdint>
#include <iostream>
#include <math.h>

double gamma_moments(int, double, double);
std::pair<double, double> stats_from_sample(const std::vector<double> &);

int main()
{
    
    int my_seed = 2000;
    double alpha(2.4), beta(1.37);
    int num_samples = 1e6;

    std::seed_seq seed_seq{my_seed};
    // std::ranlux is 'better' but slower. std::ranlux48 is *really* slow.
    // std::ranlux24 rng(seed_seq);
    // Mersenne is faster and still acceptable. The _64 version is slightly better.
    // std::mt19937 rng(seed_seq);
    std::mt19937_64 rng(seed_seq);
    
    // Don't change anything below this line.
    
    std::gamma_distribution gamma_distr(alpha, beta);
    
    std::vector<double> sample_list(num_samples);
    // fill the array with samples:
    for (int i=0; i<num_samples; i++) sample_list[i] = gamma_distr(rng);

    auto sample_stats = stats_from_sample(sample_list);
    double mu_est(sample_stats.first), sigma_est(sample_stats.second);
    
    // theoretical mu and sigma:
    double mu, sigma;
    mu = gamma_moments(1, alpha, beta); // or: alpha * beta
    sigma = sqrt(gamma_moments(2, alpha, beta) - pow(mu, 2)); // or: sqrt(alpha) * beta

    std::cout << "\nComparing sample mean and standard deviation:" << std::endl;
    std::cout << mu << "\t" << mu_est << "\n" << sigma << "\t" << sigma_est << "\n" << std::endl;
    
}


std::pair<double, double> stats_from_sample(const std::vector<double> & v) {
    /**
       Given a vector v of floats, compute the mean and standard deviation.
       
       Input: a std::vector<> of doubles, v[0], v[1], ..., v[N-1].
       
       Output: an std::pair containing the mean and standard deviation.
       To extract, use e.g.
       
           out = stats_from_sample(my_samples);
           double mu = out.first, sigma = out.second;
        
     */
    
    int n = v.size();
    double av(0), std(0);
    
    for (int i=0; i<n; i++) av += v[i];
    av /= n;
    
    for (int i=0; i<n; i++) std += pow(v[i] - av, 2);
    std /= (n-1);
    std = sqrt(std);
    
    std::pair<double, double> out(av, std);
    return out;
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
