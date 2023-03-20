#include <random>
#include <cstdint>
#include <iostream>
#include <math.h>
#include <string>
#include <map>
#include <stdlib.h> 

double gamma_moments(int, double, double);
std::pair<double, double> stats_from_sample(const std::vector<double>&);

// typedef std::tuple<double, double, double> VGparams;
typedef std::map<std::string, double> parameterSet;
// std::vector<double> simulate_gamma_process(parameterSet, double, int);
std::vector<double> simulate_GBM(parameterSet, int, std::mt19937_64&);

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
    
    // Sample from the gamma distribution:
    
    std::gamma_distribution gamma_distr(alpha, beta);
    std::normal_distribution normal_distr;
    
    std::vector<double> sample_list(num_samples);
    for (int i=0; i<num_samples; i++) sample_list[i] = gamma_distr(rng);

    auto sample_stats = stats_from_sample(sample_list);
    double mu_est(sample_stats.first), sigma_est(sample_stats.second);
    
    // theoretical mu and sigma:
    double mu, sigma;
    mu = gamma_moments(1, alpha, beta); // or: alpha * beta
    sigma = sqrt(gamma_moments(2, alpha, beta) - pow(mu, 2)); // or: sqrt(alpha) * beta

    std::cout << "\nComparing sample mean and standard deviation (gamma):" << std::endl;
    std::cout << mu << "\t" << mu_est << "\n" << sigma << "\t" << sigma_est << std::endl;

    

    // Simulate num_samples different geometric Brownian motions with specified parameters.

    num_samples = 1e4;
    parameterSet brownianParams = {{"sigma", 0.3}, {"mu", 1}, {"S0", 100}, {"T", 0.2}};
    int n_steps = 100; // steps for every single walk; total calls to the RNG is num_samples * n_steps.
    
    sample_list = {};
    for (int i=0; i<num_samples; i++) {
        auto sim = simulate_GBM(brownianParams, n_steps, rng);
        sample_list.push_back(sim.back());
    }
    // extract statistics and compare to analytics:
    sample_stats = stats_from_sample(sample_list);
    mu_est = sample_stats.first, sigma_est = sample_stats.second;
    
    double S0 = brownianParams["S0"], T = brownianParams["T"];
    mu = brownianParams["mu"], sigma = brownianParams["sigma"];
    double exp_S_T = S0*exp(mu*T);
    double std_S_T = exp_S_T * sqrt( exp(pow(sigma,2)*T) - 1 );
    
    std::cout << "Comparing sample mean and standard deviation (GBM):" << std::endl;
    std::cout << exp_S_T << "\t" << mu_est << "\n" << std_S_T << "\t" << sigma_est << "\n" << std::endl;
    
    
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

std::vector<double> simulate_GBM(parameterSet params, int N, std::mt19937_64& rng) {
    /** Simulate a geometric Brownian motion. The map params should contain {mu, sigma, T, S0}.
        N is the number of steps.

        Output: a vector of N+1 doubles v[0] = S_0, ..., v[N] = S_T.
    */
    
    std::vector<double> samples(N+1);
    double mu, sigma, S0, T, dt;
    if (params.contains("mu") and params.contains("sigma") and
        params.contains("S0") and params.contains("T")) {
        mu = params["mu"], sigma = params["sigma"];
        S0 = params["S0"], T = params["T"];
        dt = T/N;
    }
    else {
        std::cerr << "Error: parameter set for Brownian motion is not complete." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    double mu_bar = (mu - 0.5*pow(sigma,2))*dt, sigma_bar = sigma*sqrt(dt);
    std::normal_distribution Wiener(mu_bar, sigma_bar);
    
    double S = S0;
    samples[0] = S;
    for (int i=1; i<=N; i++) {
        S *= exp(Wiener(rng));
        samples[i] = S;
    }
    
    return samples;
}

