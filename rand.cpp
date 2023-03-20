#include <random>
#include <cstdint>
#include <iostream>
#include <math.h>
#include <string>
#include <map>
#include <stdlib.h> 

// double gamma_moments(int, double, double);

typedef std::map<std::string, double> parameterSet;
typedef std::pair<double, double> doublePair;

std::vector<double> simulate_GBM(parameterSet, int, std::mt19937_64&);
doublePair stats_GBM(parameterSet);

std::vector<double> simulate_VG(parameterSet, int, std::mt19937_64&);
doublePair stats_VG(parameterSet);

doublePair stats_from_sample(const std::vector<double>&);
void print_stats(const doublePair, const doublePair);

int main()
{
    int my_seed = 2023;
    int num_samples = 2e4;
    int n_steps = 150; // steps for every single walk; total calls to the RNG is num_samples * n_steps.

    std::seed_seq seed_seq{my_seed};
    std::mt19937_64 rng(seed_seq);
    // std::ranlux is 'better' but slower. std::ranlux48 is *really* slow.
    // std::ranlux24 rng(seed_seq);
    // Mersenne is faster and still acceptable. The _64 version is slightly better.
    // std::mt19937 rng(seed_seq);

    // I: Simulate num_samples different geometric Brownian motions with specified parameters.
   
    parameterSet brownianParams = {{"sigma", 0.3}, {"mu", 1}, {"S0", 100}, {"T", 0.7}};
    std::vector<double> sample_list = {};
    for (int i=0; i<num_samples; i++) {
        auto sim = simulate_GBM(brownianParams, n_steps, rng);
        sample_list.push_back(sim.back());
    }
    // extract statistics and compare to analytics:
    doublePair sample_stats = stats_from_sample(sample_list);
    doublePair stats_th = stats_GBM(brownianParams);
    
    std::cout << "\nComparing sample mean and standard deviation (GBM), over " << num_samples << " realizations:" << std::endl;
    print_stats(stats_th, sample_stats);

    // -------------------------------------------
    
    // II: Simulate num_samples different VG processes with specified parameters.
    
    parameterSet VGParams = {{"theta", -1.2}, {"sigma", 2.4}, {"nu", 0.7}, {"S0", 30.}, {"T", 0.4}};
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
    double av(0), std(0);
    
    for (int i=0; i<n; i++) av += v[i];
    av /= n;
    
    for (int i=0; i<n; i++) std += pow(v[i] - av, 2);
    std /= (n-1);
    std = sqrt(std);
    
    std::pair<double, double> out(av, std);
    return out;
}



doublePair stats_GBM(parameterSet params) {
    /* Given a set of parameters for the GBM, compute the expected mean and standard deviation,
       returned as a pair.
    */
    double mu, sigma, S0, T;
    if (params.contains("mu") and params.contains("sigma") and
        params.contains("S0") and params.contains("T")) {
        mu = params["mu"], sigma = params["sigma"];
        S0 = params["S0"], T = params["T"];
    }
    else {
        std::cerr << "Error: parameter set for Brownian motion is not complete." << std::endl;
        exit(EXIT_FAILURE);
    }

    double mean = S0*exp(mu*T);
    double std = mean * sqrt( exp(pow(sigma,2)*T) - 1 );

    doublePair out(mean, std);
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

doublePair stats_VG(parameterSet params) {
    /* Given a set of parameters for the VG process, compute the expected mean and standard deviation,
       returned as a pair.
    */
    double theta, sigma, nu, S0, T;
    if (params.contains("theta") and params.contains("sigma") and params.contains("nu")
        and params.contains("S0") and params.contains("T")) {
        theta = params["theta"], sigma = params["sigma"], nu = params["nu"];
        S0 = params["S0"], T = params["T"];
    }
    else {
        std::cerr << "Error: parameter set for the variance-gamma process is not complete." << std::endl;
        exit(EXIT_FAILURE);
    }

    double mean = S0 + theta*T;
    double std = sqrt((pow(theta,2)*nu + pow(sigma,2))*T);
    
    doublePair out(mean, std);
    return out;    
}

std::vector<double> simulate_VG(parameterSet params, int N, std::mt19937_64& rng) {
    /** Simulate a variance-gamma process with parameters {theta, sigma, nu}.
        The map params should also contain S0 and the total time T.
        N is the number of steps, such that dt = T/N.

        Output: a vector of N+1 doubles v[0] = S_0, ..., v[N] = S_T.
    */
    
    std::vector<double> samples(N+1);
    double theta, sigma, nu, S0, T, dt;
    if (params.contains("theta") and params.contains("sigma") and params.contains("nu")
        and params.contains("S0") and params.contains("T")) {
        theta = params["theta"], sigma = params["sigma"], nu = params["nu"];
        S0 = params["S0"], T = params["T"];
        dt = T/N;
    }
    else {
        std::cerr << "Error: parameter set for the variance-gamma process is not complete." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::gamma_distribution gamma_distr(dt/nu, nu);
    std::normal_distribution Wiener;
    
    double S = S0;
    samples[0] = S;
    for (int i=1; i<=N; i++) {
        double dG = gamma_distr(rng), Z = Wiener(rng);
        S += theta*dG + sigma*sqrt(dG)*Z;
        samples[i] = S;
    }
    
    return samples;
}

void print_stats(const doublePair stats_th, const doublePair stats_sample) {
    /* Given two pairs of (mu, sigma) in theory and for a sample, print the comparison. */
    double mu_est = stats_sample.first, sigma_est = stats_sample.second;
    double mu_th = stats_th.first, sigma_th = stats_th.second;
    std::cout << mu_th << "\t" << mu_est << "\n" << sigma_th << "\t" << sigma_est << "\n" << std::endl;
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
