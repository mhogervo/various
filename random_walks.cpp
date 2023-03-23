#include "random_walks.h"
#include <set>

bool check_GBM_params(const
                      parameterSet& p) {
    /* Take a parameterSet and check if it contains all parameter for the GBM process. */
    std::set<std::string> tokens {"mu", "sigma", "S0", "T"};
    for (const auto& s : tokens) {
        if (p.count(s) == 0) return false;
    }
    return true;
}

doublePair stats_GBM(const parameterSet& params) {
    /* Given a set of parameters for the GBM, compute the expected mean and standard deviation,
       returned as a pair.
    */
    double mu, sigma, S0, T;
    if (check_GBM_params(params)) {
        sigma = params.at("sigma");
        mu = params.at("mu");
        S0 = params.at("S0");
        T = params.at("T");
    }
    else {
        throw std::runtime_error("Error: parameter set for the GBM process is not complete.");
    }

    double mean = S0*exp(mu*T);
    double std_dev = mean * sqrt( exp(pow(sigma,2)*T) - 1 );

    auto out = std::make_pair(mean, std_dev);
    return out;    
}

std::vector<double> simulate_GBM(const parameterSet& params, int N, std::mt19937_64& rng) {
    /** Simulate a geometric Brownian motion. The map params should contain {mu, sigma, T, S0}.
        N is the number of steps.

        Output: a vector of N+1 doubles v[0] = S_0, ..., v[N] = S_T.
    */
    
    std::vector<double> samples(N+1);
    double mu, sigma, S0, T, dt;
    if (check_GBM_params(params)) {
        sigma = params.at("sigma");
        mu = params.at("mu");
        S0 = params.at("S0");
        T = params.at("T");
        dt = T/N;
    }
    else {
        throw std::runtime_error("Error: parameter set for the GBM process is not complete.");
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

bool check_VG_params(const parameterSet& p) {
    /* Take a parameterSet and check if it contains all parameter for the VG process. */
    std::set<std::string> tokens {"theta", "sigma", "nu", "S0", "T"};
    for (const auto& s : tokens) {
        if (p.count(s) == 0) return false;
    }
    return true;
}

doublePair stats_VG(const parameterSet& params) {
    /* Given a set of parameters for the VG process, compute the expected mean and standard deviation,
       returned as a pair.
    */
    double theta, sigma, nu, S0, T;
    if (check_VG_params(params)) {
        theta = params.at("theta");
        sigma = params.at("sigma");
        nu = params.at("nu");
        S0 = params.at("S0");
        T = params.at("T");
    }
    else {
        throw std::runtime_error("Error: parameter set for the variance-gamma process is not complete.");
    }

    double mean = S0 + theta*T;
    double std_dev = sqrt((pow(theta,2)*nu + pow(sigma,2))*T);
    
    auto out = std::make_pair(mean, std_dev);
    return out;    
}


std::vector<double> simulate_VG(const parameterSet& params, int N, std::mt19937_64& rng) {
    /** Simulate a variance-gamma process with parameters {theta, sigma, nu}.
        The map params should also contain S0 and the total time T.
        N is the number of steps, such that dt = T/N.

        Output: a vector of N+1 doubles v[0] = S_0, ..., v[N] = S_T.
    */
    
    std::vector<double> samples(N+1);
    double theta, sigma, nu, S0, T, dt;
    if (check_VG_params(params)) {
        theta = params.at("theta");
        sigma = params.at("sigma");
        nu = params.at("nu");
        S0 = params.at("S0");
        T = params.at("T");
        dt = T/N;
    }
    else {
        throw std::runtime_error("Error: parameter set for the variance-gamma process is not complete.");
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
