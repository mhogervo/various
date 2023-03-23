#include "random_walks.h"
#include <set>
#include <iostream>

// ---------------------------------
// A class for MC runs; see random_walks.h for more info.
// ---------------------------------

MCrun::MCrun(std::string modelName, const parameterSet& modelParameters, int timesteps, int num_runs, rngClass& rng) :
    modelName(modelName), modelParameters(modelParameters), timesteps(timesteps), num_runs(num_runs), rng(rng) {
    
    simulations.reserve(num_runs);
    check_parameter_set(modelName, modelParameters);
    
    if (modelName == "GBM") {
        for (int i=0; i<num_runs; i++) {
            auto sim = simulate_GBM(modelParameters, timesteps, rng);
            simulations.push_back(sim);
        }
    }
    else if (modelName == "VG") {
        for (int i=0; i<num_runs; i++) {
            auto sim = simulate_VG(modelParameters, timesteps, rng);
            simulations.push_back(sim);
        }
    }
    else {
        throw std::runtime_error("Error: model name isn't recognized.");
    }
}

void MCrun::printParameters() {
    // print the specifications of this run
    std::cout << "Model parameters for this Monte Carlo run (of " << modelName << ") are as follows:\n";
    for ( const auto &p : modelParameters ) {
        std::cout << p.first << ":\t" << p.second << std::endl;
    }
    std::cout << "Doing " << num_runs << " runs, with " << timesteps << " timesteps each.\n" << std::endl;
}

std::vector<double> MCrun::fetchClosingPrices() {
    // Return a vector with all closing prices.
    std::vector<double> closingPrices;
    closingPrices.reserve(num_runs);
    
    for (int i=0; i<num_runs; i++) {
        closingPrices.push_back(simulations[i].back());
    }
    return closingPrices;
}


bool check_parameter_set(std::string modelName, const parameterSet& p) {
    /* Take a parameterSet and check if it contains all parameters for the GBM or VG processes. */
    
    std::set<std::string> tokens;
    if (modelName == "GBM") {
        tokens = {"mu", "sigma", "S0", "T"};
    }
    else if (modelName == "VG") {
        tokens = {"theta", "sigma", "nu", "S0", "T"};
    }
    else {
        throw std::runtime_error("Error: model name isn't recognized.");
    }
    
    for (const auto& s : tokens) {
        if (p.count(s) == 0) return false;
    }
    return true;
}

// ---------------------------------
// Functions for Geometric Brownian Motion.
// ---------------------------------


doublePair stats_GBM(const parameterSet& params) {
    /* Given a set of parameters for the GBM, compute the expected mean and standard deviation,
       returned as a pair.
    */
    double mu, sigma, S0, T;
    if (check_parameter_set("GBM", params)) {
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

std::vector<double> simulate_GBM(const parameterSet& params, int N, rngClass& rng) {
    /** Simulate a geometric Brownian motion. The map params should contain {mu, sigma, T, S0}.
        N is the number of steps.

        Output: a vector of N+1 doubles v[0] = S_0, ..., v[N] = S_T.
    */
    
    std::vector<double> samples;
    samples.reserve(N+1);
    
    double mu, sigma, S, T, dt;
    if (check_parameter_set("GBM", params)) {
        sigma = params.at("sigma");
        mu = params.at("mu");
        S = params.at("S0");
        T = params.at("T");
        dt = T/N;
    }
    else {
        throw std::runtime_error("Error: parameter set for the GBM process is not complete.");
    }
    
    double mu_bar = (mu - 0.5*pow(sigma,2))*dt, sigma_bar = sigma*sqrt(dt);
    std::normal_distribution Wiener(mu_bar, sigma_bar);
    
    samples[0] = S;
    for (int i=1; i<=N; i++) {
        S *= exp(Wiener(rng));
        samples.push_back(S);
    }
    
    return samples;
}

// ---------------------------------
// Functions for the Variance-Gamma process.
// ---------------------------------

doublePair stats_VG(const parameterSet& params) {
    /* Given a set of parameters for the VG process, compute the expected mean and standard deviation,
       returned as a pair.
    */
    double theta, sigma, nu, S0, T;
    if (check_parameter_set("VG", params)) {
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


std::vector<double> simulate_VG(const parameterSet& params, int N, rngClass& rng) {
    /** Simulate a variance-gamma process with parameters {theta, sigma, nu}.
        The map params should also contain S0 and the total time T.
        N is the number of steps, such that dt = T/N.

        Output: a vector of N+1 doubles v[0] = S_0, ..., v[N] = S_T.
    */
    
    std::vector<double> samples;
    samples.reserve(N+1);
    
    double theta, sigma, nu, S, T, dt;
    if (check_parameter_set("VG", params)) {
        theta = params.at("theta");
        sigma = params.at("sigma");
        nu = params.at("nu");
        S = params.at("S0");
        T = params.at("T");
        dt = T/N;
    }
    else {
        throw std::runtime_error("Error: parameter set for the variance-gamma process is not complete.");
    }
    
    std::gamma_distribution gamma_distr(dt/nu, nu);
    std::normal_distribution Wiener;

    samples[0] = S;
    for (int i=1; i<=N; i++) {
        double dG = gamma_distr(rng), Z = Wiener(rng);
        S += theta*dG + sigma*sqrt(dG)*Z;
        samples.push_back(S);
    }
    
    return samples;
}
