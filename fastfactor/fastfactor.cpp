#include <vector>
#include <omp.h>
#include <cmath>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using namespace std;
namespace py = pybind11;




vector<double> log_returns(const vector<double>& prices) {
    vector<double> returns;
    for (size_t i = 1; i < prices.size(); ++i) {
        returns.push_back(log(prices[i]/prices[i-1]));
    }
    return returns;
}   




std::vector<double> rolling_volatility(const std::vector<double>& log_returns, int window) {
    size_t n = log_returns.size();
    if (n < window) return {};

    std::vector<double> volatilities(n - window, 0.0);

    double sum = 0.0, sum_sq = 0.0;
    
    // Compute initial sum and sum of squares
    for (int i = 0; i < window; ++i) {
        sum += log_returns[i];
        sum_sq += log_returns[i] * log_returns[i];
    }

    #pragma omp parallel for
    for (size_t i = window; i < n; ++i) {
        // Compute mean and variance
        double mean = sum / window;
        double variance = (sum_sq / window) - (mean * mean);
        volatilities[i - window] = std::sqrt(variance);

        // Update rolling sum & sum_sq for next window
        sum += log_returns[i] - log_returns[i - window];
        sum_sq += log_returns[i] * log_returns[i] - log_returns[i - window] * log_returns[i - window];
    }

    return volatilities;
}






//Could assume the data is clean already for futher optimisation;
vector<double> momentum(const vector<double>& prices, int lookback) {
    size_t n = prices.size();
    if (n < lookback) return {};

    vector<double> result(n - lookback, 0.0);

    #pragma omp parallel for
    for (size_t i = lookback; i < n; ++i) {
        if (prices[i - lookback] <= 0 || prices[i] <= 0) {
            result[i - lookback] = 0;  // Handle invalid data
        } else {
            result[i - lookback] = (prices[i] / prices[i - lookback]) - 1;
        }
    }

    return result;
}


PYBIND11_MODULE(fastfactor, m) {
    m.def("momentum", &momentum, "Calculate momentum factor");
    m.def("log_returns", &log_returns, "Compute log returns");
    m.def("rolling_volatility", &rolling_volatility, "Compute rolling volatility");
}
