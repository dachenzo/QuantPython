#include <vector>
#include <omp.h>
#include <cmath>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <set>
using namespace std;
namespace py = pybind11;

//Super fast rolling percentile needed, O(nlogwindow) is too large




//Possible futher optimisation to avoid sorting every loop and the erase function
vector<double> rolling_percentile(const vector<double>& x, int window, double percentile) {
    size_t n = x.size();
    if (n < window) return {};

    vector<double> percentiles(n - window, 0.0);
    multiset<double> window_data(x.begin(), x.begin() + window);  // Correct window size

    for (size_t i = window; i < n; ++i) {
        // Compute percentile index
        size_t index = static_cast<size_t>(percentile * window / 100.0);
        auto it = window_data.begin();
        advance(it, index);
        percentiles[i - window] = *it;

        // **Remove the oldest value before inserting new value**
        window_data.erase(window_data.find(x[i - window]));  // Correct order of operations
        window_data.insert(x[i]);
    }

    return percentiles;
}

vector<double> exponential_moving_average(const vector<double>& prices, int window, double smoothing) {
    size_t n = prices.size();
    if (n < window) return {};

    vector<double> ema(n, 0.0);

    double alpha = smoothing / (window + 1);

    ema[window-1] = prices[window - 1];

    for (size_t i = window; i < n; i++) {
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1];
    }

    return ema;

}
 

vector<double> rolling_zscore(const vector<double>& x, int window) {

    size_t n = x.size();
    if (n < window) return {};

    vector<double> zscores(n - window, 0.0);
    double sum_x  = 0.0, sum_x2 = 0.0;

    for (int i = 0; i < window; i++) {
        sum_x += x[i];
        sum_x2 += x[i] * x[i];
    }


    #pragma omp parallel for 
    for (size_t i = window; i < n; i++) {
        double mean_x = sum_x / window;
        double variance_x = (sum_x2 / window) - (mean_x * mean_x);
        double std_x = sqrt(variance_x);

        zscores[i - window] = (std_x == 0) ? 0: (x[i]- mean_x) / std_x;


        sum_x += x[i] - x[i - window];
        sum_x2 += x[i] * x[i] - x[i - window] * x[i - window];


    }
    return zscores;

}

vector<double> rolling_correlation(const vector<double> x, const vector<double> y, int window) {

    size_t n = x.size();
    if (n < window || y.size() != n) return {};

    vector<double> correlations(n-window);

    double sumX = 0.0, sumY = 0.0, sumX2 = 0.0, sumY2 = 0.0, sumXY = 0.0;

    for (int i = 0; i < window; i++) {
        sumX += x[i];
        sumY += y[i];
        sumX2 += x[i] * x[i];
        sumY2 += y[i] * y[i];
        sumXY += x[i] * y[i];
    }

    #pragma omp parallel for 
    for (size_t i = window; i < n; i++) {
        double meanX  = sumX / window;
        double meanY = sumY / window;

        double numerator = sumXY - window * meanX * meanY;
        double denominator = sqrt((sumX2 - window * meanX * meanX) * (sumY2 - window * meanY * meanY));

        correlations[i-window] = (denominator == 0) ? 0.0 : numerator / denominator;

        //update sums for next window;
        sumX +=  x[i] -x[i - window];
        sumY += y[i] - y[i-window];
        sumX2 += x[i] * x[i] - x[i - window] * x[i - window];
        sumY2 += y[i] * y[i] - y[i - window] * y[i - window];
        sumXY += x[i] * y[i] - x[i - window] * y[i - window];

        
    }


    return correlations;
}

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
    m.def("rolling_correlation", &rolling_correlation, "Compute rolling correlation");
    m.def("rolling_zscore", &rolling_zscore, "Compute rolling Z-score normalization");
    m.def("exponential_moving_average", &exponential_moving_average, "Compute EMA");
    m.def("rolling_percentile", &rolling_percentile, "Compute rolling percentile");
}
