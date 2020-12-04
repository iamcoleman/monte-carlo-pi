/*
 * Defines the Test operator that is used to time the PiEstimator operation.
 * Also produces the analysis and results of the Monte Carlo pi estimation.
 * 
 * Includes code snippets written by NVIDIA Corporation.
 *
 */

#include "../inc/test.h"

#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <memory>
#include <iostream>
#include <cassert>
#include <typeinfo>
#include <stdio.h>
#include <helper_timer.h>
#include <math.h>
#include <cuda_runtime.h>

#include "../inc/piestimator.h"

template <typename Real>
bool Test<Real>::operator()()
{
    using std::stringstream;
    using std::endl;
    using std::setw;

    // Create a Stop Watch to measure time spent calculating pi
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);

    // Get device properties
    struct cudaDeviceProp deviceProperties;
    cudaError_t cudaResult = cudaGetDeviceProperties(&deviceProperties, device);

    if (cudaResult != cudaSuccess)
    {
        std::string msg("Could not get device properties: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Evaluate on GPU
    printf("Estimating Pi on GPU (%s)\n\n", deviceProperties.name);
    PiEstimator<Real> estimator(numSims, device, threadBlockSize, seed);
    sdkStartTimer(&timer);
    Real result = estimator();
    sdkStopTimer(&timer);
    elapsedTime = sdkGetAverageTimerValue(&timer)/1000.0f;

    // Tolerance to compare result with expected
    // This is just to check that nothing has gone very wrong with the
    // test, the actual accuracy of the result depends on the number of
    // Monte Carlo trials
    const Real tolerance = static_cast<Real>(0.01);

    // Display results
    Real abserror = fabs(result - static_cast<Real>(PI));
    Real relerror = abserror / static_cast<float>(PI);
    printf("Precision:      %s\n", (typeid(Real) == typeid(double)) ? "double" : "single");
    printf("Number of sims: %d\n", numSims);
    printf("Tolerance:      %e\n", tolerance);
    printf("GPU result:     %e\n", result);
    printf("Expected:       %e\n", PI);
    printf("Absolute error: %e\n", abserror);
    printf("Relative error: %e\n\n", relerror);

    // Check result against tolerance, should be fine unless something horrible happened
    if (relerror > tolerance)
    {
        printf("computed result (%e) does not match expected result (%e).\n", result, PI);
        pass = false;
    }
    else
    {
        pass = true;
    }

    // Print results
    printf("Performance = %.2f sims/s\n", numSims / elapsedTime);
    printf("Time = %.2f(ms)\n", elapsedTime * 1000.0f);

    sdkDeleteTimer(&timer);

    return pass;
}

// Explicit template instantiation
template struct Test<float>;
template struct Test<double>;
