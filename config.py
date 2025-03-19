# Neutrino Paramaters: from warwick and nufit
# ------------------------- #
E = 1                       # E_ν term units GeV
L = 1000                    # L distance travelled units km        
theta = 0.55                # mixing angle 12 from nufit September 2024
sinSqrCoefficient = 0.7943  # sin²(2θ)
deltaM12 = 3e-3             # Δm² term units eV^2
# ------------------------- #

### KEY PROGRAM PARAMATERS ###
tokenIBM = 'TOKEN'
shots = 1000
numTrials = 5
timeSamplesSim = 10 # Number of points to simulate
timeSamplesAnalytical = L # Default to match length of propagation
realQPU = False