# Neutrino Paramaters: from nufit
# -------------------------- #
E = energyNu = 1             # E_ν term units GeV
L = 40000                    # L distance travelled units km        
theta = 0.5878               # mixing angle 12 from nufit September 2024
sinSqrCoefficient = 0.85178  # sin²(2θ)
deltaM12 = 7.49e-5           # Δm² term units eV^2
# -------------------------- #

### KEY PROGRAM PARAMATERS ###
tokenIBM = 'ENTER YOUR IBM TOKEN'
shots = 1000
numTrials = 10
timeSamplesSim = 10 # Number of points to simulate
timeSamplesAnalytical = L # Default to match length of propagation
realQPU = False