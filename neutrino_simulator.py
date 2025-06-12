#General Python Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sp
from scipy import stats
from config import *
import time
import pandas as pd
import os
import csv
# Importing Qiskit Libraries
import qiskit
from qiskit import __version__, QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.circuit.library import RealAmplitudes, PauliEvolutionGate
from qiskit.circuit import Parameter
# qiskit.visualization import *
from qiskit_aer import AerSimulator #Importing AerSimulator (Local Sim)
#Runtime
from qiskit_ibm_runtime import Options, QiskitRuntimeService,SamplerV2 as Sampler #Sampler
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

### COMPUTING ANALYTICAL SOLUTION CLASSICALLY ###

def analytical_probabilities():
    """Analytical solution"""
    timePoints = np.linspace(0, L, timeSamplesAnalytical)  # Time points approx = distance km
    electronToMu = []
    electronToElectron = []
    
    def probabilityElectronToMu(distance, coefficient, massDifference, energy):
        probability = coefficient*(np.sin((1.27*massDifference*distance)/(energy)))**2
        return probability
        
    for i in range(len(timePoints)):
        distance = timePoints[i]
        electronToMu.append(probabilityElectronToMu(distance, sinSqrCoefficient, deltaM12, E))
        electronToElectron.append(1-electronToMu[i])
        # P(νe--->νμ) & 1-P(νe--->νe) respectively
    
    return electronToMu, electronToElectron, timePoints

### SIMULATING NEUTRINO PROPAGATION IN QISKIT ###

def quantum_simulation():
    """Quantum simulation of neutrino propagation"""
    timePoints = np.linspace(0, L, timeSamplesSim)  # Time points approx = distance km

    allTrialProbabilitiesElectron = []
    allTrialProbabilitiesMuon = []
    probabilitiesElectronNu = []
    probabilitiesMuonNu = []
    errorsElectronUpper = []
    errorsElectronLower = []
    errorsMuonUpper = []
    errorsMuonLower = []

    # Allows for fake backends and real qpu usage
    if realQPU == False:
        fake_backend = FakeBrisbane()
        simulator = AerSimulator.from_backend(fake_backend)
        
    elif realQPU == True:
        #Loading your IBM Quantum account
        service = QiskitRuntimeService(channel="ibm_quantum",token=tokenIBM)
        simulator = service.backend("ibm_sherbrooke")

    # Define Hamiltonian
    gamma = 2 * 1.27 * deltaM12 / E
    hamiltonianMassBasis = gamma * (0.5 * (SparsePauliOp("I") - SparsePauliOp("Z")))  # Hamiltonian is Pauli matrix form

    # Begin propagation
    for t in timePoints:
        #Storage for trial probabilities at given time point
        trialProbabilitiesElectron = []
        trialProbabilitiesMuon = []

        # Repeat the experiment over a number of trials
        for trial in range(numTrials):
            # Single qubit system
            qc = QuantumCircuit(1, 1)  # Initialised to |0> representing electron neutrino
            
            # Apply RY rotation as our mixing matrix
            qc.ry(-2 * theta, 0)
            
            # Time Propagation
            evo = PauliEvolutionGate(hamiltonianMassBasis, time=t)
            qc.append(evo, [0])
            
            # Rotate back to flavour basis
            qc.ry(2 * theta, 0)
            qc.measure([0], [0])
            
            # Execute the circuit
            if realQPU == True:
                # Execute the circuit
                compiledCircuit = transpile(qc, simulator)
                sampler = Sampler(mode=simulator ,options={"default_shots": shots})
                simResult = sampler.run([compiledCircuit]).result()
                pub_result= simResult[0]
                
                # Extract the measurement counts.
                counts = pub_result.data.c.get_counts() 
                count0 = pub_result.data.c.get_counts()["0"] # Count for state "0" (electron neutrino)
                count1 = pub_result.data.c.get_counts()["1"] # Count for state "1" (muon neutrino)
                
            elif realQPU == False:
                compiledCircuit = transpile(qc, simulator)
                simResult = simulator.run(compiledCircuit, shots=shots).result() 
        
                # Extract the measurement counts.
                counts = simResult.get_counts(qc)
                count0 = counts.get("0", 0)  # Count for state "0" (electron neutrino)
                count1 = counts.get("1", 0)  # Count for state "1" (muon neutrino)
                
            # Compute probabilities for this trial
            probability0 = count0 / shots
            probability1 = count1 / shots
            
            # Save the trial results
            trialProbabilitiesElectron.append(probability0)
            trialProbabilitiesMuon.append(probability1)

        # Store all trials
        allTrialProbabilitiesElectron.append(trialProbabilitiesElectron)
        allTrialProbabilitiesMuon.append(trialProbabilitiesMuon)
        
        # Compute statistics for electron neutrino probability at time t
        medianElectron = np.median(trialProbabilitiesElectron)
        q1Electron = np.percentile(trialProbabilitiesElectron, 25)
        q3Electron = np.percentile(trialProbabilitiesElectron, 75)
        probabilitiesElectronNu.append(medianElectron)
        errorsElectronLower.append(medianElectron - q1Electron)  # Lower error bar
        errorsElectronUpper.append(q3Electron - medianElectron)  # Upper error bar
        
        # Compute statistics for muon neutrino probability at time t
        medianMuon = np.median(trialProbabilitiesMuon)
        q1Muon = np.percentile(trialProbabilitiesMuon, 25)
        q3Muon = np.percentile(trialProbabilitiesMuon, 75)
        probabilitiesMuonNu.append(medianMuon)
        errorsMuonLower.append(medianMuon - q1Muon)
        errorsMuonUpper.append(q3Muon - medianMuon)
    
    # Return the time points, median probabilities, and the corresponding error bars.
    return (timePoints, probabilitiesElectronNu, probabilitiesMuonNu,
            errorsElectronLower, errorsElectronUpper, errorsMuonLower, errorsMuonUpper,
            allTrialProbabilitiesElectron, allTrialProbabilitiesMuon)
    
def r2_solver(A, B):
    """ input 2 sets of data: output r^2 """
    A = np.array(A)
    B = np.array(B)
    
    # Calculate R^2 using scipy.stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(A, B)
    r_squared = r_value**2
    
    return r_squared