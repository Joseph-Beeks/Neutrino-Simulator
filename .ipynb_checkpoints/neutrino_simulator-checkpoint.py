#General Python Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sp
from scipy import stats
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

#Loading your IBM Quantum account(s)
tokenIBM = 'TOKEN'
service = QiskitRuntimeService(channel="ibm_quantum",token=tokenIBM)

### COMPUTING ANALYTICAL SOLUTION CLASSICALLY ###

def analytical_probabilities(L, numPoints, sinSqrCoefficient, deltaM12, E):
    """Analytical solution"""
    timePoints = np.linspace(0, L, numPoints)  # Time points approx = distance km
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
    
    return electronToMu, electronToElectron

### SIMULATING NEUTRINO PROPAGATION IN QISKIT ###

def quantum_simulation(maxshots, numTrials, L, E, theta, deltaM12, realQPU):
    """Quantum simulation of neutrino propagation"""
    timePoints = np.linspace(0, L, 20)  # Time points approx = distance km
    
    probabilitiesElectronNu = []
    probabilitiesMuonNu = []
    errorsElectronUpper = []
    errorsElectronLower = []
    errorsMuonUpper = []
    errorsMuonLower = []
    
    #simulator = GenericBackendV2(num_qubits=1, shots=maxshots)
    #simulator = AerSimulator(method='statevector', shots=maxshots)

    if realQPU == False:
        fake_backend = FakeBrisbane()
        simulator = AerSimulator.from_backend(fake_backend)
        
    elif realQPU == True:
        service = QiskitRuntimeService(channel="ibm_quantum",token='TOKEN')
        #simulator = service.least_busy(simulator=False, operational=True)
        simulator = service.backend("ibm_brisbane")

    # Define Hamiltonian
    gamma = 2 * 1.27 * deltaM12 / E
    hamiltonianMassBasis = gamma * (0.5 * (SparsePauliOp("I") - SparsePauliOp("Z")))  # Hamiltonian is Pauli matrix form

    # Begin propagation
    for t in timePoints:
        #Storage for trial probabilities at time point
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
                sampler = Sampler(mode=simulator ,options={"default_shots": maxshots})
                simResult = sampler.run([compiledCircuit]).result()
                pub_result= simResult[0]
                
                # Extract the measurement counts.
                counts = pub_result.data.c.get_counts() 
                count0 = pub_result.data.c.get_counts()["0"] # Count for state "0" (electron neutrino)
                count1 = pub_result.data.c.get_counts()["1"] # Count for state "1" (muon neutrino)
            elif realQPU == False:
                compiledCircuit = transpile(qc, simulator)
                simResult = simulator.run(compiledCircuit, shots=maxshots).result() 
        
                # Extract the measurement counts.
                counts = simResult.get_counts(qc)
                count0 = counts.get("0", 0)  # Count for state "0" (electron neutrino)
                count1 = counts.get("1", 0)  # Count for state "1" (muon neutrino)
                
            # Compute probabilities for this trial
            probability0 = count0 / maxshots
            probability1 = count1 / maxshots
            
            # Save the trial results
            trialProbabilitiesElectron.append(probability0)
            trialProbabilitiesMuon.append(probability1)
        
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
            errorsElectronLower, errorsElectronUpper, errorsMuonLower, errorsMuonUpper)

def r2_solver(A, B):
    """ input 2 sets of data: output r^2 """
    A = np.array(A)
    B = np.array(B)
    
    # Calculate R^2 using scipy.stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(A, B)
    r_squared = r_value**2
    
    return r_squared

def save_data(filename, analyticalData, simulationData):
    # Extract data from the first simulation result in the list
    simResult = simulationData[0]  # Get the first simulation result
    
    timePoints = simResult[0]  # Time points
    simElectron = simResult[1]  # probability of finding an electron neutrino
    simMuon = simResult[2]     # probability of finding a muon neutrino
    
    # Upper and lower errors for each neutrino flavour
    electronErrorLower = simResult[3]  
    electronErrorUpper = simResult[4]  
    muonErrorLower = simResult[5]      
    muonErrorUpper = simResult[6]      
    
    # Retrieve analytical data
    analyticalElectronToMu, analyticalElectronToElectron = analyticalData

    # If the analytical data length doesn't match simulation time points,
    # recalculate the analytical solution using the same number of points.
    if len(analyticalElectronToElectron) != len(timePoints):
        numPoints = len(timePoints)
        analyticalElectronToMu, analyticalElectronToElectron = analytical_probabilities(numPoints)
   
    # Create a DataFrame with electron neutrino data first, followed by error data
    data = {
        'Time Points': timePoints,
        'Analytical Electron': analyticalElectronToElectron,
        'Simulation Electron': simElectron,
        'Electron Lower Error': electronErrorLower,
        'Electron Upper Error': electronErrorUpper,
        'Analytical Muon': analyticalElectronToMu,
        'Simulation Muon': simMuon,
        'Muon Lower Error': muonErrorLower,
        'Muon Upper Error': muonErrorUpper
    }
    
    # Create and save DataFrame
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    
    # Print confirmation with column names to verify
    print(f"Data saved to {filename} with columns: {list(data.keys())}")
    return