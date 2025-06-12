import pandas as pd
import numpy as np
import os
from config import *

def save_data(filename, analyticalData, simulationData):
    # Unpack the simulation data
    (timePoints, simElectron, simMuon,
     electronErrorLower, electronErrorUpper,
     muonErrorLower, muonErrorUpper,
     allTrialProbabilitiesElectron, allTrialProbabilitiesMuon) = simulationData

    # Retrieve analytical data
    analyticalElectronToMu, analyticalElectronToElectron, timePointsAnalytical = analyticalData

    # Ensure analytical data matches simulation time points (Useful for direct sim vs analytical comparison)
    analyticalElectronToElectronMatched = np.interp(timePoints, timePointsAnalytical, analyticalElectronToElectron)
    analyticalElectronToMuMatched = np.interp(timePoints, timePointsAnalytical, analyticalElectronToMu)

    # Create a dictionary for metadata
    metadata_dict = {
        "Neutrino Energy (GeV)": E,
        "Propagation Distance (km)": L,
        "Mixing Angle Theta": theta,
        "sin²(2θ)": sinSqrCoefficient,
        "Mass-Squared Difference (eV^2)": deltaM12,
        "Shots Used": shots,
        "Number of Trials": numTrials,
        "Simulation Time Points": timeSamplesSim,
        "Analytical Time Points": timeSamplesAnalytical,
        "Using Real QPU": realQPU
    }

    # Create a DataFrame for metadata
    metadata_df = pd.DataFrame([metadata_dict])

    # DataFrame for Time Points
    time_df = pd.DataFrame({'Time Points in Simulation': pd.Series(timePoints)})

    # DataFrame for Analytical Data (This is the lengthier data useful for plotting)
    analytical_df = pd.DataFrame({
        'Analytical Time Points': pd.Series(timePointsAnalytical),
        'Analytical Electron': pd.Series(analyticalElectronToElectron),
        'Analytical Muon': pd.Series(analyticalElectronToMu)
    })

    # DataFrame ffor analytical data matched to simulation time points
    analytical_sim_match_df = pd.DataFrame({
        'Analytical Electron Sim Time-Matched': pd.Series(analyticalElectronToElectronMatched),
        'Analytical Muon Sim Time-Matched': pd.Series(analyticalElectronToMuMatched)
    })

    # DataFrame for Simulation Results (Median & Errors)
    simulation_df = pd.DataFrame({
        'Simulation Electron Median': pd.Series(simElectron),
        'Electron Lower Error': pd.Series(electronErrorLower),
        'Electron Upper Error': pd.Series(electronErrorUpper),
        'Simulation Muon Median': pd.Series(simMuon),
        'Muon Lower Error': pd.Series(muonErrorLower),
        'Muon Upper Error': pd.Series(muonErrorUpper)  # Fixed issue here
    })

    # DataFrame for Individual Trial Probabilities
    trial_df = pd.DataFrame()
    if allTrialProbabilitiesElectron and allTrialProbabilitiesMuon:
        num_trials = len(allTrialProbabilitiesElectron[0])
        for i in range(num_trials):
            trial_df[f'Trial {i+1}: Electron Probabilities'] = [trial[i] for trial in allTrialProbabilitiesElectron]
            trial_df[f'Trial {i+1}: Muon Probabilities'] = [trial[i] for trial in allTrialProbabilitiesMuon]

    # Concatenating all data
    df = pd.concat([metadata_df, analytical_df, analytical_sim_match_df, time_df, simulation_df, trial_df], axis=1)

    # Save to CSV
    df.to_csv(filename, index=False)

    print(f"Data saved to {filename} with columns: {list(df.columns)}")

def read_data(filename):
    # Read the data from the CSV file
    df = pd.read_csv(filename)

    # Extract the metadata, iloc 0 ensures we aren't accidently sampling nans
    metadata_dict = {
        "Neutrino Energy (GeV)": df["Neutrino Energy (GeV)"].iloc[0],
        "Propagation Distance (km)": df["Propagation Distance (km)"].iloc[0],
        "Mixing Angle Theta": df["Mixing Angle Theta"].iloc[0],
        "sin²(2θ)": df["sin²(2θ)"].iloc[0],
        "Mass-Squared Difference (eV^2)": df["Mass-Squared Difference (eV^2)"].iloc[0],
        "Shots Used": int(df["Shots Used"].iloc[0]),
        "Number of Trials": int(df["Number of Trials"].iloc[0]),
        "Simulation Time Points": int(df["Simulation Time Points"].iloc[0]),
        "Analytical Time Points": int(df["Analytical Time Points"].iloc[0]),
        "Using Real QPU": bool(df["Using Real QPU"].iloc[0]),
    }

    # Extract the time points
    timePoints = df["Time Points in Simulation"].dropna().to_numpy()

    # Extract the simulation data (Median and Errors)
    simElectron = df["Simulation Electron Median"].dropna().to_numpy()
    simMuon = df["Simulation Muon Median"].dropna().to_numpy()
    electronErrorLower = df["Electron Lower Error"].dropna().to_numpy()
    electronErrorUpper = df["Electron Upper Error"].dropna().to_numpy()
    muonErrorLower = df["Muon Lower Error"].dropna().to_numpy()
    muonErrorUpper = df["Muon Upper Error"].dropna().to_numpy()

    # Extract the trial probabilities
    # Identify the number of trials from the trial columns
    trial_columns = [col for col in df.columns if 'Trial' in col]
    allTrialProbabilitiesElectron = []
    allTrialProbabilitiesMuon = []
    
    for i in range(len(timePoints)):
        trialProbabilitiesElectron = []
        trialProbabilitiesMuon = []
        
        for col in trial_columns:
            if "Electron" in col:
                trialProbabilitiesElectron.append(df[col].iloc[i])
            elif "Muon" in col:
                trialProbabilitiesMuon.append(df[col].iloc[i])
        
        allTrialProbabilitiesElectron.append(trialProbabilitiesElectron)
        allTrialProbabilitiesMuon.append(trialProbabilitiesMuon)
    
    # Extract the analytical data (both general and time-matched)
    analyticalElectronToElectron = df["Analytical Electron"].to_numpy()
    analyticalElectronToMu = df["Analytical Muon"].to_numpy()
    analyticalElectronToElectronMatched = df["Analytical Electron Sim Time-Matched"].dropna().to_numpy()
    analyticalElectronToMuMatched = df["Analytical Muon Sim Time-Matched"].dropna().to_numpy()
    
    # Create the result structure similar to the output of quantum simulation
    simulationData = (
        timePoints,
        simElectron,
        simMuon,
        electronErrorLower,
        electronErrorUpper,
        muonErrorLower,
        muonErrorUpper,
        allTrialProbabilitiesElectron,
        allTrialProbabilitiesMuon,
    )

    analyticalData = (
        analyticalElectronToMu,
        analyticalElectronToElectron,
        df["Analytical Time Points"].to_numpy(),
        analyticalElectronToElectronMatched,
        analyticalElectronToMuMatched
    )
    
    # Return the data as a tuple, similar to the output format of the quantum simulation
    return analyticalData, simulationData, metadata_dict
