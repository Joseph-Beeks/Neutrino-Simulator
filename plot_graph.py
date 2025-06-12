import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from config import *

def sim_vs_analytical(data):
    # Plotting analytical vs quantum
    fig, ax = plt.subplots(figsize=(6,4)) 

    analyticalData = data[0]
    qData = data[1]
    
    # Plot analytical solutions
    ax.plot(range(L), analyticalData[0], '-', label="Analytical P(νe → νμ)", color='blue')
    ax.plot(range(L), analyticalData[1], '-', label="Analytical P(νe → νe)", color='red')
    
    # Quantum simulation with error bars for electron neutrino
    ax.errorbar(qData[0], qData[1], 
                yerr=[qData[3], qData[4]],  # Lower and upper error bars
                fmt='o', 
                markerfacecolor='orange', 
                markeredgecolor='orange',
                alpha=0.5,
                ecolor='orange',
                label="Quantum Simulation P(νe → νe)", color='orange')
    
    # Quantum simulation with error bars for muon neutrino
    ax.errorbar(qData[0], qData[2], 
                yerr=[qData[5], qData[6]],  # Lower and upper error bars
                fmt='s', 
                markerfacecolor='green', 
                markeredgecolor='green',
                alpha=0.5,
                ecolor='green',
                label="Quantum Simulation P(νe → νμ)", color='green')
    
    # Formatting
    ax.set_ylim(top=1.3)
    ax.set_xlabel("L/E (km/GeV)")
    ax.set_ylabel("Probability")
    ax.set_title("Analytical vs Simulated  Neutrino Flavour Probability")
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.425, 1) )#loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig("simulation_vs_analytical.png", dpi=300) 

    plt.show()
    return

def correlation_plot(data):
    fig, ax = plt.subplots(figsize=(6,4))

    # Set title for the plot
    fig.suptitle("Linear Regression of Quantum Simulation vs. Analytical Solution")#, fontsize=14)

    analyticalData = data[0]
    qData = data[1]

    # Get analytical probabilities for electron neutrinos
    analytical_electron = analyticalData[3]
    sim_electron = qData[1]

    # Scatter plot of data points
    ax.plot(analytical_electron, sim_electron, 'o', color='orange', label='Electron neutrino data', alpha=0.7)

    # Calculate and plot linear regression
    m_e, b_e, r_e, p_e, std_err_e = stats.linregress(analytical_electron, sim_electron)
    x_line = np.linspace(0, 1, 100)
    y_line_e = m_e * x_line + b_e
    ax.plot(x_line, y_line_e, '-', color='red', label=f'Electron fit: y = {m_e:.2f}x {b_e:+.2f} (R²={r_e**2:.4f})')

    # Add ideal y = x correlation line
    ax.plot([0, 1], [0, 1], '--', color='black', alpha=0.5, label='Perfect correlation')

    # Set limits, labels, and title
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel("Analytical Probability")
    ax.set_ylabel("Quantum Simulation Probability")
    ax.grid(True, alpha=0.3)
    ax.legend()#fontsize=8)

    plt.tight_layout()
    plt.savefig("correlation.png", dpi=300)  
    
    plt.show()