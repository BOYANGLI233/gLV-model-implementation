import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec

# Define a function to visualise the model parameters
def plot_model_parameters(mu, M, epsilon, species):

    fig= plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 2, 1])  # Middle plot twice as wide
    
    # Panel A: Growth rates (μ)
    ax1 = fig.add_subplot(gs[0])
    colors = ['blue' if x < 0 else 'red' for x in mu]
    ax1.barh(species[::-1], mu[::-1], color=colors)
    ax1.set_xlabel('1/day')
    ax1.set_title('A) Growth Rates ($\mu$)')
    ax1.set_xlim(-0.2, 1.0)
    
    # Panel B: Interaction matrix (M)
    ax2 = fig.add_subplot(gs[1]) 
    cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'white', 'red'], N=256)
    cax = ax2.matshow(M, cmap=cmap, vmin=-5, vmax=5, aspect='auto')

    # Add text annotations for each cell
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            text = ax2.text(j, i, f'{M[i, j]:.2f}',  # Display with 2 decimal places
                           ha="center", va="center", color="black", fontsize=8)

    # Add colorbar specifically for this subplot
    cbar = fig.colorbar(cax, ax=ax2, label='Interaction strength', shrink=0.6)
    ax2.set_title('B) Species-Species Interactions ($M_{ij}$)')
    ax2.set_xticks(np.arange(len(species)))
    ax2.set_xticklabels(species, rotation=90)
    ax2.xaxis.set_ticks_position('bottom')
    
    # Panel C: Susceptibilities (ε)
    ax3 = fig.add_subplot(gs[2])
    colors = ['blue' if x < 0 else 'red' for x in epsilon[::-1]]
    ax3.barh(species[::-1], epsilon[::-1], color=colors)
    ax3.set_title('C) Susceptibilities ($\epsilon_{il}$)')
    ax3.set_xlabel('1/day')
    ax3.set_yticks([])  # Remove y-axis ticks
    ax3.set_ylabel('')  # Remove y-axis label
    
    # Adjust layout to prevent label clipping
    plt.tight_layout()

    return fig


# define the function to plot the microbial dynamics
def plot_simulation(species, orig_data, orig_time, sim_data, sim_time, sim_dx):
    
    # Set up figure and subplots with 2x2 layout
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    color = plt.cm.tab20.colors[:len(species)]  # Assign each species a unique color

    # Subplot 1: Original data plot (top-left, position [0,0])
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Ensure data is float64 for precision
    orig_data = orig_data.astype(np.float64)
    orig_total = np.sum(orig_data, axis=0)  # Total abundance for scaling
    log_orig_total = np.log10(orig_total) + 11  # Adjusted for better visibility
    
    # Plot stacked bars
    bottom = np.zeros(len(orig_time), dtype=np.float64)  # Ensure float64
    rel_orig = orig_data / orig_total  # Relative abundance
    
    for sp in range(len(species)):
        species_height = rel_orig[sp] * log_orig_total
        ax1.bar(orig_time, species_height, width=0.6,
                bottom=bottom, color=color[sp], label=species[sp])
        bottom += species_height
    
    ax1.set_title("Original data")
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Log10(Abundance)')

    # Subplot 2: Derivatives plot (top-right, position [0,1])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Ensure derivatives data is float64
    sim_dx = sim_dx.astype(np.float64)
        
    # Plot derivatives as line plots for each species
    for sp in range(len(species)):
        ax2.plot(sim_time, sim_dx[sp, :], color=color[sp], label=species[sp], linewidth=2)
    
    ax2.set_title("Computed Derivatives (dx/dt)")
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('dx/dt')
    ax2.grid(True, alpha=0.3)
        
    # Add horizontal line at y=0 for reference
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
    # Set x-axis to match simulation time range
    ax2.set_xlim([sim_time[0], sim_time[-1]])
        
    # Subplot 3: Simulated data plot (bottom row, spanning both columns [1,:])
    ax3 = fig.add_subplot(gs[1, :])  # Span both columns in the second row
    sim_data = sim_data.astype(np.float64)
    sim_total = np.sum(sim_data, axis=0)
    log_sim_total = np.log10(sim_total) + 11
    
    bottom = np.zeros(len(sim_time), dtype=np.float64)
    rel_sim = sim_data / sim_total
    
    for sp in range(len(species)):
        species_height = rel_sim[sp] * log_sim_total
        ax3.bar(sim_time, species_height, width=0.1,
                bottom=bottom, color=color[sp], label=species[sp])
        bottom += species_height
    
    ax3.set_title("Simulated dynamics")
    ax3.set_xlabel('Time (days)')
    ax3.set_xticks(orig_time)  # Align x-ticks with original data
    ax3.set_ylabel('Log10(Abundance)')

    # Add a global legend at the bottom
    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))

    return fig