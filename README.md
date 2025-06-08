# The Evolving Architecture of Resilience: Code for Chapter 14

This repository contains the Python simulation code for **Chapter 14: The Evolving Architecture of Resilience: Self-Organization of Elasticities** of the book draft, *Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness* by Axel Pond.

This simulation serves as the empirical basis for the chapter, demonstrating how a system's fundamental resilience strategy—its architectural reliance on different adaptive capacities—can evolve under persistent environmental pressures.

The full codebase for the book, including all simulation experiments, is available at the main repository: [https://github.com/threshold-dialectics](https://github.com/threshold-dialectics).

## Table of Contents
- [About the Simulation](#about-the-simulation)
- [Core Hypothesis](#core-hypothesis)
- [Prerequisites and Installation](#prerequisites-and-installation)
- [How to Run the Simulation](#how-to-run-the-simulation)
- [Expected Outputs](#expected-outputs)
- [Connecting Code to the Book's Concepts](#connecting-code-to-the-books-concepts)
- [Citation](#citation)
- [License](#license)

## About the Simulation

This agent-based model (ABM) explores how a population of agents, each with a heritable "resilience architecture," evolves over time when subjected to different chronic environmental conditions, known as **Threshold Dialectics (TD) regimes**.

An agent's resilience architecture is defined by its **tolerance elasticities** ($w_1, w_2, w_3$), which determine its strategic reliance on three core adaptive capacities:
- **Perception Gain ($w_1$):** Reliance on sensing and vigilance.
- **Policy Precision ($w_2$):** Reliance on efficiency and optimized rules.
- **Energetic Slack ($w_3$):** Reliance on resource buffers and reserves.

The simulation subjects populations to one of three distinct TD regimes:
1.  **Scarcity:** A resource-poor environment where efficiency is paramount.
2.  **Shock:** An environment with stable periods punctuated by severe, unpredictable resource shocks, where buffering capacity is critical for survival.
3.  **Flux:** A volatile environment with transient opportunities, where perceptual acuity and adaptability are key.

## Core Hypothesis
The central hypothesis tested in this simulation is that persistent TD regimes act as powerful selective pressures, shaping the evolution of the population's average $w_k$ profile. Over generations, we expect to see the emergence of distinct, specialized resilience architectures that are FEP-optimal for each environment:
- **Scarcity** should select for high **$w_2$** (precision/efficiency).
- **Shock** should select for high **$w_3$** (slack/buffering).
- **Flux** should select for high **$w_1$** (perception/adaptability).

> **Note on the Simulation Design:** To ensure a clear and unambiguous evolutionary signal for this "Minimal Viable Simulation," a direct fitness benefit was explicitly hardcoded for the favored $w_k$ profile in the "Scarcity" and "Flux" regimes. This is a methodological simplification designed to create a clean test of selection on a heritable TD trait. Chapter 14 of the book discusses this design choice and its implications in detail.

## Prerequisites and Installation

The simulation is written in Python 3 and requires several scientific computing libraries.

1.  **Clone the repository:**
    """bash
    git clone https://github.com/threshold-dialectics/your-repo-name.git
    cd your-repo-name
    """

2.  **Install the dependencies:**
    It is recommended to use a virtual environment. The required packages are:
    - "numpy"
    - "pandas"
    - "matplotlib"
    - "seaborn"
    - "scipy"
    - "tqdm"
    - "python-ternary"
    - "scikit-learn"

    You can install them all using pip:
    """bash
    pip install numpy pandas matplotlib seaborn scipy tqdm python-ternary scikit-learn
    """

## How to Run the Simulation

The entire experiment can be run from a single script.

1.  **Execute the script:**
    """bash
    python evolved_simulation_code_refactored.py
    """

2.  **Simulation Process:**
    The script will automatically:
    - Create a "results/" directory if it doesn't exist.
    - Run the simulation for all three regimes ("Shock", "Scarcity", "Flux").
    - Execute a number of replicate runs for each regime, as defined in the "CONFIG" dictionary.
    - Save all raw data, statistical analyses, and plots to the "results/" directory.
    - Print progress bars and statistical test summaries to the console.

3.  **Modifying Parameters:**
    You can easily modify simulation parameters by editing the "CONFIG" dictionary at the top of "evolved_simulation_code_refactored.py". Key parameters include:
    - "num_generations"
    - "num_replicates"
    - "initial_population_size"
    - "mutation_strength_w"

## Expected Outputs

After the script finishes, the "results/" directory will contain several types of files, each named with the simulation parameters (e.g., "nrep5_ngen200").

### Data Logs (CSV)
- "population_log_[regime]...csv": Aggregate population-level data for each generation (mean $w_k$, population size, etc.).
- "individual_agent_log_[regime]...csv": Detailed fitness-related data for individual agents (lifespan, offspring count, etc.).

### Plots (PNG)
- **Time Series Evolution:** "wk_evolution_timeseries_...png"
  - Shows the change in mean $w_k$ values, population size, and other metrics over 200 generations.
- **Ternary Plot:** "wk_evolution_ternary_...png"
  - Visualizes the evolutionary trajectories of the mean population $w_k$ profiles in the 2-simplex space, showing divergence towards different corners.
  - ![Ternary Plot](Images/wk_evolution_ternary_nrep5_ngen200.png)
- **KDE Distribution Plots:** "wk_dist_kde_gen_[generation_number]...png"
  - Shows the distribution of $w_1, w_2, w_3$ across the entire population at specific generational snapshots, illustrating the process of specialization.
  - ![KDE Plot Gen 0](Images/wk_dist_kde_gen_0_nrep5_ngen200.png)
  - ![KDE Plot Gen 199](Images/wk_dist_kde_gen_199_nrep5_ngen200.png)

### Statistical Summaries (JSON, CSV)
- "statistical_summary_...csv" and "...json": Detailed results of statistical tests (ANOVA, t-tests) comparing the final $w_k$ profiles across regimes, trend analyses, and other key metrics.
- "summary_...json": A consolidated JSON file containing simulation metadata, configuration, and all statistical results.

## Connecting Code to the Book's Concepts

This simulation operationalizes the core concepts of **Threshold Dialectics (TD)** and the **Free Energy Principle (FEP)** to explore the long-term evolution of resilience.

- **Adaptive Levers ($\gLever, \betaLever, \FEcrit$):** Agents in the simulation actively manage internal states that are proxies for these levers to survive and gather resources. Perception Gain ($\gLever$) relates to foraging efficiency, Policy Precision ($\betaLever$) to resource use strategy, and Energetic Slack ($\FEcrit$) to their stored energy.
- **Tolerance Sheet ($\Theta_T$):** Each agent's viability is governed by its individual Tolerance Sheet, $\Theta_T^i = C \cdot (\gLever^i)^{w_1^i} (\betaLever^i)^{w_2^i} (\FEcrit^i)^{w_3^i}$. The heritable $w_k^i$ profile means that selection acts on the very architecture of an agent's resilience.
- **FEP-Driven Adaptation:** While not a formal FEP model, the agent's behavior (e.g., pulsing levers in response to low safety margins) is designed to be a plausible heuristic for what an FEP-driven agent would do to minimize long-term surprise (i.e., avoid collapse).
- **Meta-Adaptation:** The core of this chapter is demonstrating meta-adaptation. The simulation shows that the FEP-driven imperative to persist in a given TD regime leads to the selection and evolution of the underlying resilience architecture ($w_k$ profile) itself.

## Citation

If you use or refer to this code or the concepts from Threshold Dialectics, please cite the accompanying book:

@book{pond2025threshold,
  author    = {Axel Pond},
  title     = {Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness},
  year      = {2025},
  isbn      = {978-82-693862-2-6},
  publisher = {Amazon Kindle Direct Publishing},
  url       = {https://www.thresholddialectics.com},
  note      = {Code repository: \url{https://github.com/threshold-dialectics}}
}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
