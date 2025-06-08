#evolved_simulation_code_refactored.py
import json
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import ternary  # For ternary plots (pip install python-ternary)
from tqdm import tqdm  # Optional for progress bars
import sys
from scipy import stats
import seaborn as sns
import os
from itertools import combinations
from sklearn.linear_model import LinearRegression  # For trend analysis

# --- Configuration Dictionary ---
# REFACTORED CONFIG to strengthen w_k selection pressures
CONFIG = {
    "agent": {
        "theta_t_constant_C": 1.0, "fcrit_initial_mean": 75.0, "fcrit_initial_std": 15.0,
        "fcrit_metabolic_cost": 0.05, "fcrit_reproduction_threshold": 120.0,
        "fcrit_reproduction_cost": 30.0, # Base cost, can be modified by resilience
        "min_survival_age_for_reproduction": 15,
        "fcrit_min_survival_threshold": 10.0,
        "max_age": 750, # <<< INCREASED max_age to make strategic deaths more likely

        "g_cost_params": {'kappa': 0.015, 'phi1': 0.7},
        "beta_cost_params": {'kappa_beta': 0.008, 'phi_beta': 0.9},
        "g_baseline": 1.0, "beta_baseline": 1.0, "g_pulse_target_factor": 1.4,
        "beta_pulse_target_factor": 1.15, "pulse_duration": 4,
        "safety_margin_pulse_trigger_factor": 0.25,
        "delta_fcrit_adapt_fraction": 0.08,

        "foraging_g_efficiency": 0.08,
        "foraging_beta_efficiency": 0.04,

        "strain_perception_noise_std": 0.03,
        "strain_avg_smoothing_alpha": 0.15, # <<< SLIGHTLY DECREASED for more reactivity to current strain

        # <<< ADDED/MODIFIED for stronger w_k selection >>>
        "theta_breach_penalty_factor": 0.1,  # Idea 1: Penalty multiplier on strain_avg if theta_t breached
        "theta_breach_death_prob": 0.05,     # Idea 1: Probability of death if theta_t breached
        "scarcity_efficiency_multiplier_w2_factor": 0.5, # Idea 4: Bonus to foraging for high w2 in Scarcity
        "scarcity_pulse_cost_multiplier": 1.5,          # Idea 5: Pulses are more costly in Scarcity
        "flux_perception_benefit_w1_factor": 0.3,       # Idea 8/9: Bonus to foraging for high w1 in Flux
        "reproduction_safety_margin_bonus_factor": 0.1, # Idea 10: Higher safety margin increases repro chance
    },
    "environment": {
        "global_resource_pool_initial": 8000.0, "global_resource_pool_max": 15000.0,
        "strain_base": 0.20, # <<< INCREASED base strain slightly
        "strain_noise_std": 0.03,

        "scarcity_resource_replenish_rate_total": 25.0, # <<< SLIGHTLY REDUCED for Scarcity
        "scarcity_strain_factor": 1.2,                  # <<< SLIGHTLY INCREASED for Scarcity
        "scarcity_foraging_potential_per_agent": 0.35,  # <<< SLIGHTLY REDUCED for Scarcity

        "shock_resource_replenish_rate_total": 55.0,
        "shock_foraging_potential_per_agent": 0.55,
        "shock_interval": 40, # <<< SLIGHTLY MORE FREQUENT SHOCKS
        "shock_duration_strain": 7, # <<< SLIGHTLY LONGER SHOCKS
        "shock_magnitude_strain_factor": 2.5, # <<< MORE SEVERE SHOCKS
        "shock_duration_resource": 2, # <<< LONGER RESOURCE SHOCK
        "shock_magnitude_resource_loss_fraction": 0.30, # <<< LARGER RESOURCE LOSS

        "flux_resource_replenish_rate_total": 50.0,
        "flux_foraging_potential_per_agent": 0.5,
        "flux_foraging_efficiency_period": 30, # <<< FASTER FLUX
        "flux_foraging_efficiency_min": 0.6,   # <<< WIDER RANGE OF FLUX
        "flux_foraging_efficiency_max": 1.2,
        "flux_strain_factor_when_inefficient": 1.4, # <<< HIGHER STRAIN IN BAD FLUX
        "flux_opportunity_bonus": 0.1, # Conceptual, for Idea 9
        "flux_adaptation_threshold_factor": 0.1 # Conceptual, for Idea 9
    },
    "simulation": {
        "num_generations": 200, # Start with fewer generations for testing refactor
        "steps_per_generation": 75,
        "initial_population_size": 50, "max_population_size": 100,
        "mutation_strength_w": 0.04, "mutation_min_w_val": 0.01,
        "seed": 42,
        "log_individual_agent_fitness_interval": 1,
        "log_wk_dist_gens": [0, 50, 100, 149, 199], # Adjusted for 200 gens
        "num_replicates": 5, # Start with fewer replicates for testing refactor
        "avg_over_last_n_gens_for_stats": 30,
    }
}

# --- Helper Functions ---
def mutate_w_profile(parent_w_profile, mutation_strength, min_w_val):
    noise = np.random.normal(0, mutation_strength, size=3)
    mutated_w = np.array(parent_w_profile) + noise
    mutated_w = np.maximum(mutated_w, min_w_val)
    mutated_w /= np.sum(mutated_w)
    return mutated_w

def initial_random_w_profile(min_w_val=0.01):
    alpha = [1.0, 1.0, 1.0] # Uniform prior for Dirichlet
    w = np.random.dirichlet(alpha)
    w = np.maximum(w, min_w_val)
    w /= np.sum(w)
    return w

# --- Analysis Helper Functions ---
def compute_time_series_metrics(series, generations):
    """Return basic peak/trough/slope metrics and AUC for a numeric series."""
    series = np.asarray(series)
    generations = np.asarray(generations)
    if len(series) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    diffs = np.diff(series)
    gen_first_peak = np.nan
    gen_first_trough = np.nan
    gen_slope_flip = np.nan
    peak_value = np.nan
    trough_value = np.nan
    for i in range(1, len(diffs)):
        if np.isnan(gen_first_peak) and diffs[i-1] > 0 and diffs[i] <= 0:
            gen_first_peak = generations[i]
            peak_value = series[i]
        if np.isnan(gen_first_trough) and diffs[i-1] < 0 and diffs[i] >= 0:
            gen_first_trough = generations[i]
            trough_value = series[i]
        if np.isnan(gen_slope_flip) and np.sign(diffs[i]) != np.sign(diffs[0]) and np.sign(diffs[i]) != 0:
            gen_slope_flip = generations[i]
        if not np.isnan(gen_first_peak) and not np.isnan(gen_first_trough) and not np.isnan(gen_slope_flip):
            break
    auc_full = np.trapz(series, generations)
    return gen_first_peak, gen_first_trough, gen_slope_flip, peak_value, trough_value, auc_full


def wk_snapshot_stats(values):
    """Return skewness, kurtosis and quantiles for a list of w_k values."""
    if len(values) == 0:
        return [np.nan]*7
    values = np.asarray(values)
    skewness = stats.skew(values)
    kurt = stats.kurtosis(values)
    q05, q25, q50, q75, q95 = np.percentile(values, [5, 25, 50, 75, 95])
    return skewness, kurt, q05, q25, q50, q75, q95


def ternary_to_cartesian(w1, w2, w3):
    """Convert ternary coordinates (summing to 1) to 2D Cartesian."""
    x = 0.5 * (2 * w2 + w3)
    y = (np.sqrt(3) / 2.0) * w3
    return np.array([x, y])


def compute_ternary_path_metrics(path_points):
    """Return path length and net angle (degrees) for a ternary trajectory."""
    if len(path_points) < 2:
        return np.nan, np.nan
    cart_points = np.array([ternary_to_cartesian(p[0], p[1], p[2]) for p in path_points])
    diffs = np.diff(cart_points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    path_length = float(segment_lengths.sum())
    delta = cart_points[-1] - cart_points[0]
    net_angle_deg = float(np.degrees(np.arctan2(delta[1], delta[0])))
    return path_length, net_angle_deg


def _clean_for_json(obj):
    """Recursively convert arrays and NaN/Inf to JSON-safe types."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [_clean_for_json(x) for x in obj.tolist()]
    return obj

# --- Core Classes ---
class Agent:
    def __init__(self, id_num, w_profile_initial, generation, parent_id=-1, initial_fcrit=None, config=CONFIG["agent"], env_regime="Scarcity"): # <<< Added env_regime
        self.id = id_num
        self.w_profile = np.array(w_profile_initial)
        self.generation = generation
        self.parent_id = parent_id
        self.config = config
        self.env_regime = env_regime # <<< Store env_regime

        self.g_lever = config["g_baseline"]
        self.beta_lever = config["beta_baseline"]

        if initial_fcrit is None:
            self.fcrit_lever = max(config["fcrit_min_survival_threshold"] + 1e-3,
                                   np.random.normal(config["fcrit_initial_mean"], config["fcrit_initial_std"]))
        else:
            self.fcrit_lever = initial_fcrit
        self.fcrit_lever = max(self.fcrit_lever, config["fcrit_min_survival_threshold"] + 1e-3)

        self.strain_avg = CONFIG["environment"]["strain_base"] # Use global strain_base for initial guess
        self.theta_t_constant_C = config["theta_t_constant_C"]
        self.age = 0
        self.current_pulse_timer_g = 0
        self.current_pulse_timer_beta = 0
        self.is_alive = True
        self.offspring_count = 0
        self.total_fcrit_gathered = 0
        self.cumulative_safety_margin = 0 # For potential fitness calcs
        self.steps_lived = 0


    def calculate_theta_t(self):
        g = max(1e-6, self.g_lever)
        b = max(1e-6, self.beta_lever)
        f = max(1e-6, self.fcrit_lever) # Agent's own Fcrit contributes to its Theta_T
        return self.theta_t_constant_C * (g**self.w_profile[0]) * (b**self.w_profile[1]) * (f**self.w_profile[2])

    def calculate_safety_margin(self):
        return self.calculate_theta_t() - self.strain_avg

    def perceive_strain(self, env_strain_signal):
        noise = np.random.normal(0, self.config["strain_perception_noise_std"])
        perceived_strain = max(0, env_strain_signal + noise)
        alpha = self.config["strain_avg_smoothing_alpha"]
        self.strain_avg = alpha * perceived_strain + (1 - alpha) * self.strain_avg

    def adapt_levers(self):
        if not self.is_alive: return

        if self.current_pulse_timer_g > 0:
            self.current_pulse_timer_g -= 1
        if self.current_pulse_timer_g == 0:
            self.g_lever = self.config["g_baseline"]

        if self.current_pulse_timer_beta > 0:
            self.current_pulse_timer_beta -= 1
        if self.current_pulse_timer_beta == 0:
            self.beta_lever = self.config["beta_baseline"]

        safety_margin = self.calculate_safety_margin()
        self.cumulative_safety_margin += safety_margin # Track for potential fitness
        self.steps_lived +=1

        # Approximate Theta_T with baseline levers and a modified fcrit for trigger threshold calculation
        # This is to prevent agents with very low fcrit (but high w3) from never pulsing
        # or agents with very high fcrit (but low w3) from pulsing too easily.
        # Use a representative Fcrit value for establishing the pulse trigger sensitivity.
        modified_fcrit_for_trigger = self.config["fcrit_initial_mean"] * 0.75 # Or some other representative value
        base_theta_approx = self.theta_t_constant_C * \
                            (self.config["g_baseline"]**self.w_profile[0]) * \
                            (self.config["beta_baseline"]**self.w_profile[1]) * \
                            (max(1e-6, modified_fcrit_for_trigger)**self.w_profile[2])

        trigger_threshold = self.config["safety_margin_pulse_trigger_factor"] * max(0.1, base_theta_approx)


        if safety_margin < trigger_threshold:
            # Budget for adaptation based on current Fcrit and w3 (less reliance on Fcrit -> more willingness to spend for adaptation)
            w3_modulation_factor = (1.0 - self.w_profile[2] * 0.75) # Higher w3 means smaller factor, more conservative
            effective_adaptation_budget_potential = self.fcrit_lever * self.config["delta_fcrit_adapt_fraction"] * max(0.1, w3_modulation_factor)

            # Ensure agent can survive the pulse duration + 1 step of metabolic cost
            min_fcrit_for_pulse_survival = self.config["fcrit_min_survival_threshold"] + \
                                           self.config["fcrit_metabolic_cost"] * (self.config["pulse_duration"] + 1)
            spendable_fcrit_for_pulses = max(0, self.fcrit_lever - min_fcrit_for_pulse_survival)
            actual_spendable_budget = min(effective_adaptation_budget_potential, spendable_fcrit_for_pulses)

            if actual_spendable_budget <= 0: return # Cannot afford to adapt via pulsing

            cost_g_pulse_per_step = self.config["g_cost_params"]['kappa'] * (self.config["g_baseline"] * self.config["g_pulse_target_factor"])**self.config["g_cost_params"]['phi1']
            cost_beta_pulse_per_step = self.config["beta_cost_params"]['kappa_beta'] * (self.config["beta_baseline"] * self.config["beta_pulse_target_factor"])**self.config["beta_cost_params"]['phi_beta']

            total_cost_g_full_pulse = cost_g_pulse_per_step * self.config["pulse_duration"]
            total_cost_beta_full_pulse = cost_beta_pulse_per_step * self.config["pulse_duration"]

            # Regime-specific cost adjustments
            if self.env_regime == "Scarcity":
                total_cost_g_full_pulse *= self.config.get("scarcity_pulse_cost_multiplier", 1.0)
                total_cost_beta_full_pulse *= self.config.get("scarcity_pulse_cost_multiplier", 1.0)


            can_afford_g_pulse = actual_spendable_budget >= total_cost_g_full_pulse
            can_afford_beta_pulse = actual_spendable_budget >= total_cost_beta_full_pulse

            if self.current_pulse_timer_g == 0 and self.current_pulse_timer_beta == 0: # Can only initiate one pulse type at a time
                action_taken = False
                # Preference for g-pulse vs beta-pulse based on w1 vs w2
                preference_g = self.w_profile[0] / max(1e-6, self.w_profile[0] + self.w_profile[1]) if (self.w_profile[0] + self.w_profile[1]) > 1e-6 else 0.5

                if random.random() < preference_g: # Prefer g
                    if can_afford_g_pulse:
                        self.g_lever = self.config["g_baseline"] * self.config["g_pulse_target_factor"]
                        self.current_pulse_timer_g = self.config["pulse_duration"]
                        # self.fcrit_lever -= total_cost_g_full_pulse # Cost is paid via metabolize_and_pay_costs over duration
                        action_taken = True
                else: # Prefer beta
                    if can_afford_beta_pulse:
                        self.beta_lever = self.config["beta_baseline"] * self.config["beta_pulse_target_factor"]
                        self.current_pulse_timer_beta = self.config["pulse_duration"]
                        # self.fcrit_lever -= total_cost_beta_full_pulse
                        action_taken = True

                if not action_taken: # If preferred action was unaffordable, try the other if affordable
                    if preference_g >= 0.5 and can_afford_beta_pulse and self.current_pulse_timer_beta == 0: # Original preference was g
                        self.beta_lever = self.config["beta_baseline"] * self.config["beta_pulse_target_factor"]
                        self.current_pulse_timer_beta = self.config["pulse_duration"]
                    elif preference_g < 0.5 and can_afford_g_pulse and self.current_pulse_timer_g == 0: # Original preference was beta
                        self.g_lever = self.config["g_baseline"] * self.config["g_pulse_target_factor"]
                        self.current_pulse_timer_g = self.config["pulse_duration"]

    def calculate_lever_costs(self):
        cost = 0
        if self.g_lever > self.config["g_baseline"] + 1e-3: # Cost only if above baseline
            cost += self.config["g_cost_params"]['kappa'] * (self.g_lever**self.config["g_cost_params"]['phi1'])
        if self.beta_lever > self.config["beta_baseline"] + 1e-3: # Cost only if above baseline
            cost += self.config["beta_cost_params"]['kappa_beta'] * (self.beta_lever**self.config["beta_cost_params"]['phi_beta'])

        # <<< MODIFIED: Regime-specific pulse cost multiplier >>>
        if self.env_regime == "Scarcity" and (self.current_pulse_timer_g > 0 or self.current_pulse_timer_beta > 0) :
            cost *= self.config.get("scarcity_pulse_cost_multiplier", 1.0)
        return cost

    def metabolize_and_pay_costs(self):
        if not self.is_alive: return
        total_cost = self.config["fcrit_metabolic_cost"] + self.calculate_lever_costs()
        self.fcrit_lever -= total_cost

    def forage_for_resources(self, environment, current_foraging_efficiency=1.0):
        if not self.is_alive: return

        base_foraging_potential = environment.get_resource_for_agent(self)

        g_bonus_factor = 1.0
        if self.g_lever > self.config["g_baseline"]:
             g_bonus_factor = 1 + (self.g_lever - self.config["g_baseline"]) * self.config["foraging_g_efficiency"]

        beta_bonus_factor = 1.0
        if self.beta_lever > self.config["beta_baseline"]:
            beta_bonus_factor = 1 + (self.beta_lever - self.config["beta_baseline"]) * self.config["foraging_beta_efficiency"]

        # Regime-specific foraging modifications
        regime_specific_efficiency_bonus = 1.0
        if self.env_regime == "Scarcity": # Idea 4
            regime_specific_efficiency_bonus *= (1 + self.config.get("scarcity_efficiency_multiplier_w2_factor", 0.0) * self.w_profile[1])
        elif self.env_regime == "Flux": # Idea 8/9
            # Conceptual: agent's ability to adapt to flux could be tied to w1
            # This is a simplified way to give a bonus for high w1 in Flux
            # A more complex model might have agent learn to match env.current_foraging_efficiency
            regime_specific_efficiency_bonus *= (1 + self.config.get("flux_perception_benefit_w1_factor", 0.0) * self.w_profile[0])
            # Example for Idea 9: if agent is well adapted (e.g. its g_lever is high, signifying good perception)
            # if self.g_lever > self.config["g_baseline"] * 1.1: # Simple proxy for good adaptation
            #    base_foraging_potential += CONFIG["environment"].get("flux_opportunity_bonus", 0.0)


        resources_gained = base_foraging_potential * \
                           max(0.1, g_bonus_factor) * \
                           max(0.1, beta_bonus_factor) * \
                           float(current_foraging_efficiency) * \
                           regime_specific_efficiency_bonus # <<< ADDED regime bonus

        self.fcrit_lever += resources_gained
        self.total_fcrit_gathered += resources_gained


    def check_survival(self):
        if not self.is_alive: return False

        # <<< MODIFIED: Idea 1 - Penalties/death for Theta_T breach >>>
        safety_margin = self.calculate_safety_margin()
        if safety_margin < 0: # Theta_T breached
            breach_penalty = self.config.get("theta_breach_penalty_factor", 0.0) * abs(safety_margin) # Penalty scales with breach magnitude
            self.fcrit_lever -= breach_penalty
            if random.random() < self.config.get("theta_breach_death_prob", 0.0):
                self.is_alive = False
                # print(f"Agent {self.id} died from Theta_T breach. SM: {safety_margin:.2f}") # For debugging
                return False # Died from breach prob

        if self.fcrit_lever < self.config["fcrit_min_survival_threshold"]:
            self.is_alive = False
            # print(f"Agent {self.id} died from Fcrit depletion. Fcrit: {self.fcrit_lever:.2f}") # For debugging
        elif self.config.get("max_age") and self.age >= self.config["max_age"]:
            self.is_alive = False
            # print(f"Agent {self.id} died from old age. Age: {self.age}") # For debugging
        return self.is_alive

    def can_reproduce(self):
        if not (self.is_alive and \
                self.fcrit_lever >= self.config["fcrit_reproduction_threshold"] and \
                self.age >= self.config["min_survival_age_for_reproduction"]):
            return False

        # <<< MODIFIED: Idea 10 - Safety margin bonus for reproduction chance >>>
        # Calculate recent average safety margin (e.g. over last N steps, or just current for simplicity)
        # For simplicity, using current safety margin. A more robust approach would average over recent history.
        current_sm = self.calculate_safety_margin()
        sm_bonus_factor = self.config.get("reproduction_safety_margin_bonus_factor", 0.0)
        
        # Probability scaled by safety margin. If SM is negative, this can reduce prob.
        # We want bonus if SM is positive. Max probability is 1.0.
        # A simple scaling: if SM is positive, increase base chance. If negative, decrease.
        # Let's use a sigmoid-like or bounded linear scaling.
        # For now, a simpler approach: make it a multiplier on a base "attempt" probability,
        # or directly influence if the "can_reproduce" check passes.
        # Simplified: if SM is good, more likely to pass this check if other conditions met.
        # Let's make it a threshold: if SM is too low, even if Fcrit is high, less likely to reproduce.
        # This is implicitly handled if low SM leads to Fcrit costs/death.

        # More direct: higher SM increases *chance* of passing this check IF other conditions are met.
        # This could be implemented by having can_reproduce() return a probability rather than bool,
        # or by adding a random check here.
        # For this refactor, let's stick to the base conditions and rely on indirect SM effects via Theta_T breach.
        # A more explicit link could be:
        # `repro_chance_modifier = 1.0 + current_sm * sm_bonus_factor`
        # `if random.random() > np.clip(repro_chance_modifier, 0.1, 1.0): return False`
        # This makes it probabilistic. For now, keeping it deterministic based on Fcrit/Age.
        # The selection pressure from safety margin will come from survival due to Theta_T breach effects.

        return True


    def reproduce(self, next_agent_id, mutation_strength, min_w_val):
        if not self.can_reproduce(): return None # Should be redundant if called after check

        # Idea 11: Reproduction cost scaled by resilience (e.g., average safety margin)
        # This requires tracking lifetime average safety margin, which adds complexity.
        # For now, using base reproduction cost.
        # avg_sm_lifetime = self.cumulative_safety_margin / max(1, self.steps_lived)
        # cost_reduction_factor = max(0.1, 1 - (avg_sm_lifetime * self.config.get("reproduction_cost_resilience_scaling_factor", 0.0)))
        # current_reproduction_cost = self.config["fcrit_reproduction_cost"] * cost_reduction_factor
        current_reproduction_cost = self.config["fcrit_reproduction_cost"]


        parent_fcrit_after_cost = self.fcrit_lever - current_reproduction_cost
        if parent_fcrit_after_cost < self.config["fcrit_min_survival_threshold"]:
            return None # Cannot afford to reproduce and survive

        self.fcrit_lever = parent_fcrit_after_cost
        self.offspring_count += 1

        offspring_w_profile = mutate_w_profile(self.w_profile, mutation_strength, min_w_val)
        # Offspring starts with mean fcrit, not inheriting parent's current fcrit
        offspring_initial_fcrit = self.config["fcrit_initial_mean"] # Or add some variance

        return Agent(id_num=next_agent_id, w_profile_initial=offspring_w_profile,
                     generation=self.generation + 1, parent_id=self.id,
                     initial_fcrit=offspring_initial_fcrit, config=self.config, env_regime=self.env_regime) # Pass env_regime

class Environment:
    def __init__(self, regime_name, population_size_func, config=CONFIG["environment"]):
        self.current_regime = regime_name
        self.config = config
        self.time = 0
        self.get_population_size = population_size_func # Function to get current pop size

        self.global_resource_pool = config["global_resource_pool_initial"]
        self.strain_level = config["strain_base"] # Initial base strain
        self.current_foraging_efficiency = 1.0 # For Flux regime

        # Shock specific
        self.time_since_last_shock = config.get("shock_interval", 100) + 1 # Start as if a shock just finished far in past

        if self.current_regime not in ["Scarcity", "Shock", "Flux"]:
            raise ValueError(f"Unknown regime: {self.current_regime}")

    def update_resources_and_strain(self):
        self.time += 1
        self.time_since_last_shock +=1
        current_pop_size = max(1, self.get_population_size()) # Avoid division by zero

        # Replenish global resources
        replenish_total = self.config.get(f"{self.current_regime.lower()}_resource_replenish_rate_total", 0)
        self.global_resource_pool = min(self.config["global_resource_pool_max"],
                                        self.global_resource_pool + replenish_total)

        # Update strain level based on regime
        self.strain_level = self.config["strain_base"] # Reset to base

        if self.current_regime == "Scarcity":
            self.strain_level *= self.config["scarcity_strain_factor"]
            # Foraging efficiency is implicitly handled by scarcity_foraging_potential_per_agent

        if self.current_regime == "Shock":
            if self.time_since_last_shock >= self.config["shock_interval"]:
                self._apply_shock_event() # Apply resource shock
                self.time_since_last_shock = 0 # Reset timer, strain shock starts now

            # Strain shock active for its duration AFTER resource shock
            if 0 < self.time_since_last_shock <= self.config["shock_duration_strain"]: # Strain shock duration
                 self.strain_level = self.config["strain_base"] * self.config["shock_magnitude_strain_factor"]

        elif self.current_regime == "Flux":
            period = max(1, self.config["flux_foraging_efficiency_period"])
            # Sine wave for foraging efficiency fluctuation
            phase = (self.time % period) / period
            eff_range = self.config["flux_foraging_efficiency_max"] - self.config["flux_foraging_efficiency_min"]
            self.current_foraging_efficiency = self.config["flux_foraging_efficiency_min"] + \
                                               (eff_range / 2.0) * (1 + np.sin(2 * np.pi * phase - np.pi/2)) # Starts at min

            # Apply strain factor when foraging efficiency is low
            # e.g. if efficiency is in the bottom 30% of its dynamic range
            if self.current_foraging_efficiency < (self.config["flux_foraging_efficiency_min"] + eff_range * 0.3):
                self.strain_level = self.config["strain_base"] * self.config["flux_strain_factor_when_inefficient"]


    def _apply_shock_event(self): # This is for resource shock
        # Apply resource shock
        # The duration of resource shock is 1 step as per config
        loss_fraction = self.config.get("shock_magnitude_resource_loss_fraction",0.0) # Get from shock_config
        resource_loss = self.global_resource_pool * loss_fraction
        self.global_resource_pool = max(0, self.global_resource_pool - resource_loss)
        # print(f"Time {self.time}: Shock event! Resource pool reduced by {resource_loss:.2f} to {self.global_resource_pool:.2f}")


    def get_resource_for_agent(self, agent): # agent parameter is for API consistency, not used here yet
        current_pop_size = self.get_population_size()
        if current_pop_size == 0:
            return 0.0

        # Base potential from environment config
        attempted_forage_potential = self.config.get(f"{self.current_regime.lower()}_foraging_potential_per_agent", 0.0)
        return float(attempted_forage_potential)


    def get_strain_signal(self):
        # Actual strain signal perceived by agents, includes noise
        return max(0, np.random.normal(self.strain_level, self.config["strain_noise_std"]))


class Simulation:
    def __init__(self, sim_config=CONFIG["simulation"], agent_config=CONFIG["agent"], env_config=CONFIG["environment"], env_regime="Scarcity"):
        self.sim_config = sim_config
        self.agent_config = agent_config
        self.env_config = env_config
        self.env_regime = env_regime

        if sim_config["seed"] is not None:
            np.random.seed(sim_config["seed"])
            random.seed(sim_config["seed"])

        self.population = []
        self.next_agent_id_counter = 0
        self.environment = Environment(env_regime, lambda: len(self.population), self.env_config) # Pass self.env_config
        self.current_generation = 0
        self.current_step_in_generation = 0
        self.num_births_this_generation = 0
        self.data_log = [] # For aggregate data
        self.individual_agent_data_log = [] # For individual agent data

    def _get_next_agent_id(self):
        val = self.next_agent_id_counter
        self.next_agent_id_counter += 1
        return val

    def _initialize_population(self):
        self.population = []
        self.next_agent_id_counter = 0 # Reset for each simulation instance
        for _ in range(self.sim_config["initial_population_size"]):
            w_init = initial_random_w_profile(self.sim_config["mutation_min_w_val"])
            # Pass the correct agent_config and env_regime to the Agent
            self.population.append(Agent(id_num=self._get_next_agent_id(), w_profile_initial=w_init,
                                         generation=0, config=self.agent_config, env_regime=self.env_regime))


    def _run_step(self):
        self.environment.update_resources_and_strain()
        current_agents_in_step = list(self.population) # Operate on a copy
        random.shuffle(current_agents_in_step) # Randomize order of agent actions

        for agent in current_agents_in_step:
            if not agent.is_alive:
                continue # Skip dead agents that might be in the list before pruning
            agent.age += 1
            agent.perceive_strain(self.environment.get_strain_signal())
            agent.adapt_levers() # Agent adapts its g and beta levers
            agent.forage_for_resources(self.environment, self.environment.current_foraging_efficiency)
            agent.metabolize_and_pay_costs() # Pay metabolic and lever costs
            agent.check_survival() # Check if agent survives the step

        # Remove dead agents from the main population list
        self.population = [agent for agent in self.population if agent.is_alive]


    def _selection_and_reproduction(self):
        # Survivors who lived through the generation's steps
        survivors_who_lived_generation = [agent for agent in self.population if agent.is_alive]
        newly_born_offspring = []

        # Determine parents (shuffle to randomize who gets to reproduce if pop > max_pop later)
        potential_parents = [agent for agent in survivors_who_lived_generation if agent.can_reproduce()]
        random.shuffle(potential_parents)

        for parent in potential_parents:
            offspring = parent.reproduce(self._get_next_agent_id(),
                                             self.sim_config["mutation_strength_w"],
                                             self.sim_config["mutation_min_w_val"])
            if offspring:
                newly_born_offspring.append(offspring)
                self.num_births_this_generation += 1

        # New population includes parents who survived reproduction + non-reproducing survivors + offspring
        new_population = [agent for agent in survivors_who_lived_generation if agent.is_alive] + newly_born_offspring
        self.population = new_population

        # Population cap
        if len(self.population) > self.sim_config["max_population_size"]:
            self.population = random.sample(self.population, self.sim_config["max_population_size"])


    def _log_generation_data(self):
        pop_size = len(self.population)
        log_entry = {
            "generation": self.current_generation,
            "environment_time": self.environment.time, # Total steps in environment
            "population_size": pop_size,
            "num_births": self.num_births_this_generation,
            "env_strain": self.environment.strain_level,
            "env_resources": self.environment.global_resource_pool,
            "env_foraging_eff": self.environment.current_foraging_efficiency
        }
        # Initialize w_k stats to NaN
        for i in range(3):
            log_entry[f"mean_w{i+1}"] = np.nan
            log_entry[f"std_w{i+1}"] = np.nan # Intra-population std of w_k
        log_entry.update({
            "mean_age": np.nan, "mean_fcrit": np.nan, "median_fcrit": np.nan,
            "mean_g_lever": np.nan, "mean_beta_lever": np.nan,
            "wk_dist_snapshot": np.nan # Store as NaN if no data or not a snapshot gen
        })
        if (self.sim_config["log_wk_dist_gens"]
            and self.current_generation in self.sim_config["log_wk_dist_gens"]):
            # store a list of [w1,w2,w3] for every agent alive this generation
            log_entry["wk_dist_snapshot"] = [
                agent.w_profile.tolist() for agent in self.population
            ]
        if pop_size > 0:
            w_profiles = np.array([agent.w_profile for agent in self.population])
            for i in range(3):
                log_entry[f"mean_w{i+1}"] = np.mean(w_profiles[:,i])
                log_entry[f"std_w{i+1}"] = np.std(w_profiles[:,i])

            log_entry["mean_age"] = np.mean([a.age for a in self.population])
            fcrits = [a.fcrit_lever for a in self.population]
            log_entry["mean_fcrit"] = np.mean(fcrits)
            log_entry["median_fcrit"] = np.median(fcrits)
            log_entry["mean_g_lever"] = np.mean([a.g_lever for a in self.population])
            log_entry["mean_beta_lever"] = np.mean([a.beta_lever for a in self.population])

            # Snapshot w_k distributions for specific generations
            # if self.sim_config["log_wk_dist_gens"] and \
            #    self.current_generation in self.sim_config["log_wk_dist_gens"]:
            #     log_entry["wk_dist_snapshot"] = [list(agent.w_profile) for agent in self.population]
        self.data_log.append(log_entry)

        # Log individual agent data based on interval
        if self.sim_config["log_individual_agent_fitness_interval"] > 0 and \
           self.current_generation > 0 and \
           self.current_generation % self.sim_config["log_individual_agent_fitness_interval"] == 0:
            for agent in self.population: # Log current state of SURVIVORS at this interval
                self.individual_agent_data_log.append({
                    "log_type": "interval_survivor",
                    "generation_logged": self.current_generation, # Generation when this log entry is made
                    "regime": self.env_regime,
                    "agent_id": agent.id,
                    "parent_id": agent.parent_id,
                    "birth_generation": agent.generation, # Generation agent was born
                    "current_age_at_log": agent.age, # Agent's age within its current generation
                    "current_fcrit_at_log": agent.fcrit_lever,
                    "offspring_count_total": agent.offspring_count, # Cumulative over agent's life
                    "total_fcrit_gathered_total": agent.total_fcrit_gathered, # Cumulative
                    "w1": agent.w_profile[0], "w2": agent.w_profile[1], "w3": agent.w_profile[2]
                })

    def _log_dead_agent_fitness(self, agent, death_generation_step):
        # death_generation_step is a tuple (generation_idx_of_death, step_idx_of_death)
        if self.sim_config["log_individual_agent_fitness_interval"] > 0 : # Check if fitness logging is enabled
            self.individual_agent_data_log.append({
                "log_type": "death",
                "generation_logged": death_generation_step[0], # Generation when death occurred and was logged
                "step_logged": death_generation_step[1], # Step within generation when death occurred
                "regime": self.env_regime,
                "agent_id": agent.id,
                "parent_id": agent.parent_id,
                "birth_generation": agent.generation, # Generation agent was born
                "final_age_at_log": agent.age, # Agent's age at death
                "final_fcrit_at_log": agent.fcrit_lever, # Fcrit at death
                "offspring_count_total": agent.offspring_count, # Cumulative
                "total_fcrit_gathered_total": agent.total_fcrit_gathered, # Cumulative
                "w1": agent.w_profile[0], "w2": agent.w_profile[1], "w3": agent.w_profile[2]
            })

    def run_simulation(self):
        print(f"Starting simulation for {self.env_regime} regime (Seed: {self.sim_config['seed']})...")
        self._initialize_population()

        for gen_idx in tqdm(range(self.sim_config["num_generations"]), desc=f"Regime: {self.env_regime}"):
            self.current_generation = gen_idx
            self.num_births_this_generation = 0 # Reset for this generation

            if not self.population: # Extinction before generation starts
                self._log_generation_data() # Log current (empty) state
                # print(f"Extinction in {self.env_regime} before generation {gen_idx} started.")
                break

            # Age agents and run steps within the generation
            for step_idx in range(self.sim_config["steps_per_generation"]):
                self.current_step_in_generation = step_idx
                agents_alive_before_step = {agent.id: agent for agent in self.population if agent.is_alive}
                self._run_step() # This also removes dead agents from self.population

                # Log agents that died this step
                for agent_id, prev_agent_state in agents_alive_before_step.items():
                    current_agent_in_pop = next((a for a in self.population if a.id == agent_id), None)
                    if current_agent_in_pop is None or not current_agent_in_pop.is_alive: # Agent died
                         self._log_dead_agent_fitness(prev_agent_state, (self.current_generation, self.current_step_in_generation))

                if not self.population: # Extinction mid-generation
                    break

            if not self.population: # Extinction by end of steps for this generation
                self._log_generation_data()
                # print(f"Extinction in {self.env_regime} at gen {gen_idx}, step {self.current_step_in_generation}.")
                break

            # Store state of agents alive before selection/reproduction might kill some (e.g. reproduction cost)
            agents_alive_before_selection = {agent.id: agent for agent in self.population if agent.is_alive}
            self._selection_and_reproduction() # Updates self.population

            # Log agents that died during selection/reproduction (e.g. parent died from repro cost)
            for agent_id, prev_agent_state in agents_alive_before_selection.items():
                current_agent_in_pop = next((a for a in self.population if a.id == agent_id), None)
                if current_agent_in_pop is None or not current_agent_in_pop.is_alive:
                     self._log_dead_agent_fitness(prev_agent_state, (self.current_generation, "selection"))


            self._log_generation_data() # Log population stats at end of generation

            if not self.population and gen_idx < self.sim_config["num_generations"] -1 : # Extinction after selection
                # print(f"Extinction in {self.env_regime} after selection in generation {gen_idx}.")
                break

        # Log fitness data for any agents still alive at the end of the entire simulation
        if self.sim_config.get("log_individual_agent_fitness_interval", 0) > 0 and self.population:
             for agent in self.population: # These are the final survivors
                self.individual_agent_data_log.append({
                    "log_type": "end_sim_survivor",
                    "generation_logged": self.current_generation, # Logged at the end of the last successful generation
                    "regime": self.env_regime,
                    "agent_id": agent.id,
                    "parent_id": agent.parent_id,
                    "birth_generation": agent.generation,
                    "final_age_at_log": agent.age, # Their age at the end of sim
                    "final_fcrit_at_log": agent.fcrit_lever,
                    "offspring_count_total": agent.offspring_count,
                    "total_fcrit_gathered_total": agent.total_fcrit_gathered,
                    "w1": agent.w_profile[0],
                    "w2": agent.w_profile[1],
                    "w3": agent.w_profile[2],
                })

        if self.population:
            print(f"Finished {self.env_regime}. Population: {len(self.population)}")
        else:
            print(f"Finished {self.env_regime} due to extinction at generation {self.current_generation}.")
        return pd.DataFrame(self.data_log), pd.DataFrame(self.individual_agent_data_log)


# --- Main Execution & Analysis ---
if __name__ == "__main__":
    # Create a unique results folder for this run using a timestamp or run ID if needed
    # For now, keeping it simple
    RESULTS_FOLDER = "results"
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    STATISTICAL_SUMMARY_LOG = []

    ESSENTIAL_STATS_GROUPS = {
        "TimeSeries_Peaks",
        "Ternary_Path",
        "Final_Mean_Wk_Endpoint",
        "Delta_Wk_Endpoint",
        "Longitudinal_Trend",
    }

    def append_stat(entry):
        if entry.get("analysis_group") not in ESSENTIAL_STATS_GROUPS:
            return
        SIM_OUTPUT["statistics"].append(entry)

    # Master container for everything we will write
    SIM_OUTPUT = {
        "meta": {
            "num_generations": CONFIG["simulation"]["num_generations"],
            "num_replicates": CONFIG["simulation"]["num_replicates"],
            "seed":           CONFIG["simulation"]["seed"],
            "regimes":        ["Shock", "Scarcity", "Flux"],
        },
        "group_legend": {
            "TimeSeries_Peaks": "peak/trough positions & AUC of per-generation means",
            "Ternary_Path": "length & net angle of mean w-vector trajectory",
            "Final_Mean_Wk_Endpoint": "final 30-gen mean ± CI95",
            "Delta_Wk_Endpoint": "final-minus-initial change ± CI95",
            "Longitudinal_Trend": "slope, p, r for mean vs generation"
        },
        "variable_legend": {
            "environment_time": "elapsed simulation steps",
            "population_size": "number of living agents",
            "num_births": "agents born during each generation",
            "env_strain": "environmental strain signal perceived by agents",
            "env_resources": "total resources available in environment",
            "env_foraging_eff": "global multiplier on foraging success",
            "mean_age": "mean age of living agents",
            "mean_fcrit": "mean critical energy reserve per agent",
            "median_fcrit": "median critical energy reserve per agent",
            "mean_w1": "population-mean perceptual weighting (w1)",
            "mean_w2": "population-mean operational weighting (w2)",
            "mean_w3": "population-mean resilience weighting (w3)",
            "std_w1": "standard deviation of w1 across agents",
            "std_w2": "standard deviation of w2 across agents",
            "std_w3": "standard deviation of w3 across agents",
            "delta_w1": "change in mean w1 from start to finish",
            "delta_w2": "change in mean w2 from start to finish",
            "delta_w3": "change in mean w3 from start to finish",
            "path_length": "distance travelled in w-space trajectory",
            "net_angle_deg": "angle from start to final mean w-vector"
        },
        "statistics": STATISTICAL_SUMMARY_LOG,
        "group_tests": []
    }

    all_regime_pop_data = {}
    all_regime_indiv_data = {}
    regimes_to_run = ["Shock", "Scarcity", "Flux"]
    num_replicates = CONFIG["simulation"]["num_replicates"]
    num_generations_config = CONFIG["simulation"]["num_generations"]


    for regime_name in regimes_to_run:
        print(f"\n--- Running {regime_name.upper()} REGIME ({num_replicates} replicates) ---")
        replicate_pop_dfs = []
        replicate_indiv_dfs = []
        for i in range(num_replicates):
            print(f"  Replicate {i+1}/{num_replicates} for {regime_name}...")
            # Create a fresh copy of configs for each replicate to ensure seed is reset if used per sim instance
            current_sim_config = CONFIG["simulation"].copy()
            current_agent_config = CONFIG["agent"].copy() # Pass correct config
            current_env_config = CONFIG["environment"].copy()

            if CONFIG["simulation"]["seed"] is not None:
                current_sim_config["seed"] = CONFIG["simulation"]["seed"] + i # Vary seed per replicate for independence

            sim_instance = Simulation(sim_config=current_sim_config,
                                      agent_config=current_agent_config, # Use copied config
                                      env_config=current_env_config,   # Use copied config
                                      env_regime=regime_name)
            df_pop_rep, df_indiv_rep = sim_instance.run_simulation()

            if not df_pop_rep.empty:
                df_pop_rep['replicate'] = i
                replicate_pop_dfs.append(df_pop_rep)
            if not df_indiv_rep.empty:
                df_indiv_rep['replicate'] = i
                replicate_indiv_dfs.append(df_indiv_rep)

        if replicate_pop_dfs:
            all_regime_pop_data[regime_name] = pd.concat(replicate_pop_dfs, ignore_index=True)
            all_regime_pop_data[regime_name].to_csv(
                os.path.join(
                    RESULTS_FOLDER,
                    f"population_log_{regime_name}_nrep{num_replicates}_ngen{num_generations_config}.csv",
                ),
                index=False,
            )
        if replicate_indiv_dfs:
            all_regime_indiv_data[regime_name] = pd.concat(replicate_indiv_dfs, ignore_index=True)
            all_regime_indiv_data[regime_name].to_csv(
                os.path.join(
                    RESULTS_FOLDER,
                    f"individual_agent_log_{regime_name}_nrep{num_replicates}_ngen{num_generations_config}.csv",
                ),
                index=False,
            )
        else:
            print(f"\n{regime_name} resulted in no population/individual data across all replicates.")


    print(f"\nRaw data saved to {RESULTS_FOLDER}")

    # --- Plotting ---
    # (Plotting code remains largely the same, ensure it uses num_generations_config for titles/filenames)
    if all_regime_pop_data:
        plot_colors = {'Scarcity': 'coral', 'Shock': 'dodgerblue', 'Flux': 'mediumseagreen'}

        # Time Series Plots
        num_metrics_left = 3 + 3
        num_metrics_right = 5
        num_rows_ts_plot = max(num_metrics_left, num_metrics_right)
        fig_ts, axes_ts = plt.subplots(num_rows_ts_plot, 2, figsize=(22, 5 * num_rows_ts_plot), sharex='col')

        for i_reg, regime_name in enumerate(regimes_to_run):
            if regime_name in all_regime_pop_data and not all_regime_pop_data[regime_name].empty:
                df_full = all_regime_pop_data[regime_name]
                df_numeric = df_full.select_dtypes(include=np.number).drop(columns=['replicate'], errors='ignore')
                df_mean_over_reps = df_numeric.groupby('generation').mean().reset_index()
                df_std_over_reps = df_numeric.groupby('generation').std().reset_index()

                generations_ts = df_mean_over_reps['generation'].values
                for metric in df_mean_over_reps.columns:
                    if metric == 'generation':
                        continue
                    series_vals = df_mean_over_reps[metric].values
                    gp, gt, gs, pv, tv, auc = compute_time_series_metrics(series_vals, generations_ts)
                    append_stat({'analysis_group': 'TimeSeries_Peaks',
                                                 'regime': regime_name,
                                                 'metric': metric,
                                                 'gen_first_peak': gp,
                                                 'gen_first_trough': gt,
                                                 'gen_slope_flip': gs,
                                                 'peak_value': pv,
                                                 'trough_value': tv,
                                                 'auc_full_run': auc})

                for k_idx, wk_label in enumerate(["w1", "w2", "w3"]):
                    ax_mean = axes_ts[k_idx, 0]; ax_std_intra = axes_ts[k_idx + 3, 0]
                    mean_col = f"mean_{wk_label}"; std_col_of_means = f"mean_{wk_label}"
                    std_intra_pop_col = f"std_{wk_label}"

                    if mean_col in df_mean_over_reps:
                        ax_mean.plot(df_mean_over_reps["generation"], df_mean_over_reps[mean_col], label=f'{regime_name}', color=plot_colors[regime_name], alpha=0.9)
                        if std_col_of_means in df_std_over_reps and pd.notna(df_std_over_reps[std_col_of_means]).all():
                             ax_mean.fill_between(df_mean_over_reps["generation"], df_mean_over_reps[mean_col]-df_std_over_reps[std_col_of_means],
                                             df_mean_over_reps[mean_col]+df_std_over_reps[std_col_of_means], color=plot_colors[regime_name], alpha=0.15)
                    ax_mean.set_ylabel(f"Mean Pop. ${wk_label}$"); ax_mean.grid(True)
                    if k_idx ==0: ax_mean.legend(title="Regime - Mean $w_k$", fontsize='small')

                    if std_intra_pop_col in df_mean_over_reps:
                        ax_std_intra.plot(df_mean_over_reps["generation"], df_mean_over_reps[std_intra_pop_col], label=f'{regime_name}', color=plot_colors[regime_name], alpha=0.9, linestyle='--')
                        if std_intra_pop_col in df_std_over_reps and pd.notna(df_std_over_reps[std_intra_pop_col]).all():
                            ax_std_intra.fill_between(df_mean_over_reps["generation"], df_mean_over_reps[std_intra_pop_col]-df_std_over_reps[std_intra_pop_col],
                                                df_mean_over_reps[std_intra_pop_col]+df_std_over_reps[std_intra_pop_col], color=plot_colors[regime_name], alpha=0.1)
                    ax_std_intra.set_ylabel(f"Mean Intra-Pop SD ${wk_label}$"); ax_std_intra.grid(True)
                    if k_idx == 0: ax_std_intra.legend(title="Regime - $w_k$ Spread", fontsize='small')


                metrics_to_plot_right = ["population_size", "mean_fcrit", "num_births", "mean_g_lever", "mean_beta_lever"]
                y_labels_right = ["Population Size", "Mean Agent $F_{crit}$", "Births per Gen", "Mean $g$ Lever", "Mean $\\beta$ Lever"]
                for r_idx, metric_name in enumerate(metrics_to_plot_right):
                    ax_r = axes_ts[r_idx, 1]
                    if metric_name in df_mean_over_reps:
                        ax_r.plot(df_mean_over_reps["generation"], df_mean_over_reps[metric_name], label=f'{regime_name}', color=plot_colors[regime_name], alpha=0.9)
                        if metric_name in df_std_over_reps and pd.notna(df_std_over_reps[metric_name]).all():
                            ax_r.fill_between(df_mean_over_reps["generation"], df_mean_over_reps[metric_name]-df_std_over_reps[metric_name],
                                              df_mean_over_reps[metric_name]+df_std_over_reps[metric_name], color=plot_colors[regime_name], alpha=0.15)
                    ax_r.set_ylabel(y_labels_right[r_idx]); ax_r.grid(True)
                    if r_idx == 0: ax_r.legend(title="Regime - Demographics/Levers", fontsize='small')
                axes_ts[len(metrics_to_plot_right)-1, 1].set_xlabel("Generation")

        fig_ts.suptitle(f"Evolution of Population $w_k$ Profiles, Demographics, and Levers (Mean $\pm$ SD over {num_replicates} Replicates, {num_generations_config} Generations)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig_ts.savefig(os.path.join(RESULTS_FOLDER, f"wk_evolution_timeseries_nrep{num_replicates}_ngen{num_generations_config}.png"), dpi=300); plt.show()

        # Ternary Plot
        if 'ternary' in sys.modules:
            figure_tern, tax_reordered = ternary.figure(scale=1.0); tax_reordered.boundary(linewidth=1.5)
            tax_reordered.gridlines(color="gray", multiple=0.2, linewidth=0.5, alpha=0.6)
            tax_reordered.set_title(f"Mean $w_k$ Trajectories & Endpoints (N Replicates={num_replicates}, {num_generations_config} Gens)", fontsize=14)

            for regime_name in regimes_to_run:
                if regime_name in all_regime_pop_data and not all_regime_pop_data[regime_name].empty:
                    df_full = all_regime_pop_data[regime_name]
                    avg_over_last_n_gens = CONFIG["simulation"]["avg_over_last_n_gens_for_stats"]
                    min_gen_for_final_avg = num_generations_config - avg_over_last_n_gens
                    actual_min_gen_for_final_avg = max(0, min_gen_for_final_avg)


                    final_wk_per_replicate = (
                        df_full[df_full['generation'] >= actual_min_gen_for_final_avg]
                        .groupby('replicate')[['mean_w1', 'mean_w2', 'mean_w3']]
                        .mean()
                    )

                    for idx, row in final_wk_per_replicate.iterrows():
                        if pd.notna(row["mean_w1"]) and pd.notna(row["mean_w2"]) and pd.notna(row["mean_w3"]):
                            point_reordered = (row["mean_w3"], row["mean_w1"], row["mean_w2"])
                            tax_reordered.scatter([point_reordered], marker='.', color=plot_colors.get(regime_name), s=30, alpha=0.3, zorder=3)

            for regime_name in regimes_to_run:
                if regime_name in all_regime_pop_data and not all_regime_pop_data[regime_name].empty:
                    df_full = all_regime_pop_data[regime_name]
                    df_numeric_traj = df_full.select_dtypes(include=np.number).drop(columns=['replicate'], errors='ignore')
                    df_mean_traj = df_numeric_traj.groupby('generation')[['mean_w1', 'mean_w2', 'mean_w3']].mean().reset_index()

                    if not df_mean_traj.empty and pd.notna(df_mean_traj["mean_w1"].iloc[0]):
                        path_points_raw = df_mean_traj[['mean_w1', 'mean_w2', 'mean_w3']].dropna().values
                        if len(path_points_raw) > 0:
                            path_points_reordered = [(p[2], p[0], p[1]) for p in path_points_raw]
                            tax_reordered.plot(path_points_reordered, linewidth=2.5, color=plot_colors.get(regime_name), linestyle="-", alpha=0.8, zorder=5, label=f"{regime_name} Traj.")
                            tax_reordered.scatter([path_points_reordered[0]], marker='s', color=plot_colors.get(regime_name), s=70, edgecolors='k', alpha=0.9, zorder=8, label=f"{regime_name} Start")
                            tax_reordered.scatter([path_points_reordered[-1]], marker='o', color=plot_colors.get(regime_name), label=f"{regime_name} End", s=180, edgecolors='k', zorder=10)

                            path_len, net_ang = compute_ternary_path_metrics(path_points_raw)
                            append_stat({'analysis_group': 'Ternary_Path', 'regime': regime_name, 'path_length': path_len, 'net_angle_deg': net_ang})

            start_dummy = (1/3, 1/3, 1/3)
            tax_reordered.scatter([start_dummy], marker='s', color='lightgray', edgecolors='k', s=70, alpha=0.0, label='Start Point')
            tax_reordered.legend(title="Regime", fontsize='small', loc='center left', bbox_to_anchor=(1.05, 0.5))
            tax_reordered.ticks(axis='lbr', linewidth=1, multiple=0.2, tick_formats="%.1f", offset=0.02, fontsize=8)
            tax_reordered.get_axes().axis('off'); tax_reordered.clear_matplotlib_ticks()
            tax_reordered.right_corner_label("$w_3$ (Slack)", fontsize=11); tax_reordered.top_corner_label("$w_1$ (Perception)", fontsize=11); tax_reordered.left_corner_label("$w_2$ (Precision)", fontsize=11)
            plt.tight_layout(); figure_tern.savefig(os.path.join(RESULTS_FOLDER, f"wk_evolution_ternary_nrep{num_replicates}_ngen{num_generations_config}.png"), dpi=300, bbox_inches='tight'); plt.show()

        # KDE plots
        log_wk_dist_gens_plot = CONFIG["simulation"]["log_wk_dist_gens"]
        if log_wk_dist_gens_plot:
            for gen_snapshot in log_wk_dist_gens_plot:
                if gen_snapshot >= num_generations_config:
                    print(f"Skipping KDE plot for gen {gen_snapshot} as it exceeds num_generations {num_generations_config}")
                    continue
                fig_kde, axes_kde = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
                fig_kde.suptitle(f"$w_k$ Population Distributions at Generation {gen_snapshot} (All Replicates Pooled, N Replicates={num_replicates}, {num_generations_config} Total Gens)", fontsize=16)
                for i_wk, wk_label_short in enumerate(["w1", "w2", "w3"]):
                    ax_curr = axes_kde[i_wk]
                    for regime_name in regimes_to_run:
                        if regime_name in all_regime_pop_data and not all_regime_pop_data[regime_name].empty:
                            df_full = all_regime_pop_data[regime_name]
                            snapshot_data_series = df_full[df_full["generation"] == gen_snapshot]["wk_dist_snapshot"].dropna()
                            if not snapshot_data_series.empty:
                                all_wk_values_for_snapshot = [prof[i_wk] for rep_list in snapshot_data_series for prof in rep_list]
                                if all_wk_values_for_snapshot:
                                     sns.kdeplot(all_wk_values_for_snapshot, ax=ax_curr, label=f"{regime_name}",
                                                 color=plot_colors[regime_name], fill=True, alpha=0.3, bw_adjust=0.75)
                                     sk, ku, q05, q25, q50, q75, q95 = wk_snapshot_stats(all_wk_values_for_snapshot)
                                     append_stat({'analysis_group': 'KDE_Snapshot',
                                                                    'regime': regime_name,
                                                                    'generation': gen_snapshot,
                                                                    'metric': wk_label_short,
                                                                    'skewness': sk,
                                                                    'kurtosis': ku,
                                                                    'q05': q05,
                                                                    'q25': q25,
                                                                    'q50': q50,
                                                                    'q75': q75,
                                                                    'q95': q95})
                    ax_curr.set_title(f"Distribution of ${wk_label_short}$"); ax_curr.set_xlabel(f"${wk_label_short}$ value")
                    ax_curr.legend(fontsize='small'); ax_curr.grid(True); ax_curr.set_xlim(0,1)
                axes_kde[0].set_ylabel("Density"); plt.tight_layout(rect=[0, 0, 1, 0.95])
                fig_kde.savefig(os.path.join(RESULTS_FOLDER, f"wk_dist_kde_gen_{gen_snapshot}_nrep{num_replicates}_ngen{num_generations_config}.png"), dpi=300); plt.show()

        # --- Statistical Tests ---
        # (Statistical test logging code from previous response goes here)
        # Statistical Tests for Final Mean w_k
        final_gen_data_for_stats = {}; avg_over_last_n_gens = CONFIG["simulation"]["avg_over_last_n_gens_for_stats"]
        min_gens_for_avg = num_generations_config - avg_over_last_n_gens
        print_header_final_mean = f"\n--- Statistical Tests for Final Mean $w_k$ (Averaged over last {avg_over_last_n_gens} generations of {num_generations_config} total) ---"
        print(print_header_final_mean)
        append_stat({'analysis_group': 'Header', 'detail': print_header_final_mean.strip()})

        for regime_name in regimes_to_run:
            if regime_name in all_regime_pop_data:
                df_full = all_regime_pop_data[regime_name]
                actual_min_gens_for_avg = max(0, min_gens_for_avg)
                df_to_avg = df_full[df_full['generation'] >= actual_min_gens_for_avg]
                if not df_to_avg.empty:
                    numeric_cols = df_to_avg.select_dtypes(include=np.number).drop(columns=['replicate'], errors='ignore').columns
                    cols_for_final_avg = [col for col in ['mean_w1', 'mean_w2', 'mean_w3'] if col in numeric_cols]
                    if cols_for_final_avg:
                        final_vals_per_replicate = df_to_avg.groupby('replicate')[cols_for_final_avg].mean()
                        final_gen_data_for_stats[regime_name] = final_vals_per_replicate
                        for col in cols_for_final_avg:
                            vals = final_vals_per_replicate[col].dropna().values
                            if len(vals) > 0:
                                mean_val = float(np.mean(vals))
                                if len(vals) > 1:
                                    ci_low, ci_high = stats.t.interval(0.95, len(vals)-1, loc=mean_val, scale=stats.sem(vals))
                                else:
                                    ci_low = ci_high = mean_val
                                append_stat({'analysis_group': 'Final_Mean_Wk_Endpoint',
                                                               'regime': regime_name,
                                                               'metric': col,
                                                               'mean': mean_val,
                                                               'ci95_low': ci_low,
                                                               'ci95_high': ci_high,
                                                               'N_replicates': len(vals)})
        for k_idx, wk_label in enumerate(["w1", "w2", "w3"]):
            mean_col_name = f"mean_{wk_label}"
            print_line = f"\nTesting for significant differences in {mean_col_name}:"
            print(print_line)
            append_stat({'analysis_group': 'Final_Mean_Wk_Overall_Header', 'metric': mean_col_name, 'detail': print_line.strip()})
            data_groups = []; group_names_for_test = []
            for r_idx_test, r_name_test in enumerate(regimes_to_run):
                if r_name_test in final_gen_data_for_stats and not final_gen_data_for_stats[r_name_test].empty and mean_col_name in final_gen_data_for_stats[r_name_test]:
                    group_data = final_gen_data_for_stats[r_name_test][mean_col_name].dropna().values
                    if len(group_data) >= 2: data_groups.append(group_data); group_names_for_test.append(r_name_test)
            if len(data_groups) < 2:
                no_data_print = f"  Not enough data groups for tests on {mean_col_name}."
                print(no_data_print); append_stat({'analysis_group': 'Final_Mean_Wk_Overall', 'metric': mean_col_name, 'notes': no_data_print.strip()}); continue
            if len(data_groups) >= 2:
                f_val, p_val_anova = stats.f_oneway(*data_groups)
                grand_mean = np.mean(np.concatenate(data_groups)); ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in data_groups)
                ss_total = sum(((g - grand_mean) ** 2).sum() for g in data_groups); eta_sq = ss_between / ss_total if ss_total > 1e-9 else 0.0
                df_total = sum(len(g) for g in data_groups) - 1
                anova_print = f"  ANOVA for {mean_col_name}: F={f_val:.3f}, p={p_val_anova:.4f}, eta^2={eta_sq:.3f}"; print(anova_print)
                append_stat({'analysis_group': 'Final_Mean_Wk_Overall', 'metric': mean_col_name, 'test_type': 'ANOVA', 'statistic_F': f_val, 'p_value': p_val_anova, 'eta_squared': eta_sq, 'df_total': df_total})
                pairwise_pvals = {}
                if p_val_anova < 0.05 and len(data_groups) > 1:
                    pairwise_header_print = f"  Significant difference. Pairwise Welch's t-tests (Bonferroni corrected alpha):"; print(pairwise_header_print)
                    num_comparisons = len(list(combinations(range(len(data_groups)), 2))); alpha_corrected = 0.05 / max(1, num_comparisons)
                    for r1_idx_combo, r2_idx_combo in combinations(range(len(data_groups)), 2):
                        r1_name_combo = group_names_for_test[r1_idx_combo]; r2_name_combo = group_names_for_test[r2_idx_combo]
                        data1_combo = data_groups[r1_idx_combo]; data2_combo = data_groups[r2_idx_combo]
                        if len(data1_combo) <2 or len(data2_combo) < 2: continue
                        t_stat, p_val_ttest = stats.ttest_ind(data1_combo, data2_combo, equal_var=False)
                        pooled_std_val = 0; cohens_d = 0.0
                        if (data1_combo.size + data2_combo.size - 2) > 0: pooled_std_val = np.sqrt(((data1_combo.size - 1) * np.var(data1_combo, ddof=1) + (data2_combo.size - 1) * np.var(data2_combo, ddof=1)) / (data1_combo.size + data2_combo.size - 2))
                        if pooled_std_val > 1e-9: cohens_d = (np.mean(data1_combo) - np.mean(data2_combo)) / pooled_std_val
                        s1_sq = np.var(data1_combo, ddof=1); s2_sq = np.var(data2_combo, ddof=1)
                        df_num = (s1_sq/len(data1_combo) + s2_sq/len(data2_combo))**2
                        df_denom = (s1_sq**2/((len(data1_combo)**2)*(len(data1_combo)-1))) + (s2_sq**2/((len(data2_combo)**2)*(len(data2_combo)-1)))
                        df_val = df_num/df_denom if df_denom > 0 else len(data1_combo)+len(data2_combo)-2
                        s_marker = " ***" if p_val_ttest < 0.001 else (" **" if p_val_ttest < 0.01 else (" *" if p_val_ttest < 0.05 else "")); bc_sig = p_val_ttest < alpha_corrected; bc_marker = " (BC sig.)" if bc_sig else ""
                        pairwise_print = f"    {r1_name_combo} vs {r2_name_combo}: t={t_stat:.3f}, p={p_val_ttest:.4f}{s_marker}{bc_marker}, d={cohens_d:.3f}"; print(pairwise_print)
                        append_stat({'analysis_group': 'Final_Mean_Wk_Pairwise', 'metric': mean_col_name, 'comparison_group1': r1_name_combo, 'comparison_group2': r2_name_combo, 'test_type': "Welch's t-test", 'statistic_t': t_stat, 'p_value': p_val_ttest, 'cohens_d': cohens_d, 'significance_marker': s_marker.strip(), 'bonferroni_alpha': alpha_corrected, 'bonferroni_significant': bc_sig, 'df': df_val})
                        pairwise_pvals[f"{r1_name_combo}-{r2_name_combo}"] = p_val_ttest
                elif p_val_anova >=0.05 : print(f"  No significant overall difference for {mean_col_name} via ANOVA.")
                SIM_OUTPUT["group_tests"].append({"metric": mean_col_name, "F": f_val, "p": p_val_anova, "eta_sq": eta_sq, "pairwise": _clean_for_json(pairwise_pvals)})

        # Analysis of Total Change Scores (Delta w_k)
        print_header_delta_wk = "\n--- Statistical Tests for Total Change Scores (Delta w_k) ---"; print(print_header_delta_wk)
        append_stat({'analysis_group': 'Header', 'detail': print_header_delta_wk.strip()})
        delta_wk_per_regime = {}
        for regime_name in regimes_to_run:
            if regime_name in all_regime_pop_data:
                df_full = all_regime_pop_data[regime_name]; reps = df_full['replicate'].unique(); delta_records = []
                for rep_idx, rep in enumerate(reps):
                    df_rep = df_full[df_full['replicate'] == rep];                    
                    if df_rep.empty: continue
                    init_gen_data = df_rep[df_rep['generation'] == 0]
                    actual_min_gens_for_avg_delta = max(0, num_generations_config - avg_over_last_n_gens)
                    final_gen_data = df_rep[df_rep['generation'] >= actual_min_gens_for_avg_delta]
                    if init_gen_data.empty or final_gen_data.empty: continue
                    init_vals = init_gen_data[['mean_w1', 'mean_w2', 'mean_w3']].mean()
                    final_vals = final_gen_data[['mean_w1', 'mean_w2', 'mean_w3']].mean()
                    if not init_vals.isna().any() and not final_vals.isna().any(): delta_records.append((final_vals - init_vals).values)
                if delta_records: delta_wk_per_regime[regime_name] = np.array(delta_records)
        for k_idx, wk_label in enumerate(["w1", "w2", "w3"]):
            delta_metric_name = f"delta_{wk_label}"; print_line_delta = f"\nTesting for significant differences in {delta_metric_name}:"; print(print_line_delta)
            append_stat({'analysis_group': 'Delta_Wk_Overall_Header', 'metric': delta_metric_name, 'detail': print_line_delta.strip()})
            delta_groups = []; group_names_for_delta_test = []
            for r_name_test in regimes_to_run:
                if r_name_test in delta_wk_per_regime and delta_wk_per_regime[r_name_test].shape[0] > 0:
                    group_data = delta_wk_per_regime[r_name_test][:, k_idx]
                    if len(group_data) >= 2: delta_groups.append(group_data); group_names_for_delta_test.append(r_name_test)
                    mean_val = float(np.mean(group_data))
                    if len(group_data) > 1:
                        ci_low, ci_high = stats.t.interval(0.95, len(group_data)-1, loc=mean_val, scale=stats.sem(group_data))
                    else:
                        ci_low = ci_high = mean_val
                    append_stat({'analysis_group': 'Delta_Wk_Endpoint',
                                                   'regime': r_name_test,
                                                   'metric': delta_metric_name,
                                                   'mean': mean_val,
                                                   'ci95_low': ci_low,
                                                   'ci95_high': ci_high,
                                                   'N_replicates': len(group_data)})
            if len(delta_groups) < 2: no_data_delta_print = f"  Not enough data groups for tests on {delta_metric_name}."; print(no_data_delta_print); append_stat({'analysis_group': 'Delta_Wk_Overall', 'metric': delta_metric_name, 'notes': no_data_delta_print.strip()}); continue
            f_val_delta, p_val_anova_delta = stats.f_oneway(*delta_groups)
            grand_mean_delta = np.mean(np.concatenate(delta_groups)); ss_between_delta = sum(len(g) * (np.mean(g) - grand_mean_delta) ** 2 for g in delta_groups)
            ss_total_delta = sum(((g - grand_mean_delta) ** 2).sum() for g in delta_groups); eta_sq_delta = ss_between_delta / ss_total_delta if ss_total_delta > 1e-9 else 0.0
            df_total_delta = sum(len(g) for g in delta_groups) - 1
            anova_delta_print = f"  ANOVA for {delta_metric_name}: F={f_val_delta:.3f}, p={p_val_anova_delta:.4f}, eta^2={eta_sq_delta:.3f}"; print(anova_delta_print)
            append_stat({'analysis_group': 'Delta_Wk_Overall', 'metric': delta_metric_name, 'test_type': 'ANOVA', 'statistic_F': f_val_delta, 'p_value': p_val_anova_delta, 'eta_squared': eta_sq_delta, 'df_total': df_total_delta})
            pairwise_delta_pvals = {}
            if p_val_anova_delta < 0.05 and len(delta_groups) > 1:
                pairwise_delta_header = f"  Significant difference. Pairwise Welch's t-tests (Bonferroni corrected alpha):"; print(pairwise_delta_header)
                num_comparisons_delta = len(list(combinations(range(len(delta_groups)), 2))); alpha_corrected_delta = 0.05 / max(1, num_comparisons_delta)
                for r1_idx_combo, r2_idx_combo in combinations(range(len(delta_groups)), 2):
                    r1_name_delta = group_names_for_delta_test[r1_idx_combo]; r2_name_delta = group_names_for_delta_test[r2_idx_combo]
                    data1_delta = delta_groups[r1_idx_combo]; data2_delta = delta_groups[r2_idx_combo]
                    if len(data1_delta) < 2 or len(data2_delta) < 2: continue
                    t_stat_delta, p_val_ttest_delta = stats.ttest_ind(data1_delta, data2_delta, equal_var=False)
                    pooled_std_delta_val = 0; cohens_d_delta = 0.0
                    if (data1_delta.size + data2_delta.size - 2) > 0: pooled_std_delta_val = np.sqrt(((data1_delta.size-1)*np.var(data1_delta, ddof=1) + (data2_delta.size-1)*np.var(data2_delta, ddof=1)) / (data1_delta.size + data2_delta.size - 2))
                    if pooled_std_delta_val > 1e-9: cohens_d_delta = (np.mean(data1_delta) - np.mean(data2_delta)) / pooled_std_delta_val
                    s1_sq_delta = np.var(data1_delta, ddof=1); s2_sq_delta = np.var(data2_delta, ddof=1)
                    df_num_delta = (s1_sq_delta/len(data1_delta) + s2_sq_delta/len(data2_delta))**2
                    df_denom_delta = (s1_sq_delta**2/((len(data1_delta)**2)*(len(data1_delta)-1))) + (s2_sq_delta**2/((len(data2_delta)**2)*(len(data2_delta)-1)))
                    df_val_delta = df_num_delta/df_denom_delta if df_denom_delta > 0 else len(data1_delta)+len(data2_delta)-2
                    s_marker_delta = " ***" if p_val_ttest_delta < 0.001 else (" **" if p_val_ttest_delta < 0.01 else (" *" if p_val_ttest_delta < 0.05 else "")); bc_sig_delta = p_val_ttest_delta < alpha_corrected_delta; bc_marker_delta = " (BC sig.)" if bc_sig_delta else ""
                    pairwise_delta_print = f"    {r1_name_delta} vs {r2_name_delta}: t={t_stat_delta:.3f}, p={p_val_ttest_delta:.4f}{s_marker_delta}{bc_marker_delta}, d={cohens_d_delta:.3f}"; print(pairwise_delta_print)
                    append_stat({'analysis_group': 'Delta_Wk_Pairwise', 'metric': delta_metric_name, 'comparison_group1': r1_name_delta, 'comparison_group2': r2_name_delta, 'test_type': "Welch's t-test", 'statistic_t': t_stat_delta, 'p_value': p_val_ttest_delta, 'cohens_d': cohens_d_delta, 'significance_marker': s_marker_delta.strip(), 'bonferroni_alpha': alpha_corrected_delta, 'bonferroni_significant': bc_sig_delta, 'df': df_val_delta})
                    pairwise_delta_pvals[f"{r1_name_delta}-{r2_name_delta}"] = p_val_ttest_delta
            elif p_val_anova_delta >= 0.05: print(f"  No significant overall difference for {delta_metric_name} via ANOVA.")
            SIM_OUTPUT["group_tests"].append({"metric": delta_metric_name, "F": f_val_delta, "p": p_val_anova_delta, "eta_sq": eta_sq_delta, "pairwise": _clean_for_json(pairwise_delta_pvals)})

        # Trend Analysis
        print_header_trend = "\n--- Longitudinal Trend Analysis (Mean $w_k$ vs. Generation) ---"; print(print_header_trend)
        append_stat({'analysis_group': 'Header', 'detail': print_header_trend.strip()})
        for regime_name in regimes_to_run:
            if regime_name in all_regime_pop_data and not all_regime_pop_data[regime_name].empty:
                regime_header_print = f"\nTrends for {regime_name} Regime:"; print(regime_header_print)
                append_stat({'analysis_group': 'Longitudinal_Trend_Header', 'regime': regime_name, 'detail': regime_header_print.strip()})
                df_full = all_regime_pop_data[regime_name]
                df_mean_over_reps = df_full.select_dtypes(include=np.number).drop(columns=['replicate'], errors='ignore').groupby('generation').mean().reset_index()
                if df_mean_over_reps.empty or len(df_mean_over_reps["generation"]) < 2 : print(f"  Not enough data points for trend analysis in {regime_name}"); append_stat({'analysis_group': 'Longitudinal_Trend', 'regime': regime_name, 'notes': "Not enough data points for trend analysis"}); continue
                generations = df_mean_over_reps["generation"].values
                for k_idx, wk_label in enumerate(["w1", "w2", "w3"]):
                    mean_col = f"mean_{wk_label}"
                    if mean_col in df_mean_over_reps and df_mean_over_reps[mean_col].notna().sum() > 1:
                        wk_values = df_mean_over_reps[mean_col].values; valid_indices = np.isfinite(generations) & np.isfinite(wk_values)
                        if valid_indices.sum() < 2: print(f"  Not enough valid (finite) data points for trend on {mean_col} in {regime_name}"); append_stat({'analysis_group': 'Longitudinal_Trend', 'regime': regime_name, 'metric': mean_col, 'notes': "Not enough valid (finite) data points"}); continue
                        slope, intercept, r_value, p_value, std_err = stats.linregress(generations[valid_indices], wk_values[valid_indices])
                        first_val = wk_values[valid_indices][0] if len(wk_values[valid_indices]) > 0 else np.nan; last_val = wk_values[valid_indices][-1] if len(wk_values[valid_indices]) > 0 else np.nan
                        s_marker = " ***" if p_value < 0.001 else (" **" if p_value < 0.01 else (" *" if p_value < 0.05 else ""))
                        trend_print = f"  Trend for {mean_col}: slope={slope:.5f}, p-value={p_value:.4f}{s_marker} | first={first_val:.3f}, last={last_val:.3f}"; print(trend_print)
                        append_stat({'analysis_group': 'Longitudinal_Trend', 'regime': regime_name, 'metric': mean_col, 'statistic_slope': slope, 'p_value': p_value, 'r_value': r_value, 'std_err': std_err, 'intercept': intercept, 'significance_marker': s_marker.strip(), 'first_value': first_val, 'last_value': last_val})
                    else: print(f"  Not enough valid data or column {mean_col} missing for trend analysis in {regime_name}"); append_stat({'analysis_group': 'Longitudinal_Trend', 'regime': regime_name, 'metric': mean_col, 'notes': f"Not enough valid data or column missing"})

        # Fitness Analysis
        print_header_fitness = "\n--- Fitness Analysis (Correlation of $w_k$ with Fitness Proxies) ---"; print(print_header_fitness)
        append_stat({'analysis_group': 'Header', 'detail': print_header_fitness.strip()})
        if CONFIG["simulation"]["log_individual_agent_fitness_interval"] > 0:
            for regime_name in regimes_to_run:
                if regime_name in all_regime_indiv_data and not all_regime_indiv_data[regime_name].empty:
                    regime_fitness_header = f"\nFitness analysis for {regime_name} Regime:"; print(regime_fitness_header)
                    append_stat({'analysis_group': 'Fitness_Correlation_Header', 'regime': regime_name, 'detail': regime_fitness_header.strip()})
                    df_indiv = all_regime_indiv_data[regime_name].copy()
                    df_indiv.dropna(subset=['w1','w2','w3','total_fcrit_gathered_total','offspring_count_total', 'final_age_at_log'], inplace=True)
                    num_gens_total_fitness = num_generations_config
                    bins = [0, num_gens_total_fitness // 3, 2 * num_gens_total_fitness // 3, num_gens_total_fitness + 1]; labels = ["Early", "Mid", "Late"]
                    df_indiv['birth_generation'] = pd.to_numeric(df_indiv['birth_generation'], errors='coerce'); df_indiv.dropna(subset=['birth_generation'], inplace=True)
                    df_indiv['gen_band'] = pd.cut(df_indiv['birth_generation'], bins=bins, labels=labels, right=False, include_lowest=True)
                    df_fitness_analysis = df_indiv.copy()
                    if len(df_fitness_analysis) >= 20:
                        analyzing_print = f"  Analyzing {len(df_fitness_analysis)} individual agent records for fitness in {regime_name}."; print(analyzing_print)
                        append_stat({'analysis_group': 'Fitness_Correlation_Setup', 'regime': regime_name, 'num_records': len(df_fitness_analysis), 'detail': analyzing_print.strip()})
                        for i_wk_fit, wk_label_short_fit in enumerate(["w1", "w2", "w3"]):
                            for band in labels:
                                df_band = df_fitness_analysis[df_fitness_analysis["gen_band"] == band]
                                if df_band.empty or len(df_band) < 2: continue
                                wk_vals = pd.to_numeric(df_band[wk_label_short_fit], errors="coerce"); gathered_vals = pd.to_numeric(df_band["total_fcrit_gathered_total"], errors="coerce")
                                offspring_vals = pd.to_numeric(df_band["offspring_count_total"], errors="coerce"); age_vals = pd.to_numeric(df_band["final_age_at_log"], errors="coerce")
                                for fitness_proxy_name, fitness_values in [("Total_Fcrit_Gathered", gathered_vals), ("Offspring_Count", offspring_vals), ("Lifespan_Age", age_vals)]:
                                    valid_idx = wk_vals.notna() & fitness_values.notna()
                                    if valid_idx.sum() > 1:
                                        corr, p_val_corr = stats.pearsonr(wk_vals[valid_idx], fitness_values[valid_idx])
                                        fitness_corr_print = f"    {regime_name} - {wk_label_short_fit} vs {fitness_proxy_name} ({band}): r={corr:.3f}, p={p_val_corr:.3f}, N={valid_idx.sum()}"; print(fitness_corr_print)
                                        append_stat({'analysis_group': 'Fitness_Correlation', 'regime': regime_name, 'independent_var': wk_label_short_fit, 'dependent_var': fitness_proxy_name, 'generational_band': band, 'statistic_pearson_r': corr, 'p_value': p_val_corr, 'N_pairs': int(valid_idx.sum())})
                    else: not_enough_fitness_data_print = f"  Not enough individual agent data (N={len(df_fitness_analysis)}) for robust fitness correlation in {regime_name} regime."; print(not_enough_fitness_data_print); append_stat({'analysis_group': 'Fitness_Correlation_Setup', 'regime': regime_name, 'num_records': len(df_fitness_analysis), 'notes': not_enough_fitness_data_print.strip()})
                else: no_indiv_data_print = f"  No individual agent data logged for {regime_name} for fitness analysis."; print(no_indiv_data_print); append_stat({'analysis_group': 'Fitness_Correlation_Setup', 'regime': regime_name, 'notes': no_indiv_data_print.strip()})
            regime_for_plot = "Shock" # Example for scatter
            if regime_for_plot in all_regime_indiv_data and not all_regime_indiv_data[regime_for_plot].empty:
                df_scatter = all_regime_indiv_data[regime_for_plot].copy(); df_scatter.dropna(subset=['w3', 'total_fcrit_gathered_total', 'birth_generation'], inplace=True)
                df_scatter['birth_generation'] = pd.to_numeric(df_scatter['birth_generation'], errors='coerce'); df_scatter.dropna(subset=['birth_generation'], inplace=True)
                num_gens_total_scatter = num_generations_config; bins_scatter = [0, num_gens_total_scatter // 3, 2 * num_gens_total_scatter // 3, num_gens_total_scatter + 1]; labels_scatter = ["Early", "Mid", "Late"]
                if not df_scatter.empty:
                    df_scatter['gen_band'] = pd.cut(df_scatter['birth_generation'], bins=bins_scatter, labels=labels_scatter, right=False, include_lowest=True)
                    plt.figure(figsize=(8,6)); sns.scatterplot(data=df_scatter, x='w3', y='total_fcrit_gathered_total', hue='gen_band', alpha=0.6, hue_order=["Early", "Mid", "Late"])
                    for band_scatter in labels_scatter:
                        df_band_scatter = df_scatter[df_scatter['gen_band'] == band_scatter]
                        if len(df_band_scatter) > 1:
                            x_data = pd.to_numeric(df_band_scatter['w3'], errors='coerce'); y_data = pd.to_numeric(df_band_scatter['total_fcrit_gathered_total'], errors='coerce'); valid_scatter_indices = np.isfinite(x_data) & np.isfinite(y_data)
                            if valid_scatter_indices.sum() > 1:
                                slope, intercept, _, _, _ = stats.linregress(x_data[valid_scatter_indices], y_data[valid_scatter_indices])
                                x_vals_plot = np.array([x_data[valid_scatter_indices].min(), x_data[valid_scatter_indices].max()]); plt.plot(x_vals_plot, intercept + slope * x_vals_plot, label=f'{band_scatter} fit')
                    plt.title(f'{regime_for_plot} Regime: w3 vs Total Fcrit Gathered (N Replicates={num_replicates}, {num_generations_config} Gens)'); plt.xlabel('w3'); plt.ylabel('Total Fcrit Gathered'); plt.legend(); plt.tight_layout()
                    plt.savefig(os.path.join(RESULTS_FOLDER, f"fitness_scatter_{regime_for_plot.lower()}_w3_vs_fcrit_nrep{num_replicates}_ngen{num_generations_config}.png"), dpi=300); plt.show()
        else: no_fitness_log_print = "Individual agent fitness logging was not enabled sufficiently for analysis."; print(no_fitness_log_print); append_stat({'analysis_group': 'Fitness_Correlation_Setup', 'notes': no_fitness_log_print.strip()})
    else: print("No simulation data was generated for plotting.")

    if SIM_OUTPUT["statistics"]:
        filtered_stats = [
            e for e in SIM_OUTPUT["statistics"]
            if e.get("metric") not in {"wk_dist_snapshot", "mean_g_lever", "mean_beta_lever"}
        ]
        SIM_OUTPUT["statistics"] = filtered_stats
        summary_df = pd.DataFrame(filtered_stats)
        summary_csv_path = os.path.join(RESULTS_FOLDER, f"statistical_summary_nrep{num_replicates}_ngen{num_generations_config}.csv")
        try:
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"\nStatistical summary saved to {summary_csv_path}")
        except Exception as e:
            print(f"\nError saving statistical summary to CSV: {e}")
            summary_txt_path = os.path.join(RESULTS_FOLDER, f"statistical_summary_nrep{num_replicates}_ngen{num_generations_config}.txt")
            with open(summary_txt_path, 'w') as f:
                for item in SIM_OUTPUT["statistics"]:
                    f.write(str(item) + '\n')
            print(f"Statistical summary (fallback) saved to {summary_txt_path}")
        summary_json_path = os.path.join(RESULTS_FOLDER, f"statistical_summary_nrep{num_replicates}_ngen{num_generations_config}.json")
        try:
            with open(summary_json_path, 'w', encoding='utf-8') as f_json:
                json.dump(SIM_OUTPUT["statistics"], f_json, indent=2, ensure_ascii=False)
            print(f"Statistical summary saved to {summary_json_path}")
        except Exception as e:
            print(f"Error saving summary to JSON: {e}")
    else:
        print("\nNo summary statistics were generated to save to CSV.")

    # Write consolidated JSON summary of config, logs and statistics
    json_path = os.path.join(RESULTS_FOLDER, f"summary_nrep{num_replicates}_ngen{num_generations_config}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_clean_for_json(SIM_OUTPUT), f, indent=2, ensure_ascii=False)
    print(f"Full simulation summary written to {json_path}")
