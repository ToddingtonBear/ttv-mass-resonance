# -*- coding: utf-8 -*-
"""
Updated since presentation. The main updates are:
- reducing the size of the neural network model
- also incorporating a Boosted Decision Tree
- plotting loss curves


To use the script, just pop into your environment of choice and run.
You can change NUM_SIMULATIONS, MAX_TRANSITS or USE_RESONANCES if you wish
"""

import numpy as np
import rebound
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from astropy.timeseries import LombScargle

#-----------------------------------------------------------------------------
# Script Variables & Physics Constants
#-----------------------------------------------------------------------------
NUM_SIMULATIONS = 1000
MAX_TRANSITS = 100
USE_RESONANCES = True    # for period ratio selection
# if True, select from range of set resonances +/- 0.07
# if False, select random number between 1.25 and 3.0

SOLAR_TO_EARTH = 332946.0
# Assuming 2*pi in simulation = 365.25 days
SIM_TIME_TO_DAYS = 365.25 / (2 * np.pi)

#-----------------------------------------------------------------------------
# Utility Functions
#-----------------------------------------------------------------------------

def detect_transits(sim, integration_time, dt, max_transits, planet_index):
    """
    Detects the times when a specified planet transits the star (crosses the y-axis
    while in front of the star) within a REBOUND simulation.

    Inputs:
        :param sim: the simulation being run
        :param integration_time: the 'time' the sim is to run for
        :param dt: the integration time step
        :param max_transits: max number of transits we want to detect
        :param planet_index: index of the planet we want to detect transits of
    Outputs:
        array of transit times
    """
    transit_times = []
    star = sim.particles[0]             # star is always the first element in our sim (or should be!)
    planet = sim.particles[planet_index]# specify the index of the planet in our sim whose transits we want to detect
    prev_y = planet.y - star.y

    time = 0
    while time < integration_time and len(transit_times) < max_transits:
        sim.integrate(sim.t + dt)           # progress simulation
        time = sim.t                        # get time from sim
        curr_y = planet.y - star.y
        # if the product of previous and current y is negative, they they are on
        # opposite sides of the y axis, and so there has been a transit of the y-axis
        # if (planet.x - star.x) > 0, then the planet is in front of the star: transit occurred
        if prev_y * curr_y < 0 and (planet.x - star.x) > 0:
            # perform linear interpolation to find more precise moment of crossing
            weight = -prev_y / (curr_y - prev_y)
            precise_time = (time - dt) + (weight * dt)
            transit_times.append(precise_time)
        prev_y = curr_y
    return np.array(transit_times)


def detrend_ttv(ttv_data):
    """
    Remove the linear trend from a single TTV array

    Inputs:
        array: ttv data
    Outputs:
        array: detrended ttv data
    """
    N = len(ttv_data)                                   # get length of input
    transit_numbers = np.arange(N)                      # create array of transit numbers
    P = np.polyfit(transit_numbers, ttv_data, 1)        # fit straight line to ttv data
    return ttv_data - (P[0] * transit_numbers + P[1])   # subtract line from ttv and return

#-----------------------------------------------------------------------------
# PHASE 1: GENERATE DATA WITH PERIODOGRAM
#-----------------------------------------------------------------------------

def generate_simulation_data(num_simulations, max_transits):
    """
    Generate the simulation data to train our model
    Inputs:
        :param num_simulations: number of simulations to be run
        :param max_transits: number of transits to reach before terminating a sim
    Outputs:
        array of physical features
        array of masses
        array of TTVs
    """
    print(f"Generating {num_simulations} simulations...")
    all_features = []
    all_masses = []
    all_ttvs = []

    # Use strong resonances to ensure a learnable signal
    resonances = [1.5, 2.0, 3.0]
    # as an example, a resonance ration of 1.5, or 3/2, would mean planet B's period is
    # 1.5 times planet C's, or put another way, by the time planet B has orbited three times,
    # planet C has orbited only twice

    for i in range(num_simulations):
        if (i + 1) % 50 == 0: print(f"  Simulation {i+1}...")

        try:
            # Randomize Planet B (The Transitor)
            m_b = np.random.uniform(1e-6, 1e-4) # mass (earth range to jupiter range)
            P_b = np.random.uniform(2*np.pi, 4*np.pi)   # period
            e_b = np.random.uniform(0.0, 0.02)  # eccentricity

            # Randomize Planet C (The Perturber)
            # ratio = period C / period B
            # define if setting period ratio from resonance array or set range
            if USE_RESONANCES:
                ratio = np.random.choice(resonances) + np.random.uniform(-0.07, 0.07)
            else:
                ratio = np.random.uniform(1.25, 3)
            m_c = np.random.uniform(1e-4, 5e-3) # mass
            P_c = P_b * ratio                   # period
            e_c = np.random.uniform(0.0, 0.05)  # eccentricity

            dt = P_b / 50   # time step to be used by integrators

            # Unperturbed Simulation
            sim_u = rebound.Simulation()    # create simulation
            sim_u.integrator = "whfast";    # specify the integrator
            sim_u.dt = dt                   # set time step of integrator
            sim_u.add(m=1.0);               # add 1 solar mass star to sim
            sim_u.add(m=m_b, P=P_b, e=e_b)  # add planet B to sim
            sim_u.move_to_com()             # set centre of mass of system as the origin
            transits_u = detect_transits(   # in unperturbed sim, find transits of planet B
                sim_u,                  # simulation to use
                P_b*(max_transits+5),   # integration time
                dt,                     # integrator time step
                max_transits,           # max number of transits we're looking for
                1                       # index of transiting planet in simulation
            )

            # Perturbed Simulation
            sim_p = rebound.Simulation()    # create simulation
            sim_p.integrator = "whfast";    # specify the integrator
            sim_p.dt = dt                   # set time step of integrator
            sim_p.add(m=1.0);               # add 1 solar mass star to sim
            sim_p.add(m=m_b, P=P_b, e=e_b)  # add planet B to sim
            sim_p.add(m=m_c, P=P_c, e=e_c)  # in unperturbed sim, find transits of planet B
            sim_p.move_to_com()             # set centre of mass of system as the origin
            transits_p = detect_transits(
                sim_p,                  # simulation to use
                P_b*(max_transits+5),   # integration time
                dt,                     # integrator time step
                max_transits,           # max number of transits we're looking for
                1                       # index of transiting planet in simulation
            )

            # proceed if we have a sufficient number of transits in both simulations
            if len(transits_u) >= max_transits and len(transits_p) >= max_transits:
                ttv_vec = detrend_ttv(transits_p[:max_transits] - transits_u[:max_transits])

                # Convert TTV wiggles into a frequency spectrum to identify the dominant
                # orbital resonance frequency between the planets.
                # Search for cycles that repeat between 3 transits and 50 transits, with 40 sample bins
                freqs = np.linspace(1/50, 1/3, 40)
                # compare TTV against sine wave at every frequency in 'freqs' and take power
                power = LombScargle(np.arange(max_transits), ttv_vec).power(freqs)

                # Feature Construction: [Amplitude, Power Spectrum, Phys Params]
                amp = np.std(ttv_vec)
                phys = [m_b, P_b, e_b, ratio]    # our physical features of the planets

                features = np.hstack(([amp], power, phys)) # stack our features
                all_features.append(features)
                all_masses.append(m_c)
                all_ttvs.append(ttv_vec)


        except Exception:
            print(f"Exception in simulation {i}. Proceeding to next simulation.")
            continue

    return np.array(all_features), np.array(all_masses), np.array(all_ttvs)

#-----------------------------------------------------------------------------
# PHASE 2: MASS PREDICTOR MODEL
#-----------------------------------------------------------------------------

class MassPredictor(nn.Module):
    """
    Neural Network model designed to estimate planetary mass from
    TTV amplitudes and periodogram features.
    Because we're working with normalised data and will have negative values,
    use LeakyReLU to stop neurons "dying".
    """
    def __init__(self, input_size):
        super(MassPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 45),
            nn.BatchNorm1d(45),        # Stabilizes training by normalizing layer outputs
            nn.LeakyReLU(0.1),          # Allows small gradients for negative values to prevent "dead neurons"
            nn.Dropout(0.1),            # Randomly zeros 10% of neurons to prevent overfitting to specific noise

            nn.Linear(45, 22),
            nn.BatchNorm1d(22),
            nn.LeakyReLU(0.1),

            nn.Linear(22, 11),
            nn.BatchNorm1d(11),
            nn.LeakyReLU(0.1),

            nn.Linear(11, 1)
        )

    def forward(self, x):
        return self.network(x)

def train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=16):
    print("\nStarting model training...")
    # --- Data Preparation ---
    # Convert NumPy arrays to PyTorch Tensors (the required data format for the model)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    # .view(-1, 1) ensures the target labels have the correct shape for the loss function
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # HuberLoss is robust to outliers, combining Mean Squared Error and Mean Absolute Error
    criterion = nn.HuberLoss()
    # Adam optimizer handles weight updates; weight_decay adds L2 regularization to prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    # Scheduler reduces the learning rate when the validation loss plateaus to fine-tune results
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # --- Early Stopping Setup ---
    best_val_loss, patience_counter, max_patience = float('inf'), 0, 30
    best_model_state = None

    # --- Tracking Loss for Plotting ---
    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        # 1. Training Phase
        model.train() # Set model to training mode (enables Dropout/BatchNorm)
        train_loss = 0
        indices = torch.randperm(len(X_train_t)) # Shuffle data every epoch to improve generalization. Prevents learning order.

        # Mini-batch loop
        for i in range(0, len(X_train_t), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X, batch_y = X_train_t[batch_indices], y_train_t[batch_indices]
            # Forward pass: Compute predicted outputs by passing inputs to the model
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            # Backward pass and optimization
            optimizer.zero_grad()   # Clear previous gradients
            loss.backward()         # Compute gradients using backpropagation
            optimizer.step()        # Update model weights
            train_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / len(X_train_t)

        # 2. Validation Phase
        model.eval() # Set model to evaluation mode (disables Dropout/BatchNorm)
        with torch.no_grad():   # Disable gradient calculation to save memory and time
            val_loss = criterion(model(X_val_t), y_val_t)

        # --- Record Losses ---
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(val_loss.item())

        scheduler.step(val_loss) # Update learning rate based on validation performance
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss.item():.6f}')

        # --- Early Stopping Logic ---
        # If model improves, save the current best weights
        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            best_model_state = model.state_dict().copy()
        else:
            # If no improvement, increment counter; stop training if max_patience reached
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Reload the best performing weights before returning the model
    if best_model_state:
        model.load_state_dict(best_model_state)
    # Return the trained model AND the loss history
    return model, train_loss_history, val_loss_history

#-----------------------------------------------------------------------------
# PHASE 3: TRAINING AND EVALUATION
#-----------------------------------------------------------------------------

if __name__ == "__main__":
    # --- data generation ---
    X_raw, y_raw, ttvs = generate_simulation_data(NUM_SIMULATIONS, MAX_TRANSITS)

    # Log Scaling for Mass (converts tiny decimals to manageable numbers like -1 to -4)
    y_log = np.log10(y_raw)

    # --- preprocessing ---
    # We split first so we only fit the scaler on the training data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_log, test_size=0.2, random_state=42
    )

    # Fit scalar to training data, use this scalar to transform test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Validation split for the NN
    X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )

    # --- train neural network ---
    print("\n--- Training Neural Network ---")
    model_nn = MassPredictor(X_train_nn.shape[1])
    model_nn, nn_train_loss, nn_val_loss = train_model(
        model_nn, X_train_nn, y_train_nn, X_val_nn, y_val_nn, epochs=200
    )

    # --- train boosted decision tree ---
    print("\n--- Training Boosted Decision Tree (Standard Squared Error) ---")

    model_bdt = xgb.XGBRegressor(
        n_estimators=500,               # Number of trees
        max_depth=6,                    # max depth of single tree
        learning_rate=0.05,
        objective="reg:squarederror",   # Most stable for log-mass targets
        eval_metric="rmse",             # Tracks root-mean-square error
        random_state=42
    )

    # Watch both Train and Test to check for overfitting
    model_bdt.fit(
        X_train_raw, y_train,
        eval_set=[(X_train_raw, y_train), (X_test_raw, y_test)],
        verbose=True
    )

    # --- STEP 5: COMPARISON & RESULTS ---
    model_nn.eval()
    with torch.no_grad():
        preds_nn = model_nn(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy().flatten()

    preds_bdt = model_bdt.predict(X_test_raw)

    mae_nn = mean_absolute_error(y_test, preds_nn)
    mae_bdt = mean_absolute_error(y_test, preds_bdt)

    print("\n" + "="*30)
    print(f"FINAL PERFORMANCE ({NUM_SIMULATIONS} SIMS, {MAX_TRANSITS} TRANSITS EACH)")
    print(f"Neural Network MAE: {mae_nn:.4f} dex")
    print(f"XGBoost BDT MAE:    {mae_bdt:.4f} dex")
    print("="*30)



    # -----------------------------------------------------------------------------
    # FINAL SIDE-BY-SIDE COMPARISON
    # -----------------------------------------------------------------------------

    # Visualize sample TTV signal
    plt.figure(figsize=(10, 6))
    plt.plot(ttvs[0] * SIM_TIME_TO_DAYS, 'o-', label='TTV Signal')
    plt.title(f'Sample TTV Signal (Mass = {y_raw[0]*SOLAR_TO_EARTH:.2f} Earth Masses)')
    plt.xlabel('Transit Number')
    plt.ylabel('Time Variation (Days)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Feature Importance Analysis ---
    # Map names to your indices
    feature_names = ["TTV Amp"] + [f"Freq_{i}" for i in range(40)] + ["Mass_B", "Period_B", "Ecc_B", "Period_Ratio"]
    importances = model_bdt.feature_importances_

    # Sort them for the plot
    indices = np.argsort(importances)[-10:]  # Top 10 most important

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align='center', color='teal')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("XGBoost Feature Importance Score")
    plt.title("Which features helped predict the Perturber Mass?")
    plt.tight_layout()
    plt.show()

    # 1. LEARNING CURVE COMPARISON
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    # NN
    ax[0].plot(nn_train_loss, label='Train Loss', color='blue')
    ax[0].plot(nn_val_loss, label='Val Loss', color='red', linestyle='--')
    ax[0].set_title("Neural Network: Huber Loss")
    ax[0].set_xlabel("Epoch"); ax[0].legend()

    # BDT Learning Curve
    results = model_bdt.evals_result()
    ax[1].plot(results['validation_0']['rmse'], label='Train RMSE', color='green')
    ax[1].plot(results['validation_1']['rmse'], label='Test RMSE', color='orange', linestyle='--')
    ax[1].set_title("BDT: RMSE Learning Curve")
    ax[1].set_xlabel("Number of Trees"); ax[1].legend()

    plt.suptitle("Figure 1: Training Convergence & Overfitting Check", fontsize=14)
    plt.tight_layout()
    plt.show()



    # 2. PREDICTED VS ACTUAL (The "Scatter" Test)
    y_test_earth = 10**y_test * SOLAR_TO_EARTH
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True, sharex=True)
    # NN
    ax[0].scatter(y_test_earth, 10**preds_nn * SOLAR_TO_EARTH, alpha=0.3, color='blue')
    ax[0].plot([y_test_earth.min(), y_test_earth.max()], [y_test_earth.min(), y_test_earth.max()], 'r--')
    ax[0].set_title(f"NN (MAE: {mae_nn:.3f} dex)")
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_ylabel("Predicted Mass (Earths)")

    # BDT
    ax[1].scatter(y_test_earth, 10**preds_bdt * SOLAR_TO_EARTH, alpha=0.3, color='green')
    ax[1].plot([y_test_earth.min(), y_test_earth.max()], [y_test_earth.min(), y_test_earth.max()], 'r--')
    ax[1].set_title(f"BDT (MAE: {mae_bdt:.3f} dex)")
    plt.suptitle("Figure 2: Mass Prediction Accuracy", fontsize=14)
    plt.tight_layout()
    plt.show()

    # 3. PERCENTAGE ERROR VS MASS
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    res_nn = (10**(preds_nn - y_test) - 1) * 100
    res_bdt = (10**(preds_bdt - y_test) - 1) * 100

    ax[0].scatter(y_test_earth, res_nn, alpha=0.4, color='purple')
    ax[0].axhline(0, color='black', ls='--')
    ax[0].set_title("NN Percentage Error")
    ax[0].set_xscale('log')

    ax[1].scatter(y_test_earth, res_bdt, alpha=0.4, color='orange')
    ax[1].axhline(0, color='black', ls='--')
    ax[1].set_title("BDT Percentage Error")
    ax[1].set_xscale('log')
    plt.suptitle("Figure 3: Error Trends across Mass Ranges", fontsize=14)
    plt.show()



    # 4. ERROR DISTRIBUTION (Histograms)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    ax[0].hist(res_nn, bins=30, color='blue', alpha=0.6, edgecolor='black')
    ax[0].set_title("NN Error Distribution")
    ax[0].set_xlabel("Error %")

    ax[1].hist(res_bdt, bins=30, color='green', alpha=0.6, edgecolor='black')
    ax[1].set_title("BDT Error Distribution")
    ax[1].set_xlabel("Error %")
    plt.suptitle("Figure 4: Bias and Variance Comparison", fontsize=14)
    plt.show()

    # 5. RESONANCE ANALYSIS (Error vs Period Ratio)
    test_ratios = X_test_raw[:, 44]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Define common orbital resonances to mark
    resonances = [1.5, 2.0, 3.0]
    res_labels = ["3:2", "2:1", "3:1"]

    for i, axes in enumerate(ax):
        # Determine which model predictions to use for coloring/y-axis
        current_res = res_nn if i == 0 else res_bdt
        title = "NN: Error vs Resonance" if i == 0 else "BDT: Error vs Resonance"

        # Plot the data
        im = axes.scatter(test_ratios, np.abs(current_res), c=y_test_earth,
                          cmap='viridis', norm=mcolors.LogNorm(), alpha=0.5)

        # Add vertical resonance markers
        for val, label in zip(resonances, res_labels):
            axes.axvline(val, color='red', linestyle='--', alpha=0.6, lw=1)
            # Add text labels at the top of the plot
            axes.text(val, axes.get_ylim()[1]*0.9, label, color='red',
                      fontsize=9, ha='center', fontweight='bold')

        axes.set_title(title)
        axes.set_xlabel("Period Ratio ($P_c/P_b$)")
        if i == 0:
            axes.set_ylabel("Absolute Error (%)")

    # Add a unified colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Make room for colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='True Mass ($M_{Earth}$)')

    plt.suptitle("Figure 5: Physics Check - Model Performance near Orbital Resonances", fontsize=14, y=1.05)
    plt.show()
