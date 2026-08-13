"""
Updated since presentation. The main updates are:
- reducing the size of the neural network model
- also incorporating a Boosted Decision Tree
- plotting loss curves


To use the script, just pop into your environment of choice and run.
You can change NUM_SIMULATIONS, MAX_TRANSITs if you wish
"""

import numpy as np
import rebound
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.ioff() # Turn off interactive mode
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from astropy.timeseries import LombScargle
from scipy.fft import fft

#-----------------------------------------------------------------------------
# Script Variables & Physics Constants
#-----------------------------------------------------------------------------
NUM_SIMULATIONS = 5000
MAX_TRANSITS = 100

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
    planet = sim.particles[planet_index]# index of the planet in our sim whose transits we want to detect
    prev_y = planet.y - star.y

    time = 0
    while time < integration_time and len(transit_times) < max_transits:
        sim.integrate(sim.t + dt)           # progress simulation
        time = sim.t                        # get time from sim
        curr_y = planet.y - star.y
        # if the product of previous and current y is negative, they are on
        # opposite sides of the y axis, and so there has been a transit of the y-axis
        # if (planet.x - star.x) > 0, then the planet is in front of the star: transit occured
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

def compute_oc_residuals(transit_times, max_transits):
    """
    Compute O-C residuals by fitting a linear ephemeris to observed transit times.

    Inputs:
        transit_times: Array of observed (perturbed) transit times
        max_transits: Maximum number of transits to use
    Outputs:
        Array of detrended O-C residuals
    """
    # Use only the first `max_transits` transits
    transit_times = transit_times[:max_transits]
    transit_numbers = np.arange(max_transits)  # [0, 1, 2, ..., max_transits-1]

    # Fit linear ephemeris: t = t0 + P * n
    # np.polyfit returns [slope (P), intercept (t0)] for degree=1
    P_est, t0_est = np.polyfit(transit_numbers, transit_times, 1)

    # Calculate the linear ephemeris
    linear_ephemeris = P_est * transit_numbers + t0_est

    # Compute O-C residuals: observed - calculated
    oc_residuals = transit_times - linear_ephemeris

    # Detrend the residuals (optional, but common in TTV analysis)
    return detrend_ttv(oc_residuals)

#-----------------------------------------------------------------------------
# PHASE 1: GENERATE DATA WITH PERIODOGRAM
#-----------------------------------------------------------------------------

def generate_simulation_data(
    num_simulations,
    max_transits,
    resonances=None,
    resonance_fraction=1.0,
    add_noise=False,
    noise_level=0.01,
    missing_transits=False,
    missing_fraction=0.1
):
    """
    Generate simulation data with optional challenges (noise, missing transits, custom resonances).

    Inputs:
        num_simulations: Number of simulations to run.
        max_transits: Max number of transits to detect.
        resonances: List of resonance ratios (e.g., [1.5, 2.0, 3.0]). If None, use default.
        resonance_fraction: sets the fraction of resonant systems that will be generated
        add_noise: If True, add Gaussian noise to transit times.
        noise_level: Fraction of P_b to use as noise std (e.g., 0.01 = 1% of P_b).
        missing_transits: If True, randomly drop some transits.
        missing_fraction: Fraction of transits to drop (e.g., 0.1 = 10%).
    Outputs:
        all_features, all_masses, all_ttvs (as before)
    """
    print(f"Generating {num_simulations} simulations...")
    all_features = []
    all_masses = []
    all_ttvs = []

    # Default resonances if none provided
    if resonances is None:
        resonances = [1.5, 2.0, 3.0]

    for i in range(num_simulations):
        if (i + 1) % 50 == 0:
            print(f"  Simulation {i+1}...")

        try:
            # Randomize Planet B (The Transitor)
            m_b = np.random.uniform(1e-6, 1e-4)  # Mass (Earth to Neptune in solar masses)
            P_b = np.random.uniform(2*np.pi, 4*np.pi)  # Period
            e_b = np.random.uniform(0.0, 0.02)  # Eccentricity

            # Randomize Planet C (The Perturber)
            if np.random.rand() < resonance_fraction:
                ratio = np.random.choice(resonances) + np.random.uniform(-0.02, 0.02)  # Tighter resonance
            else:
                ratio = np.random.uniform(1.1, 4.0)  # Wider range for non-resonant systems
            m_c = np.random.uniform(1e-4, 5e-3)  # Mass
            P_c = P_b * ratio  # Period
            e_c = np.random.uniform(0.0, 0.05)  # Eccentricity

            dt = min(P_b, P_c) / 100  # Use the smaller of P_b or P_c

            # Perturbed Simulation (no unperturbed simulation needed)
            sim_p = rebound.Simulation()
            sim_p.integrator = "whfast"
            sim_p.dt = dt
            sim_p.add(m=1.0)  # Star
            sim_p.add(m=m_b, P=P_b, e=e_b)  # Planet B
            sim_p.add(m=m_c, P=P_c, e=e_c)  # Planet C
            sim_p.move_to_com()
            transits_p = detect_transits(
                sim_p,
                P_b * (max_transits + 5),
                dt,
                max_transits,
                1
            )

            # Add noise to transit times if requested
            if add_noise:
                transits_p += np.random.normal(0, noise_level * P_b, size=len(transits_p))

            # Randomly drop transits if requested
            if missing_transits and len(transits_p) > max_transits:
                num_to_keep = int(len(transits_p) * (1 - missing_fraction))
                keep_indices = np.random.choice(len(transits_p), size=num_to_keep, replace=False)
                transits_p = transits_p[np.sort(keep_indices)]

            # Proceed if we have enough transits
            if len(transits_p) >= max_transits:
                # Compute O-C residuals (realistic TTVs)
                ttv_vec = compute_oc_residuals(transits_p, max_transits)

                # Feature construction
                #freqs = np.linspace(1/50, 1/3, 40)
                fft_coeffs = fft(ttv_vec)[:20]  # Take first 20 complex coefficients
                amp = np.std(ttv_vec)
                phys = [P_b, ratio, e_b]  # Note: m_b removed to avoid leakage
                features = np.hstack(([amp], np.real(fft_coeffs), np.imag(fft_coeffs), phys))

                all_features.append(features)
                all_masses.append(m_c)
                all_ttvs.append(ttv_vec)

        except Exception as e:
            print(f"Exception in simulation {i}: {e}. Proceeding to next simulation.")
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
            nn.Linear(input_size, 32),
            # nn.BatchNorm1d(32),        # Stabilizes training by normalizing layer outputs
            nn.LeakyReLU(0.1),          # Allows small gradients for negative values to prevent "dead neurons"
            nn.Dropout(0.1),            # Randomly zeros 10% of neurons to prevent overfitting to specific noise

            nn.Linear(32, 16),
            # nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

class QuantilePredictor(nn.Module):
    """
    Neural Network for predicting mass percentiles (uncertainty estimation).
    Outputs: [10th percentile, 50th percentile (median), 90th percentile].
    """
    def __init__(self, input_size):
        super(QuantilePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 3)  # Predict 3 quantiles (10th, 50th, 90th)
        )

    def forward(self, x):
        return self.network(x)

# Custom loss for quantile regression (pinball loss)
def quantile_loss(preds, targets, quantiles=[0.1, 0.5, 0.9]):
    loss = 0
    for i, q in enumerate(quantiles):
        errors = targets - preds[:, i]
        loss += torch.mean(torch.max(q * errors, (q - 1) * errors))
    return loss

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

def train_quantile_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=16):
    """
    Train a quantile regression model (predicts percentiles for uncertainty estimation).
    Uses pinball loss for quantile regression.
    """
    print("\nStarting quantile model training...")
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Shape: (n, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # Quantiles to predict (10th, 50th, 90th)
    quantiles = [0.1, 0.5, 0.9]

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # Early stopping
    best_val_loss, patience_counter, max_patience = float('inf'), 0, 30
    best_model_state = None

    # Loss tracking
    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        indices = torch.randperm(len(X_train_t))

        for i in range(0, len(X_train_t), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X, batch_y = X_train_t[batch_indices], y_train_t[batch_indices]

            # Forward pass: outputs are 3 quantiles (shape: [batch_size, 3])
            outputs = model(batch_X)
            loss = quantile_loss(outputs, batch_y, quantiles)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(X_train_t)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)  # Shape: [n_val, 3]
            val_loss = quantile_loss(val_outputs, y_val_t, quantiles)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(val_loss.item())

        scheduler.step(val_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss.item():.6f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, train_loss_history, val_loss_history

#-----------------------------------------------------------------------------
# PHASE 3: TRAINING AND EVALUATION
#-----------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Generate training data ---
    print("\n--- Generating Training Data ---")
    X_train_raw, y_train, ttvs_train = generate_simulation_data(
        num_simulations=NUM_SIMULATIONS,
        max_transits=MAX_TRANSITS,
        resonances=[1.5, 2.0, 3.0],
        resonance_fraction=0.5,     # 50% resonant
        add_noise=False,
        missing_transits=False
    )

    # Create the initial test set (random split of training data)
    X_train_nn_raw, X_test_original, y_train_nn_raw, y_test_original = train_test_split(
        X_train_raw, y_train, test_size=0.2, random_state=42
    )

    print("\n--- Generating Test Sets ---")

    # 1. Unseen resonances (e.g., 4:3, 5:4)
    X_test_res, y_test_res, _ = generate_simulation_data(
        num_simulations=200,
        max_transits=MAX_TRANSITS,
        resonances=[4 / 3, 5 / 4, 3 / 2],  # 4:3, 5:4, 3:2 (3:2 is seen, others are new)
        resonance_fraction=1,
        add_noise=False,
        missing_transits=False
    )

    # 2. Non-resonant systems
    X_test_non_res, y_test_non_res, _ = generate_simulation_data(
        num_simulations=200,
        max_transits=MAX_TRANSITS,
        resonances=None,
        resonance_fraction=0,
        add_noise=False,
        missing_transits=False
    )

    # 3. Noisy data (same resonances as training, but with noise)
    X_test_noisy, y_test_noisy, _ = generate_simulation_data(
        num_simulations=200,
        max_transits=MAX_TRANSITS,
        resonances=[1.5, 2.0, 3.0],
        resonance_fraction=0.5,
        add_noise=True,
        noise_level=0.01,  # 1% of P_b
        missing_transits=False
    )

    # 4. Missing transits
    X_test_missing, y_test_missing, _ = generate_simulation_data(
        num_simulations=200,
        max_transits=MAX_TRANSITS,
        resonances=[1.5, 2.0, 3.0],
        resonance_fraction=0.5,
        add_noise=False,
        missing_transits=True,
        missing_fraction=0.1  # Drop 10% of transits
    )

    # --- Preprocessing ---
    # Log-scale the masses for all datasets
    y_train_log = np.log10(y_train_nn_raw)
    y_test_original_log = np.log10(y_test_original)  # We'll create this below
    y_test_res_log = np.log10(y_test_res)
    y_test_non_res_log = np.log10(y_test_non_res)
    y_test_noisy_log = np.log10(y_test_noisy)
    y_test_missing_log = np.log10(y_test_missing)

    # Split training data into train/val for NN
    X_train_final, X_val_nn_raw, y_train_nn, y_val_nn = train_test_split(
        X_train_nn_raw, y_train_log, test_size=0.2, random_state=42
    )

    # Fit scaler ONLY on the final training set
    scaler = StandardScaler()
    X_train_final_scaled = scaler.fit_transform(X_train_final)
    X_val_nn_scaled = scaler.transform(X_val_nn_raw)

    # Transform all test sets using the SAME scaler
    X_test_original_scaled = scaler.transform(X_test_original)  # Your original test set
    X_test_res_scaled = scaler.transform(X_test_res)
    X_test_non_res_scaled = scaler.transform(X_test_non_res)
    X_test_noisy_scaled = scaler.transform(X_test_noisy)
    X_test_missing_scaled = scaler.transform(X_test_missing)

    # --- Train Models ---
    # Neural Network returning single value
    model_point = MassPredictor(X_train_final_scaled.shape[1])
    model_nn, nn_train_loss, nn_val_loss = train_model(
        model_point,
        X_train_final_scaled,  # Scaled features
        y_train_nn,  # Log-scaled masses
        X_val_nn_scaled,  # Scaled features
        y_val_nn,  # Log-scaled masses
        epochs=200
    )

    # Neural network returning range
    # Train QuantilePredictor (for uncertainty estimation)
    model_quantile = QuantilePredictor(X_train_final_scaled.shape[1])
    model_quantile, q_train_loss, q_val_loss = train_quantile_model(
        model_quantile,
        X_train_final_scaled,  # Scaled features
        y_train_nn,  # Log-scaled masses
        X_val_nn_scaled,  # Scaled features
        y_val_nn,  # Log-scaled masses
        epochs=200
    )

    # XGBoost
    model_bdt = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42
    )
    model_bdt.fit(
        X_train_final,
        y_train_nn,
        eval_set=[(X_train_final, y_train_nn), (X_val_nn_raw, y_val_nn)],
        verbose=True
    )


    # --- Evaluate on Harder Test Sets ---
    def evaluate_model(model, X_test, y_test, model_type="nn"):
        """Evaluate a model on a test set and return MAE."""
        if model_type == "nn":
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()
        else:  # XGBoost or other scikit-learn models
            preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        return mae, preds


    def evaluate_quantile_model(model, X_test, y_test):
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()  # Shape: (n, 3)
        # Median predictions (50th percentile) for MAE comparison
        mae_median = mean_absolute_error(y_test, preds[:, 1])
        # Coverage: % of true values within 10th-90th percentile range
        lower = preds[:, 0]  # 10th percentile
        upper = preds[:, 2]  # 90th percentile
        coverage = np.mean((y_test >= lower) & (y_test <= upper)) * 100
        # Average range width (in dex)
        range_width = np.mean(upper - lower)
        return mae_median, coverage, range_width, preds

    # Evaluate Neural Network on all test sets
    mae_original_nn, _ = evaluate_model(model_nn, X_test_original_scaled, y_test_original_log, "nn")
    mae_res_nn, _ = evaluate_model(model_nn, X_test_res_scaled, y_test_res_log, "nn")
    mae_non_res_nn, _ = evaluate_model(model_nn, X_test_non_res_scaled, y_test_non_res_log, "nn")
    mae_noisy_nn, _ = evaluate_model(model_nn, X_test_noisy_scaled, y_test_noisy_log, "nn")
    mae_missing_nn, _ = evaluate_model(model_nn, X_test_missing_scaled, y_test_missing_log, "nn")

    # --- Evaluate QuantilePredictor on all test sets ---
    mae_original_qp, coverage_original_qp, range_original_qp, _ \
        = evaluate_quantile_model(model_quantile, X_test_original_scaled, y_test_original_log)
    mae_res_qp, coverage_res_qp, range_res_qp, _ \
        = evaluate_quantile_model(model_quantile, X_test_res_scaled, y_test_res_log)
    mae_non_res_qp, coverage_non_res_qp, range_non_res_qp, _ \
        = evaluate_quantile_model(model_quantile, X_test_non_res_scaled, y_test_non_res_log)
    mae_noisy_qp, coverage_noisy_qp, range_noisy_qp, _ \
        = evaluate_quantile_model(model_quantile, X_test_noisy_scaled, y_test_noisy_log)
    mae_missing_qp, coverage_missing_qp, range_missing_qp, _ \
        = evaluate_quantile_model(model_quantile,X_test_missing_scaled,y_test_missing_log)

    # Evaluate XGBoost on all test sets
    mae_original_bdt, _ = evaluate_model(model_bdt, X_test_original, y_test_original_log, "xgb")
    mae_res_bdt, _ = evaluate_model(model_bdt, X_test_res, y_test_res_log, "xgb")
    mae_non_res_bdt, _ = evaluate_model(model_bdt, X_test_non_res, y_test_non_res_log, "xgb")
    mae_noisy_bdt, _ = evaluate_model(model_bdt, X_test_noisy, y_test_noisy_log, "xgb")
    mae_missing_bdt, _ = evaluate_model(model_bdt, X_test_missing, y_test_missing_log, "xgb")

    # Print performance summary for Neural Network
    print("\n--- Neural Network Performance ---")
    print(f"{'Test Set':<25} {'MAE (dex)':<15}")
    print("-" * 40)
    print(f"{'Original Test Set':<25} {mae_original_nn:.4f}")
    print(f"{'Unseen Resonances':<25} {mae_res_nn:.4f}")
    print(f"{'Non-Resonant Systems':<25} {mae_non_res_nn:.4f}")
    print(f"{'Noisy Data':<25} {mae_noisy_nn:.4f}")
    print(f"{'Missing Transits':<25} {mae_missing_nn:.4f}")

    # Print performance summary for XGBoost
    print("\n--- XGBoost Performance ---")
    print(f"{'Test Set':<25} {'MAE (dex)':<15}")
    print("-" * 40)
    print(f"{'Original Test Set':<25} {mae_original_bdt:.4f}")
    print(f"{'Unseen Resonances':<25} {mae_res_bdt:.4f}")
    print(f"{'Non-Resonant Systems':<25} {mae_non_res_bdt:.4f}")
    print(f"{'Noisy Data':<25} {mae_noisy_bdt:.4f}")
    print(f"{'Missing Transits':<25} {mae_missing_bdt:.4f}")

    # Print performance summary for Quantile Predictor
    print("\n--- Quantile Predictor Performance ---")
    print(f"{'Test Set':<25} {'MAE (dex)':<15} {'Coverage (%)':<15} {'Range Width (dex)':<20}")
    print("-" * 75)
    print(f"{'Original Test Set':<25} {mae_original_qp:<15.4f} {coverage_original_qp:<15.1f} {range_original_qp:<20.4f}")
    print(f"{'Unseen Resonances':<25} {mae_res_qp:<15.4f} {coverage_res_qp:<15.1f} {range_res_qp:<20.4f}")
    print(f"{'Non-Resonant Systems':<25} {mae_non_res_qp:<15.4f} {coverage_non_res_qp:<15.1f} {range_non_res_qp:<20.4f}")
    print(f"{'Noisy Data':<25} {mae_noisy_qp:<15.4f} {coverage_noisy_qp:<15.1f} {range_noisy_qp:<20.4f}")
    print(f"{'Missing Transits':<25} {mae_missing_qp:<15.4f} {coverage_missing_qp:<15.1f} {range_missing_qp:<20.4f}")


    # --- STEP 5: COMPARISON & RESULTS ---
    model_nn.eval()
    with torch.no_grad():
        preds_nn = model_nn(torch.tensor(X_test_original_scaled, dtype=torch.float32)).numpy().flatten()
    preds_bdt = model_bdt.predict(X_test_original)  # XGBoost uses unscaled data
    mae_nn = mean_absolute_error(y_test_original_log, preds_nn)
    mae_bdt = mean_absolute_error(y_test_original_log, preds_bdt)

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
    plt.plot(ttvs_train[0] * SIM_TIME_TO_DAYS, 'o-', label='TTV Signal')
    plt.title(f'Sample TTV Signal (Mass = {y_train[0] * SOLAR_TO_EARTH:.2f} Earth Masses)')
    plt.xlabel('Transit Number')
    plt.ylabel('Time Variation (Days)')
    plt.legend()
    plt.grid(True)
    # plt.show()

    # Verify data shapes
    print(f"Training (NN): {X_train_final_scaled.shape}, {y_train_nn.shape}")
    print(f"Validation (NN): {X_val_nn_scaled.shape}, {y_val_nn.shape}")
    print(f"Test (Original): {X_test_original_scaled.shape}, {y_test_original_log.shape}")

    # --- Feature Importance Analysis ---
    # Map names to your indices
    feature_names = (
            ["TTV Amp"] +
            [f"FFT_Real_{i}" for i in range(40)] +  # 40 real parts from FFT
            [f"FFT_Imag_{i}" for i in range(40)] +  # 40 imaginary parts from FFT
            ["Period_B", "Ecc_B", "Period_Ratio"]
    )
    importances = model_bdt.feature_importances_

    # Sort them for the plot
    indices = np.argsort(importances)[-10:]  # Top 10 most important

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align='center', color='teal')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("XGBoost Feature Importance Score")
    plt.title("Which features helped predict the Perturber Mass?")
    plt.tight_layout()
    # plt.show()

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
    # plt.show()



    # 2. PREDICTED VS ACTUAL (The "Scatter" Test)
    y_test_earth = 10**y_test_original_log * SOLAR_TO_EARTH
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
    #plt.show()

    # 3. PERCENTAGE ERROR VS MASS
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    res_nn = (10 ** (preds_nn - y_test_original_log) - 1) * 100
    res_bdt = (10 ** (preds_bdt - y_test_original_log) - 1) * 100

    ax[0].scatter(y_test_earth, res_nn, alpha=0.4, color='purple')
    ax[0].axhline(0, color='black', ls='--')
    ax[0].set_title("NN Percentage Error")
    ax[0].set_xscale('log')

    ax[1].scatter(y_test_earth, res_bdt, alpha=0.4, color='orange')
    ax[1].axhline(0, color='black', ls='--')
    ax[1].set_title("BDT Percentage Error")
    ax[1].set_xscale('log')
    plt.suptitle("Figure 3: Error Trends across Mass Ranges", fontsize=14)
    #plt.show()



    # 4. ERROR DISTRIBUTION (Histograms)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    ax[0].hist(res_nn, bins=30, color='blue', alpha=0.6, edgecolor='black')
    ax[0].set_title("NN Error Distribution")
    ax[0].set_xlabel("Error %")

    ax[1].hist(res_bdt, bins=30, color='green', alpha=0.6, edgecolor='black')
    ax[1].set_title("BDT Error Distribution")
    ax[1].set_xlabel("Error %")
    plt.suptitle("Figure 4: Bias and Variance Comparison", fontsize=14)
    #plt.show()

    # 5. RESONANCE ANALYSIS (Error vs Period Ratio)
    test_ratios = X_test_original[:, 43]  # Period ratio is at index 43 (after 40 power features + 3 phys features)
    y_test_original_earth = 10 ** y_test_original_log * SOLAR_TO_EARTH  # Convert to Earth masses

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
    #plt.show()

    # Plot predicted vs. actual for all test sets (Neural Network)
    test_sets = [
        ("Original", X_test_original_scaled, y_test_original_log, mae_original_nn),
        ("Unseen Resonances", X_test_res_scaled, y_test_res_log, mae_res_nn),
        ("Non-Resonant", X_test_non_res_scaled, y_test_non_res_log, mae_non_res_nn),
        ("Noisy", X_test_noisy_scaled, y_test_noisy_log, mae_noisy_nn),
        ("Missing Transits", X_test_missing_scaled, y_test_missing_log, mae_missing_nn)
    ]

    for name, X_test, y_test, mae in test_sets:
        model_nn.eval()
        with torch.no_grad():
            preds = model_nn(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()
        y_test_earth = 10 ** y_test * SOLAR_TO_EARTH
        preds_earth = 10 ** preds * SOLAR_TO_EARTH

        plt.figure(figsize=(8, 5))
        plt.scatter(y_test_earth, preds_earth, alpha=0.3, color='blue')
        plt.plot([y_test_earth.min(), y_test_earth.max()], [y_test_earth.min(), y_test_earth.max()], 'r--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Actual Mass (Earth Masses)")
        plt.ylabel("Predicted Mass (Earth Masses)")
        plt.title(f"NN: {name} (MAE: {mae:.3f} dex)")
        plt.grid(alpha=0.3)
        #plt.show()

    # Plot quantile predictions vs. actual (for original test set)
    y_test_earth = 10 ** y_test_original_log * SOLAR_TO_EARTH
    preds_quantile = model_quantile(torch.tensor(X_test_original_scaled, dtype=torch.float32)).detach().numpy()
    lower_earth = 10 ** preds_quantile[:, 0] * SOLAR_TO_EARTH  # 10th percentile
    median_earth = 10 ** preds_quantile[:, 1] * SOLAR_TO_EARTH  # 50th percentile
    upper_earth = 10 ** preds_quantile[:, 2] * SOLAR_TO_EARTH  # 90th percentile

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_earth, median_earth, alpha=0.3, color='blue', label='Median Prediction')
    plt.fill_between(
        y_test_earth,
        lower_earth,
        upper_earth,
        alpha=0.2,
        color='blue',
        label='10th–90th Percentile Range'
    )
    plt.plot([y_test_earth.min(), y_test_earth.max()], [y_test_earth.min(), y_test_earth.max()], 'r--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Actual Mass (Earth Masses)")
    plt.ylabel("Predicted Mass (Earth Masses)")
    plt.title(f"Quantile Predictor: Median ± Uncertainty (Coverage: {coverage_original_qp:.1f}%)")
    plt.legend()
    plt.grid(alpha=0.3)



    plt.show()