"""
Updated since presentation. The main updates are:
- reducing the size of the neural network model
- also incorporating a Boosted Decision Tree
- plotting loss curves

Second-pass fixes (this version):
- FIX: "missing transits" test set was silently identical to the normal test set,
  because detect_transits can never return more than max_transits points, so the
  old `len(transits_p) > max_transits` guard was always False. Missing transits
  are now dropped from within the max_transits set itself, and the O-C ephemeris
  fit is refit using only the observed epoch numbers (not assumed-consecutive
  indices), then interpolated back onto a fixed-length grid so the feature
  vector size stays constant for the NN/XGBoost inputs.
- FIX: Figure 5 (error vs period ratio) and the XGBoost feature-importance plot
  were both indexing/labelling the feature vector incorrectly. The real layout
  is: [amp(1), fft_real(20), fft_imag(20), P_b, ratio, e_b] -> period ratio is
  at index 42, not 43, and feature_names now matches the true 44-length vector.
- ADDED: randomized argument of pericenter (omega) and mean anomaly (M) for
  both planets, so training systems don't all start at the same orbital phase.
- ADDED: a dedicated low-mass perturber test set.
- ADDED: simple baselines (median predictor, linear regression, ridge) so the
  neural network's added value can actually be judged against something trivial.
- ADDED: quantile-model loss curve plot (data was already being collected but
  never plotted).
- Removed unused LombScargle import.

To use the script, just pop into your environment of choice and run.
You can change NUM_SIMULATIONS, MAX_TRANSITS if you wish
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
plt.ioff()  # Turn off interactive mode
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from scipy.fft import fft

#-----------------------------------------------------------------------------
# Script Variables & Physics Constants
#-----------------------------------------------------------------------------
NUM_SIMULATIONS = 50000
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


def detrend_ttv(ttv_data, x=None):
    """
    Remove the linear trend from a single TTV array.

    Inputs:
        ttv_data: array of ttv/residual data
        x: optional array of x-axis values (e.g. transit epoch numbers) to fit
           against. Defaults to a simple 0..N-1 range, but should be the actual
           epoch numbers when some transits are missing (i.e. not consecutive).
    Outputs:
        array: detrended ttv data
    """
    N = len(ttv_data)
    if x is None:
        x = np.arange(N)
    P = np.polyfit(x, ttv_data, 1)          # fit straight line to ttv data
    return ttv_data - (P[0] * x + P[1])     # subtract line from ttv and return


def compute_oc_residuals(transit_times, transit_numbers=None):
    """
    Compute O-C residuals by fitting a linear ephemeris to observed transit times.

    Inputs:
        transit_times: array of observed (perturbed) transit times
        transit_numbers: epoch number of each transit time. Must be provided
            (and non-consecutive) when some transits are missing, so the
            ephemeris fit uses the correct epoch spacing rather than assuming
            every transit in the array is adjacent.
    Outputs:
        Array of detrended O-C residuals, same length/order as transit_times
    """
    if transit_numbers is None:
        transit_numbers = np.arange(len(transit_times))

    # Fit linear ephemeris: t = t0 + P * n
    P_est, t0_est = np.polyfit(transit_numbers, transit_times, 1)
    linear_ephemeris = P_est * transit_numbers + t0_est

    # Compute O-C residuals: observed - calculated
    oc_residuals = transit_times - linear_ephemeris

    # Detrend the residuals against the actual epoch numbers (not assumed-consecutive)
    return detrend_ttv(oc_residuals, transit_numbers)

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
    missing_fraction=0.1,
    mass_range=(1e-4, 5e-3)
):
    """
    Generate simulation data with optional challenges (noise, missing transits,
    custom resonances, custom perturber mass range).

    Inputs:
        num_simulations: Number of simulations to run.
        max_transits: Max number of transits to detect.
        resonances: List of resonance ratios (e.g., [1.5, 2.0, 3.0]). If None, use default.
        resonance_fraction: fraction of resonant systems that will be generated
        add_noise: If True, add Gaussian noise to transit times.
        noise_level: Fraction of P_b to use as noise std (e.g., 0.01 = 1% of P_b).
        missing_transits: If True, randomly drop some of the observed transits
            (simulating unobserved epochs) before fitting the ephemeris.
        missing_fraction: Fraction of the max_transits epochs to drop.
        mass_range: (min, max) solar masses to draw the perturber mass m_c from.
    Outputs:
        all_features, all_masses, all_ttvs (as before)
    """
    print(f"Generating {num_simulations} simulations...")
    all_features = []
    all_masses = []
    all_ttvs = []

    if resonances is None:
        resonances = [1.5, 2.0, 3.0]

    for i in range(num_simulations):
        if (i + 1) % 50 == 0:
            print(f"  Simulation {i+1}...")

        try:
            # Randomize Planet B (the transiting planet)
            m_b = np.random.uniform(1e-6, 1e-4)
            P_b = np.random.uniform(2*np.pi, 4*np.pi)
            e_b = np.random.uniform(0.0, 0.02)
            omega_b = np.random.uniform(0, 2*np.pi)  # randomize orbital orientation
            M_b = np.random.uniform(0, 2*np.pi)      # randomize starting orbital phase

            # Randomize Planet C (the perturber)
            if np.random.rand() < resonance_fraction:
                ratio = np.random.choice(resonances) + np.random.uniform(-0.02, 0.02)
            else:
                ratio = np.random.uniform(1.1, 4.0)
            m_c = np.random.uniform(mass_range[0], mass_range[1])
            P_c = P_b * ratio
            e_c = np.random.uniform(0.0, 0.05)
            omega_c = np.random.uniform(0, 2*np.pi)
            M_c = np.random.uniform(0, 2*np.pi)

            dt = min(P_b, P_c) / 100

            sim_p = rebound.Simulation()
            sim_p.integrator = "whfast"
            sim_p.dt = dt
            sim_p.add(m=1.0)  # Star
            sim_p.add(m=m_b, P=P_b, e=e_b, omega=omega_b, M=M_b)  # Planet B
            sim_p.add(m=m_c, P=P_c, e=e_c, omega=omega_c, M=M_c)  # Planet C
            sim_p.move_to_com()
            transits_p = detect_transits(
                sim_p,
                P_b * (max_transits + 5),
                dt,
                max_transits,
                1
            )

            if len(transits_p) < max_transits:
                continue

            transits_p = transits_p[:max_transits]
            epoch_numbers = np.arange(max_transits)

            # Add timing noise if requested
            if add_noise:
                transits_p = transits_p + np.random.normal(0, noise_level * P_b, size=max_transits)

            # Drop a random subset of epochs to simulate missing/unobserved transits.
            # We refit the ephemeris using only the observed epoch numbers (so the
            # fit isn't biased by treating a gappy sequence as consecutive), then
            # interpolate back onto the full epoch grid so the feature vector
            # length stays fixed for the NN/XGBoost inputs.
            if missing_transits:
                num_to_drop = int(round(max_transits * missing_fraction))
                num_to_drop = min(max(num_to_drop, 0), max_transits - 3)  # keep >=3 points to fit a line
                drop_idx = np.random.choice(max_transits, size=num_to_drop, replace=False)
                observed_mask = np.ones(max_transits, dtype=bool)
                observed_mask[drop_idx] = False

                obs_times = transits_p[observed_mask]
                obs_epochs = epoch_numbers[observed_mask]
                oc_obs = compute_oc_residuals(obs_times, obs_epochs)
                # Interpolate the O-C residual onto the full epoch grid for feature extraction
                ttv_vec = np.interp(epoch_numbers, obs_epochs, oc_obs)
            else:
                ttv_vec = compute_oc_residuals(transits_p, epoch_numbers)

            # Feature construction (FFT preserves phase, unlike periodogram power alone)
            fft_coeffs = fft(ttv_vec)[:20]  # first 20 complex coefficients
            amp = np.std(ttv_vec)
            phys = [P_b, ratio, e_b]  # m_b intentionally excluded to avoid leakage
            features = np.hstack(([amp], np.real(fft_coeffs), np.imag(fft_coeffs)[1:], phys))

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
    TTV amplitude and FFT features. Small (32 -> 16 -> 1) because the dataset
    is not large enough to justify a bigger network.
    """
    def __init__(self, input_size):
        super(MassPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(32, 16),
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
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.network(x)


def quantile_loss(preds, targets, quantiles=(0.1, 0.5, 0.9)):
    loss = 0
    for i, q in enumerate(quantiles):
        errors = targets - preds[:, i:i+1]
        loss += torch.mean(torch.max(q * errors, (q - 1) * errors))
    return loss


def train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=16):
    print("\nStarting model training...")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    best_val_loss, patience_counter, max_patience = float('inf'), 0, 30
    best_model_state = None

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        indices = torch.randperm(len(X_train_t))

        for i in range(0, len(X_train_t), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X, batch_y = X_train_t[batch_indices], y_train_t[batch_indices]
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(X_train_t)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(val_loss.item())

        scheduler.step(val_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss.item():.6f}')

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


def train_quantile_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=16):
    print("\nStarting quantile model training...")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    quantiles = (0.1, 0.5, 0.9)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    best_val_loss, patience_counter, max_patience = float('inf'), 0, 30
    best_model_state = None

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        indices = torch.randperm(len(X_train_t))

        for i in range(0, len(X_train_t), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X, batch_y = X_train_t[batch_indices], y_train_t[batch_indices]
            outputs = model(batch_X)
            loss = quantile_loss(outputs, batch_y, quantiles)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(X_train_t)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = quantile_loss(val_outputs, y_val_t, quantiles)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(val_loss.item())

        scheduler.step(val_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss.item():.6f}')

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
    DEFAULT_MASS_RANGE = (1e-4, 5e-3)
    LOW_MASS_RANGE = (1e-5, 1e-4)  # below the training floor -> genuine extrapolation test

    print("\n--- Generating Training Data ---")
    X_train_raw, y_train, ttvs_train = generate_simulation_data(
        num_simulations=NUM_SIMULATIONS,
        max_transits=MAX_TRANSITS,
        resonances=[1.5, 2.0, 3.0],
        resonance_fraction=0.5,
        add_noise=False,
        missing_transits=False,
        mass_range=DEFAULT_MASS_RANGE
    )

    X_train_nn_raw, X_test_original, y_train_nn_raw, y_test_original = train_test_split(
        X_train_raw, y_train, test_size=0.2, random_state=42
    )

    print("\n--- Generating Test Sets ---")

    # 1. Unseen resonances (4:3, 5:4 are new; 3:2 was seen in training)
    X_test_res, y_test_res, _ = generate_simulation_data(
        num_simulations=200, max_transits=MAX_TRANSITS,
        resonances=[4/3, 5/4], resonance_fraction=1,
        add_noise=False, missing_transits=False, mass_range=DEFAULT_MASS_RANGE
    )

    # 2. Non-resonant systems
    X_test_non_res, y_test_non_res, _ = generate_simulation_data(
        num_simulations=200, max_transits=MAX_TRANSITS,
        resonances=None, resonance_fraction=0,
        add_noise=False, missing_transits=False, mass_range=DEFAULT_MASS_RANGE
    )

    # 3. Noisy data
    X_test_noisy, y_test_noisy, _ = generate_simulation_data(
        num_simulations=200, max_transits=MAX_TRANSITS,
        resonances=[1.5, 2.0, 3.0], resonance_fraction=0.5,
        add_noise=True, noise_level=0.01, missing_transits=False,
        mass_range=DEFAULT_MASS_RANGE
    )

    # 4. Missing transits
    X_test_missing, y_test_missing, _ = generate_simulation_data(
        num_simulations=200, max_transits=MAX_TRANSITS,
        resonances=[1.5, 2.0, 3.0], resonance_fraction=0.5,
        add_noise=False, missing_transits=True, missing_fraction=0.1,
        mass_range=DEFAULT_MASS_RANGE
    )

    # 5. Low-mass perturbers (extrapolation below the training mass range)
    X_test_lowmass, y_test_lowmass, _ = generate_simulation_data(
        num_simulations=200, max_transits=MAX_TRANSITS,
        resonances=[1.5, 2.0, 3.0], resonance_fraction=0.5,
        add_noise=False, missing_transits=False, mass_range=LOW_MASS_RANGE
    )

    # --- Preprocessing ---
    y_train_log = np.log10(y_train_nn_raw)
    y_test_original_log = np.log10(y_test_original)
    y_test_res_log = np.log10(y_test_res)
    y_test_non_res_log = np.log10(y_test_non_res)
    y_test_noisy_log = np.log10(y_test_noisy)
    y_test_missing_log = np.log10(y_test_missing)
    y_test_lowmass_log = np.log10(y_test_lowmass)


    X_train_final, X_val_nn_raw, y_train_nn, y_val_nn = train_test_split(
        X_train_nn_raw, y_train_log, test_size=0.2, random_state=42
    )

    # Fit scaler ONLY on the final training set (no leakage)
    scaler = StandardScaler()
    X_train_final_scaled = scaler.fit_transform(X_train_final)
    X_val_nn_scaled = scaler.transform(X_val_nn_raw)
    X_test_original_scaled = scaler.transform(X_test_original)

    X_test_res_scaled = scaler.transform(X_test_res)
    X_test_non_res_scaled = scaler.transform(X_test_non_res)
    X_test_noisy_scaled = scaler.transform(X_test_noisy)
    X_test_missing_scaled = scaler.transform(X_test_missing)
    X_test_lowmass_scaled = scaler.transform(X_test_lowmass)

    X_train_final_scaled = np.clip(X_train_final_scaled, -10, 10)
    X_val_nn_scaled = np.clip(X_val_nn_scaled, -10, 10)
    X_test_original_scaled = np.clip(X_test_original_scaled, -10, 10)
    X_test_res_scaled = np.clip(X_test_res_scaled, -10, 10)
    X_test_non_res_scaled = np.clip(X_test_non_res_scaled, -10, 10)
    X_test_noisy_scaled = np.clip(X_test_noisy_scaled, -10, 10)
    X_test_missing_scaled = np.clip(X_test_missing_scaled, -10, 10)
    X_test_lowmass_scaled = np.clip(X_test_lowmass_scaled, -10, 10)

    print("Missing Transits X_test:")
    print(f"  NaN count: {np.isnan(X_test_missing_scaled).sum()}")
    print(f"  Inf count: {np.isinf(X_test_missing_scaled).sum()}")
    print(f"  Min/Max: {X_test_missing_scaled.min():.2e} / {X_test_missing_scaled.max():.2e}")

    print("Missing Transits y_test:")
    print(f"  NaN count: {np.isnan(y_test_missing_log).sum()}")
    print(f"  Inf count: {np.isinf(y_test_missing_log).sum()}")
    print(f"  Min/Max: {y_test_missing_log.min():.2e} / {y_test_missing_log.max():.2e}")

    print("\n--- DEBUG: Missing Transits Features ---")
    print("X_test_missing_scaled shape:", X_test_missing_scaled.shape)
    print("X_test_missing_scaled min/max:", X_test_missing_scaled.min(), X_test_missing_scaled.max())
    print("X_test_missing_scaled mean/std:", X_test_missing_scaled.mean(), X_test_missing_scaled.std())
    print("y_test_missing_log min/max:", y_test_missing_log.min(), y_test_missing_log.max())

    # --- Baselines (fit on the SAME training split, so comparisons are fair) ---
    print("\n--- Fitting Baselines ---")
    median_pred_value = np.median(y_train_nn)

    #linreg = LinearRegression().fit(X_train_final_scaled, y_train_nn)
    #ridge = Ridge(alpha=1.0).fit(X_train_final_scaled, y_train_nn)

    linreg = LinearRegression().fit(X_train_final, y_train_nn)  # Unscaled X
    ridge = Ridge(alpha=1.0).fit(X_train_final, y_train_nn)  # Unscaled X

    # --- Train Models ---
    model_point = MassPredictor(X_train_final_scaled.shape[1])
    model_nn, nn_train_loss, nn_val_loss = train_model(
        model_point, X_train_final_scaled, y_train_nn, X_val_nn_scaled, y_val_nn, epochs=200
    )

    model_quantile = QuantilePredictor(X_train_final_scaled.shape[1])
    model_quantile, q_train_loss, q_val_loss = train_quantile_model(
        model_quantile, X_train_final_scaled, y_train_nn, X_val_nn_scaled, y_val_nn, epochs=200
    )

    model_bdt = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        objective="reg:squarederror", eval_metric="rmse", random_state=42
    )
    model_bdt.fit(
        X_train_final, y_train_nn,
        eval_set=[(X_train_final, y_train_nn), (X_val_nn_raw, y_val_nn)],
        verbose=True
    )

    # Test Linear model on TRAINING data (should work)
    train_preds = linreg.predict(X_train_final_scaled[:5])
    print("\n--- DEBUG: Linear on TRAINING data ---")
    print("First 5 predictions (log scale):", train_preds)
    print("First 5 true y_train (log scale):", y_train_nn[:5])
    print("MAE on training:", mean_absolute_error(y_train_nn[:5], train_preds))

    # --- Evaluation helpers ---
    def evaluate_model(model, X_test, y_test, model_type="nn"):
        if model_type == "nn":
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()
        elif model_type == "median":
            preds = np.full_like(y_test, fill_value=median_pred_value, dtype=float)
        else:  # sklearn-style .predict (xgb, linreg, ridge)
            preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        return mae, preds

    def evaluate_quantile_model(model, X_test, y_test):
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        mae_median = mean_absolute_error(y_test, preds[:, 1])
        lower, upper = preds[:, 0], preds[:, 2]
        coverage = np.mean((y_test >= lower) & (y_test <= upper)) * 100
        range_width = np.mean(upper - lower)
        return mae_median, coverage, range_width, preds

    test_sets_raw = {
        "Original": (X_test_original_scaled, X_test_original, y_test_original_log),
        "Unseen Resonances": (X_test_res_scaled, X_test_res, y_test_res_log),
        "Non-Resonant": (X_test_non_res_scaled, X_test_non_res, y_test_non_res_log),
        "Noisy": (X_test_noisy_scaled, X_test_noisy, y_test_noisy_log),
        "Missing Transits": (X_test_missing_scaled, X_test_missing, y_test_missing_log),
        "Low Mass": (X_test_lowmass_scaled, X_test_lowmass, y_test_lowmass_log),
    }

    # --- Debug: Check Linear Model on Missing Transits ---
    print("\n--- DEBUG: Missing Transits Linear Model ---")
    preds_linear_missing = linreg.predict(X_test_missing_scaled[:5])
    print("First 5 Linear predictions (log scale):", preds_linear_missing)
    print("First 5 true y_test (log scale):", y_test_missing_log[:5])
    print("MAE for first 5:", mean_absolute_error(y_test_missing_log[:5], preds_linear_missing))
    print("NaN in X_test_missing_scaled:", np.isnan(X_test_missing_scaled).sum())
    print("Inf in X_test_missing_scaled:", np.isinf(X_test_missing_scaled).sum())
    print("NaN in y_test_missing_log:", np.isnan(y_test_missing_log).sum())
    print("Inf in y_test_missing_log:", np.isinf(y_test_missing_log).sum())

    results = {"Median": {}, "Linear": {}, "Ridge": {}, "XGBoost": {}, "NN": {}}
    for name, (X_scaled_set, X_raw_set, y_set) in test_sets_raw.items():
        results["Median"][name], _ = evaluate_model(None, X_scaled_set, y_set, "median")
        results["Linear"][name], _ = evaluate_model(linreg, X_raw_set, y_set, "sklearn")    #X_scaled_set
        results["Ridge"][name], _ = evaluate_model(ridge, X_raw_set, y_set, "sklearn")  #X_scaled_set
        results["XGBoost"][name], _ = evaluate_model(model_bdt, X_raw_set, y_set, "sklearn")
        results["NN"][name], _ = evaluate_model(model_nn, X_scaled_set, y_set, "nn")

    #print("\n--- Model Comparison (MAE, dex) ---")
    #header = f"{'Test Set':<22}" + "".join(f"{m:<12}" for m in results.keys())
    #print(header)
    #print("-" * len(header))
    #for name in test_sets_raw.keys():
    #    row = f"{name:<22}" + "".join(f"{results[m][name]:<12.4f}" for m in results.keys())
    #    print(row)

    # NEW CODE:
    header = f"{'Test Set':<22}" + "".join(f"{m:>12}" for m in results.keys())
    print(header)
    print("-" * len(header))
    for name in test_sets_raw.keys():
        row = f"{name:<22}"
        for m in results.keys():
            mae_val = results[m][name]
            row += f"{mae_val:>12.4f}"  # Right-align, fixed width
        print(row)

    # Quantile predictor summary
    print("\n--- Quantile Predictor Performance ---")
    print(f"{'Test Set':<22}{'MAE (dex)':<12}{'Coverage (%)':<14}{'Range Width':<12}")
    quantile_results = {}
    for name, (X_scaled_set, _, y_set) in test_sets_raw.items():
        mae_q, cov_q, width_q, _ = evaluate_quantile_model(model_quantile, X_scaled_set, y_set)
        quantile_results[name] = (mae_q, cov_q, width_q)
        print(f"{name:<22}{mae_q:<12.4f}{cov_q:<14.1f}{width_q:<12.4f}")

    mae_nn = results["NN"]["Original"]
    mae_bdt = results["XGBoost"]["Original"]
    print("\n" + "="*30)
    print(f"FINAL PERFORMANCE ({NUM_SIMULATIONS} SIMS, {MAX_TRANSITS} TRANSITS EACH)")
    print(f"Median baseline MAE: {results['Median']['Original']:.4f} dex")
    print(f"Linear reg MAE:      {results['Linear']['Original']:.4f} dex")
    print(f"Ridge MAE:           {results['Ridge']['Original']:.4f} dex")
    print(f"XGBoost BDT MAE:     {mae_bdt:.4f} dex")
    print(f"Neural Network MAE:  {mae_nn:.4f} dex")
    print("="*30)

    # -----------------------------------------------------------------------------
    # PLOTS
    # -----------------------------------------------------------------------------

    plt.figure(figsize=(10, 6))
    plt.plot(ttvs_train[0] * SIM_TIME_TO_DAYS, 'o-', label='TTV Signal')
    plt.title(f'Sample TTV Signal (Mass = {y_train[0] * SOLAR_TO_EARTH:.2f} Earth Masses)')
    plt.xlabel('Transit Number')
    plt.ylabel('Time Variation (Days)')
    plt.legend()
    plt.grid(True)

    print(f"Training (NN): {X_train_final_scaled.shape}, {y_train_nn.shape}")
    print(f"Validation (NN): {X_val_nn_scaled.shape}, {y_val_nn.shape}")
    print(f"Test (Original): {X_test_original_scaled.shape}, {y_test_original_log.shape}")

    # --- Feature Importance Analysis ---
    # True layout: [amp(1), fft_real(20), fft_imag(20), P_b, ratio, e_b] -> 44 features
    feature_names = (
        ["TTV Amp"]
        + [f"FFT_Real_{i}" for i in range(20)]
        + [f"FFT_Imag_{i}" for i in range(19)]
        + ["Period_B", "Period_Ratio", "Ecc_B"]
    )
    importances = model_bdt.feature_importances_
    indices = np.argsort(importances)[-10:]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align='center', color='teal')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("XGBoost Feature Importance Score")
    plt.title("Which features helped predict the Perturber Mass?")
    plt.tight_layout()

    # 1. LEARNING CURVE COMPARISON
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(nn_train_loss, label='Train Loss', color='blue')
    ax[0].plot(nn_val_loss, label='Val Loss', color='red', linestyle='--')
    ax[0].set_title("Point NN: Huber Loss")
    ax[0].set_xlabel("Epoch"); ax[0].legend()

    ax[1].plot(q_train_loss, label='Train Loss', color='blue')
    ax[1].plot(q_val_loss, label='Val Loss', color='red', linestyle='--')
    ax[1].set_title("Quantile NN: Pinball Loss")
    ax[1].set_xlabel("Epoch"); ax[1].legend()

    bdt_results = model_bdt.evals_result()
    ax[2].plot(bdt_results['validation_0']['rmse'], label='Train RMSE', color='green')
    ax[2].plot(bdt_results['validation_1']['rmse'], label='Val RMSE', color='orange', linestyle='--')
    ax[2].set_title("BDT: RMSE Learning Curve")
    ax[2].set_xlabel("Number of Trees"); ax[2].legend()

    plt.suptitle("Figure 1: Training Convergence & Overfitting Check", fontsize=14)
    plt.tight_layout()

    # 2. PREDICTED VS ACTUAL
    model_nn.eval()
    with torch.no_grad():
        preds_nn = model_nn(torch.tensor(X_test_original_scaled, dtype=torch.float32)).numpy().flatten()
    preds_bdt = model_bdt.predict(X_test_original)

    y_test_earth = 10**y_test_original_log * SOLAR_TO_EARTH
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True, sharex=True)
    ax[0].scatter(y_test_earth, 10**preds_nn * SOLAR_TO_EARTH, alpha=0.3, color='blue')
    ax[0].plot([y_test_earth.min(), y_test_earth.max()], [y_test_earth.min(), y_test_earth.max()], 'r--')
    ax[0].set_title(f"NN (MAE: {mae_nn:.3f} dex)")
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_ylabel("Predicted Mass (Earths)")

    ax[1].scatter(y_test_earth, 10**preds_bdt * SOLAR_TO_EARTH, alpha=0.3, color='green')
    ax[1].plot([y_test_earth.min(), y_test_earth.max()], [y_test_earth.min(), y_test_earth.max()], 'r--')
    ax[1].set_title(f"BDT (MAE: {mae_bdt:.3f} dex)")
    plt.suptitle("Figure 2: Mass Prediction Accuracy", fontsize=14)
    plt.tight_layout()

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

    # 4. ERROR DISTRIBUTION
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    ax[0].hist(res_nn, bins=30, color='blue', alpha=0.6, edgecolor='black')
    ax[0].set_title("NN Error Distribution")
    ax[0].set_xlabel("Error %")

    ax[1].hist(res_bdt, bins=30, color='green', alpha=0.6, edgecolor='black')
    ax[1].set_title("BDT Error Distribution")
    ax[1].set_xlabel("Error %")
    plt.suptitle("Figure 4: Bias and Variance Comparison", fontsize=14)

    # 5. RESONANCE ANALYSIS (Error vs Period Ratio)
    # FIX: period ratio lives at column 42 (0:amp, 1-20:fft_real, 21-40:fft_imag,
    # 41:P_b, 42:ratio, 43:e_b) -- was incorrectly read from column 43 before.
    test_ratios = X_test_original[:, 41]

    resonances_marked = [1.5, 2.0, 3.0]
    res_labels = ["3:2", "2:1", "3:1"]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    im = None
    for i, axes in enumerate(ax):
        current_res = res_nn if i == 0 else res_bdt
        title = "NN: Error vs Resonance" if i == 0 else "BDT: Error vs Resonance"
        im = axes.scatter(test_ratios, np.abs(current_res), c=y_test_earth,
                           cmap='viridis', norm=mcolors.LogNorm(), alpha=0.5)
        for val, label in zip(resonances_marked, res_labels):
            axes.axvline(val, color='red', linestyle='--', alpha=0.6, lw=1)
            axes.text(val, axes.get_ylim()[1]*0.9, label, color='red',
                       fontsize=9, ha='center', fontweight='bold')
        axes.set_title(title)
        axes.set_xlabel("Period Ratio ($P_c/P_b$)")
        if i == 0:
            axes.set_ylabel("Absolute Error (%)")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='True Mass ($M_{Earth}$)')
    plt.suptitle("Figure 5: Physics Check - Model Performance near Orbital Resonances", fontsize=14, y=1.05)

    # 6. Predicted vs actual across ALL test sets (NN), including the two new ones
    for name, (X_scaled_set, _, y_set) in test_sets_raw.items():
        model_nn.eval()
        with torch.no_grad():
            preds = model_nn(torch.tensor(X_scaled_set, dtype=torch.float32)).numpy().flatten()
        y_earth = 10 ** y_set * SOLAR_TO_EARTH
        preds_earth = 10 ** preds * SOLAR_TO_EARTH

        plt.figure(figsize=(8, 5))
        plt.scatter(y_earth, preds_earth, alpha=0.3, color='blue')
        plt.plot([y_earth.min(), y_earth.max()], [y_earth.min(), y_earth.max()], 'r--')
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel("Actual Mass (Earth Masses)")
        plt.ylabel("Predicted Mass (Earth Masses)")
        plt.title(f"NN: {name} (MAE: {results['NN'][name]:.3f} dex)")
        plt.grid(alpha=0.3)

    # 7. Quantile predictions vs actual (original test set)
    y_test_earth = 10 ** y_test_original_log * SOLAR_TO_EARTH
    preds_quantile = model_quantile(torch.tensor(X_test_original_scaled, dtype=torch.float32)).detach().numpy()
    lower_earth = 10 ** preds_quantile[:, 0] * SOLAR_TO_EARTH
    median_earth = 10 ** preds_quantile[:, 1] * SOLAR_TO_EARTH
    upper_earth = 10 ** preds_quantile[:, 2] * SOLAR_TO_EARTH

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_earth, median_earth, alpha=0.3, color='blue', label='Median Prediction')
    order = np.argsort(y_test_earth)
    plt.fill_between(
        y_test_earth[order], lower_earth[order], upper_earth[order],
        alpha=0.2, color='blue', label='10th-90th Percentile Range'
    )
    plt.plot([y_test_earth.min(), y_test_earth.max()], [y_test_earth.min(), y_test_earth.max()], 'r--')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("Actual Mass (Earth Masses)")
    plt.ylabel("Predicted Mass (Earth Masses)")
    plt.title(f"Quantile Predictor: Median +/- Uncertainty (Coverage: {quantile_results['Original'][1]:.1f}%)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()

    print("\n--- DEBUG: Retraining Linear Model ---")
    print("X_train_final_scaled shape:", X_train_final_scaled.shape)
    print("X_train_final_scaled min/max:", X_train_final_scaled.min(), X_train_final_scaled.max())
    print("y_train_nn shape:", y_train_nn.shape)
    print("y_train_nn min/max:", y_train_nn.min(), y_train_nn.max())

    # Retrain with explicit checks
    linreg = LinearRegression()
    linreg.fit(X_train_final_scaled, y_train_nn)
    print("Linear model coefficients:", linreg.coef_)
    print("Linear model intercept:", linreg.intercept_)

    # Predict on training data
    train_preds = linreg.predict(X_train_final_scaled[:5])
    print("First 5 training predictions:", train_preds)
    print("First 5 true y_train:", y_train_nn[:5])