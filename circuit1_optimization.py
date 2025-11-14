import torch
torch.set_num_threads(4)
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import random
import copy
import os
from circuit0_definitions import *

os.makedirs("best_models", exist_ok=True)
os.makedirs("plots_opt_results", exist_ok=True)

seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Note: dt is defined in circuit0_definitions.py
train_loops = 250
simtime_train = 10
simtime_test = 60
t_train = torch.linspace(0., simtime_train, int(simtime_train/dt))
t_test = torch.linspace(0., simtime_test, int(simtime_test/dt))
cutoff_time = 1.0  # seconds
cutoff_idx = (t_train > cutoff_time).nonzero(as_tuple=True)[0][0]

# Run simulation
y0 = torch.zeros(12)

model = CorticalMicrocircuit(params_init, param_bounds, params_init_ou, params_bounds_ou)
optimizer = optim.Adam(model.parameters(), lr=0.05)

# Training loop
loss_across_loop = []
models_across_loop = []
for step in range(train_loops):
    optimizer.zero_grad()
    
    # Setup OU parameters for optimization
    ou_drive_pyr2_3 = generate_ou_noise(
        model.mu_pyr2_3.clamp(*params_bounds_ou[0]),
        model.theta_pyr2_3.clamp(*params_bounds_ou[1]),
        model.sigma_pyr2_3.clamp(*params_bounds_ou[2]),
        t_train, dt)
    ou_drive_pyr5 = generate_ou_noise(
        model.mu_pyr5.clamp(*params_bounds_ou[3]),
        model.theta_pyr5.clamp(*params_bounds_ou[4]),
        model.sigma_pyr5.clamp(*params_bounds_ou[5]),
        t_train, dt)
    ou_drive_st = generate_ou_noise(
        model.mu_st.clamp(*params_bounds_ou[6]),
        model.theta_st.clamp(*params_bounds_ou[7]),
        model.sigma_st.clamp(*params_bounds_ou[8]),
        t_train, dt)
    ou_drive_PV = generate_ou_noise(
        model.mu_PV.clamp(*params_bounds_ou[9]),
        model.theta_PV.clamp(*params_bounds_ou[10]),
        model.sigma_PV.clamp(*params_bounds_ou[11]),
        t_train, dt)
    ou_drive_SST = generate_ou_noise(
        model.mu_SST.clamp(*params_bounds_ou[12]),
        model.theta_SST.clamp(*params_bounds_ou[13]),
        model.sigma_SST.clamp(*params_bounds_ou[14]),
        t_train, dt)
    ou_drive_VIP = generate_ou_noise(
        model.mu_VIP.clamp(*params_bounds_ou[15]),
        model.theta_VIP.clamp(*params_bounds_ou[16]),
        model.sigma_VIP.clamp(*params_bounds_ou[17]),
        t_train, dt)
    ou_drive = torch.stack([ou_drive_pyr2_3, ou_drive_pyr5, ou_drive_st, ou_drive_PV, ou_drive_SST, ou_drive_VIP], dim=1)
    model.set_external_drive(ou_drive)
    
    # Solve ODE using torchdiffeq
    y = odeint(model, y0, t_train, method='dopri5', rtol=1e-3, atol=1e-4)
    
    # Compute EEG signal (using torch operations)
    t_trimmed = t_train[cutoff_idx:]
    
    # Firing Rate (Hz) Targets - see Yao et al 2022, Guet-McCreight et al 2022, Yu et al 2019
    target0 = 1.0 # Pyr2_3
    target1 = 2.0 # Pyr5
    target2 = 0.5 # Stellate
    target3 = 15.0 # PV
    target4 = 10.0 # SST
    target5 = 10.0 # VIP
    
    v0 = y[cutoff_idx:, 0] # Pyr2_3
    v1 = y[cutoff_idx:, 1] # Pyr5
    v2 = y[cutoff_idx:, 2] # Stellate
    v3 = y[cutoff_idx:, 3] # PV
    v4 = y[cutoff_idx:, 4] # SST
    v5 = y[cutoff_idx:, 5] # VIP
    
    means = torch.stack([
        torch.mean(v0), torch.mean(v1), torch.mean(v2),
        torch.mean(v3), torch.mean(v4), torch.mean(v5)
    ])
    targets = torch.tensor([target0, target1, target2, target3, target4, target5])
    pop_rate_weights = torch.tensor([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    relative_errors = (means - targets) / targets
    rate_loss = 5 * torch.sum(pop_rate_weights * relative_errors**2) / torch.sum(pop_rate_weights)
    
    signals = torch.stack([v0, v1, v2, v3, v4, v5], dim=0)
    signals = (signals - signals.mean(dim=1, keepdim=True))
    stds = signals.std(dim=1, keepdim=True)
    signals = signals / (stds + 1e-6)
    corr_matrix = signals @ signals.T / signals.size(1)
    #corr_matrix = torch.corrcoef(signals)
    off_diag = corr_matrix - torch.eye(corr_matrix.shape[0])
    sync_penalty = 5 * off_diag.abs().mean()
    
    v = torch.stack([v0, v1, v2, v3, v4, v5])  # shape [6]
    params_list = list(model.parameters())
    popsizes_tensors = params_list[0][-6:]
    popsizes = torch.tensor([p.detach().float().view(-1).sum() for p in popsizes_tensors])
    N_total = popsizes.sum()
    signs = torch.tensor([1, 1, 1, -1, -1, -1], dtype=v.dtype) # exc = +, inh = -
    pop_size_weights = popsizes / N_total
    eeg = (signs * pop_size_weights) @ v # normalize the contributions of each population by their population sizes
    #eeg = v0 + v1 + v2 - v3 - v4 - v5
    eeg = eeg - eeg.mean()
        
    # Compute power spectrum with torch (or approximate)
    # For simplicity, just do FFT and find peak freq
    fft_vals = torch.fft.fft(eeg)
    freqs = torch.fft.fftfreq(eeg.size(0), d=(t_trimmed[1] - t_trimmed[0]).item())
    power = torch.abs(fft_vals)**2
    
    # Only positive frequencies
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    power_pos = power[pos_mask]
    weighted_freq = (freqs_pos * power_pos).sum() / power_pos.sum()
    weighted_freq_diff = 0.05 * (weighted_freq - 10.0)**2
    
    alpha_band = (freqs_pos >= 8.0) & (freqs_pos <= 12.0)
    high_band = (freqs_pos > 13.0)
    low_band = (freqs_pos < 7.0)
    
    alpha_power = power_pos[alpha_band].sum()
    side_power = power_pos[high_band | low_band].sum()
    log_ratio_term = 5 * torch.log((alpha_power + 1e-8) / (side_power + 1e-8))
    #total_power = power_pos.sum()
    #ou_energy = torch.mean(ou_drive**2)
    
    loss = - log_ratio_term + weighted_freq_diff + sync_penalty + rate_loss# + 0.01 * ou_energy - torch.log(alpha_power / total_power)
    
    loss.backward()
    optimizer.step()
    
    loss_across_loop.append(loss.item())
    models_across_loop.append(copy.deepcopy(model.state_dict()))
    
    print(f"Step {step+1} Loss: {loss.item():.4f}")
    print(f"	Alpha Frequency Loss: {weighted_freq_diff.item():.2f}")
    print(f"	Alpha Power Loss: {-log_ratio_term.item():.4f}")
    print(f"	Rate loss: {rate_loss.item():.4f}")
    print(f"	Sync Penalty: {sync_penalty.item():.4f}")
    print(f"	Weighted Frequency: {weighted_freq.item():.2f} Hz")
    print(f"	L2/3 Pyr: {torch.mean(v0).item():.4f} Hz")
    print(f"	L5 Pyr: {torch.mean(v1).item():.4f} Hz")
    print(f"	Stellate: {torch.mean(v2).item():.4f} Hz")
    print(f"	PV: {torch.mean(v3).item():.4f} Hz")
    print(f"	SST: {torch.mean(v4).item():.4f} Hz")
    print(f"	VIP: {torch.mean(v5).item():.4f} Hz")

# Save best models
models_sorted = [x for _, x in sorted(zip(loss_across_loop, models_across_loop))]
np.save("best_models/sorted_loss_values.npy", np.array(sorted(loss_across_loop)))

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(loss_across_loop)+1), loss_across_loop,c='k')
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.yscale('log')
plt.tight_layout()
plt.savefig("plots_opt_results/loss",dpi=300,transparent=True)
plt.close()

num_models2save = 10
for mind, model2save in enumerate(models_sorted):
	if mind >= num_models2save: break
	model.load_state_dict(model2save)
	torch.save(model.state_dict(), "best_models/optimized_microcircuit_"+str(mind)+".pt")
	
	best_model_state = models_sorted[mind]
	
	# Load best model
	model.load_state_dict(best_model_state)
			
	optimized_values = model.params.detach().numpy()
	optimized_ou_params = torch.stack([
					model.mu_pyr2_3, model.theta_pyr2_3, model.sigma_pyr2_3,
					model.mu_pyr5, model.theta_pyr5, model.sigma_pyr5,
					model.mu_st, model.theta_st, model.sigma_st,
					model.mu_PV, model.theta_PV, model.sigma_PV,
					model.mu_SST, model.theta_SST, model.sigma_SST,
					model.mu_VIP, model.theta_VIP, model.sigma_VIP
					]).detach().numpy()
	
	print("Model Rank:", mind)
	print("    Optimized parameters:", optimized_values)
	print("    Optimized OU parameters:", optimized_ou_params)
	
	# Create a new model instance with optimized params
	optimized_params_tensor = torch.tensor(optimized_values, dtype=torch.float32)
	optimized_ou_params_tensor = torch.tensor(optimized_ou_params, dtype=torch.float32)
	optimized_model = CorticalMicrocircuit(optimized_params_tensor, param_bounds, optimized_ou_params_tensor, params_bounds_ou)
	
	# Initial state
	y0 = torch.zeros(12)
	
	# Regenerate OU noise using optimized parameters
	ou_drive_pyr2_3 = generate_ou_noise(
		optimized_model.mu_pyr2_3.clamp(*params_bounds_ou[0]),
		optimized_model.theta_pyr2_3.clamp(*params_bounds_ou[1]),
		optimized_model.sigma_pyr2_3.clamp(*params_bounds_ou[2]),
		t_test, dt)
	ou_drive_pyr5 = generate_ou_noise(
		optimized_model.mu_pyr5.clamp(*params_bounds_ou[3]),
		optimized_model.theta_pyr5.clamp(*params_bounds_ou[4]),
		optimized_model.sigma_pyr5.clamp(*params_bounds_ou[5]),
		t_test, dt)
	ou_drive_st = generate_ou_noise(
		optimized_model.mu_st.clamp(*params_bounds_ou[6]),
		optimized_model.theta_st.clamp(*params_bounds_ou[7]),
		optimized_model.sigma_st.clamp(*params_bounds_ou[8]),
		t_test, dt)
	ou_drive_PV = generate_ou_noise(
		optimized_model.mu_PV.clamp(*params_bounds_ou[9]),
		optimized_model.theta_PV.clamp(*params_bounds_ou[10]),
		optimized_model.sigma_PV.clamp(*params_bounds_ou[11]),
		t_test, dt)
	ou_drive_SST = generate_ou_noise(
		optimized_model.mu_SST.clamp(*params_bounds_ou[12]),
		optimized_model.theta_SST.clamp(*params_bounds_ou[13]),
		optimized_model.sigma_SST.clamp(*params_bounds_ou[14]),
		t_test, dt)
	ou_drive_VIP = generate_ou_noise(
		optimized_model.mu_VIP.clamp(*params_bounds_ou[15]),
		optimized_model.theta_VIP.clamp(*params_bounds_ou[16]),
		optimized_model.sigma_VIP.clamp(*params_bounds_ou[17]),
		t_test, dt)
	ou_drive = torch.stack([ou_drive_pyr2_3, ou_drive_pyr5, ou_drive_st, ou_drive_PV, ou_drive_SST, ou_drive_VIP], dim=1)
	optimized_model.set_external_drive(ou_drive)
	
	# Solve ODE using optimized parameters
	with torch.no_grad():
		y = odeint(optimized_model, y0, t_test, method='dopri5', rtol=1e-3, atol=1e-4)
	
	y_np = y.numpy()
	t_np = t_test.numpy()
	
	t_trimmed = t_np[cutoff_idx:]
	
	v0 = y_np[cutoff_idx:, 0]
	v1 = y_np[cutoff_idx:, 1]
	v2 = y_np[cutoff_idx:, 2]
	v3 = y_np[cutoff_idx:, 3]
	v4 = y_np[cutoff_idx:, 4]
	v5 = y_np[cutoff_idx:, 5]
	
	fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
	populations = [
		(v0, 'k', 'L2/3 Pyr.'),
		(v1, 'b', 'L5 Pyr.'),
		(v2, 'y', 'L4 Stellate'),
		(v3, 'forestgreen', 'PV'),
		(v4, 'crimson', 'SST'),
		(v5, 'darkorange', 'VIP')
	]
	for ax, (pop, color, label) in zip(axs, populations):
		ax.plot(t_trimmed, pop, c=color, label=label)
		ax.legend(loc='upper right')
		ax.set_ylabel('Mean Rate (Hz)')
		ax.ticklabel_format(style='plain', useOffset=False, axis='y')
	axs[-1].set_xlabel('Time (s)')
	fig.suptitle("Microcircuit Output")
	fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
	plt.savefig("plots_opt_results/microcircuit-output-subplots_"+str(mind)+".png", dpi=300, transparent=True)
	plt.close()
	
	v0_y = y[cutoff_idx:, 0]
	v1_y = y[cutoff_idx:, 1]
	v2_y = y[cutoff_idx:, 2]
	v3_y = y[cutoff_idx:, 3]
	v4_y = y[cutoff_idx:, 4]
	v5_y = y[cutoff_idx:, 5]
	
	v = torch.stack([v0_y, v1_y, v2_y, v3_y, v4_y, v5_y])  # shape [6]
	params_list = list(optimized_model.parameters())
	popsizes_tensors = params_list[0][-6:]
	popsizes = torch.tensor([p.detach().float().view(-1).sum() for p in popsizes_tensors])
	N_total = popsizes.sum()
	signs = torch.tensor([1, 1, 1, -1, -1, -1], dtype=v.dtype) # exc = +, inh = -
	pop_size_weights = popsizes / N_total
	EEG_signal = (signs * pop_size_weights) @ v # normalize the contributions of eachpopulation by their population sizes
	#EEG_signal = v0 + v1 + v2 - v3 - v4 - v5
	EEG_signal = EEG_signal - EEG_signal.mean()
	EEG_signal = EEG_signal.detach().cpu().numpy()
	
	plt.plot(t_trimmed, EEG_signal, c='k')
	plt.title("EEG")
	plt.xlabel("Time (s)")
	plt.ylabel("Mean Rate (Hz)")
	plt.tight_layout()
	plt.savefig("plots_opt_results/microcircuit-EEG-output_"+str(mind)+".png",dpi=300,transparent=True)
	plt.close()
	
	# Plot power spectrum
	fs = 1/(t_trimmed[1]-t_trimmed[0])
	nperseg = len(EEG_signal) // 8
	f, Pxx = welch(EEG_signal, fs=fs, nperseg=nperseg)
	plt.figure(figsize=(6, 4))
	plt.loglog(f, Pxx)
	plt.title("EEG Power Spectrum")
	plt.xlabel("Frequency (Hz)")
	plt.ylabel("Power")
	plt.tight_layout()
	plt.savefig("plots_opt_results/microcircuit-PSD-output_"+str(mind)+".png",dpi=300,transparent=True)
	plt.close()
