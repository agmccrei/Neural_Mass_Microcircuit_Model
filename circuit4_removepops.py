import torch
torch.set_num_threads(4)
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import random
import math
import os
from circuit0_definitions import *

os.makedirs("plots_removepops", exist_ok=True)

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':26})

seed = 1234
Model_RANK = 0 # Model rank to run/plot

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

dt = 0.002
simtime_test = 200
t_test = torch.linspace(0., simtime_test, int(simtime_test/dt))
cutoff_time = 10.0  # seconds
cutoff_idx = (t_test > cutoff_time).nonzero(as_tuple=True)[0][0]

model = CorticalMicrocircuit(params_init, param_bounds, params_init_ou, params_bounds_ou)
model.load_state_dict(torch.load("best_models/optimized_microcircuit_"+str(Model_RANK)+".pt"))
model.eval()

optimized_values = model.params.detach().numpy()
optimized_ou_params = torch.stack([
                model.mu_pyr2_3, model.theta_pyr2_3, model.sigma_pyr2_3,
                model.mu_pyr5, model.theta_pyr5, model.sigma_pyr5,
                model.mu_st, model.theta_st, model.sigma_st,
                model.mu_PV, model.theta_PV, model.sigma_PV,
                model.mu_SST, model.theta_SST, model.sigma_SST,
                model.mu_VIP, model.theta_VIP, model.sigma_VIP
                ]).detach().numpy()

print("Optimized parameters:", optimized_values)
print("Optimized OU parameters:", optimized_ou_params)

param_labels = ['A_L23Pyr→L23Pyr', 'A_L23Pyr→L5Pyr', 'A_L23Pyr→Stellate', 'A_L23Pyr→PV', 'A_L23Pyr→SST', 'A_L23Pyr→VIP', 'a_L23Pyr', 'e0_L23Pyr', 'v0_L23Pyr', 'r_L23Pyr', \
'A_L5Pyr→L5Pyr', 'A_L5Pyr→Stellate', 'A_L5Pyr→PV', 'A_L5Pyr→SST', 'a_L5Pyr', 'e0_L5Pyr', 'v0_L5Pyr', 'r_L5Pyr', \
'A_Stellate→L23Pyr', 'A_Stellate→L5Pyr', 'A_Stellate→Stellate', 'A_Stellate→PV', 'A_Stellate→SST', 'a_Stellate', 'e0_Stellate', 'v0_Stellate', 'r_Stellate', \
'B_PV→L23Pyr', 'B_PV→L5Pyr', 'B_PV→Stellate', 'B_PV→PV', 'B_PV→SST', 'B_PV→VIP', 'b_PV', 'e0_PV', 'v0_PV', 'r_PV', \
'B_SST→L23Pyr', 'B_SST→L5Pyr', 'B_SST→PV', 'B_SST→SST', 'B_SST→VIP', 'b_SST', 'e0_SST', 'v0_SST', 'r_SST', \
'B_VIP→PV', 'B_VIP→SST', 'B_VIP→VIP', 'b_VIP', 'e0_VIP', 'v0_VIP', 'r_VIP', \
'N_L23Pyr', 'N_L5Pyr', 'N_Stellate', 'N_PV', 'N_SST', 'N_VIP']

# Nullify population contributions by zeroing e0
e0_inds = [7, 15, 24, 34, 43, 50, None]
output_inds = [[0,1,2,3,4,5,6], [10,11,12,13,14], [18,19,20,21,22,23], [27,28,29,30,31,32,33], [37,38,39,40,41,42], [46,47,48,49], None]
input_inds = [[0,18,27,37], [1,10,19,28,38], [2,11,20,29], [3,12,21,30,39,46], [4,13,22,31,40,47], [5,32,41,48], None]
inds_labels = ['-L23Pyr','-L5Pyr','-Stellate','-PV','-SST','-VIP','Full']
e0_cols = ['cornflowerblue','b','y','forestgreen','crimson','darkorange','k']

fig_psd, ax_psd = plt.subplots(figsize=(10, 7))

for e0ind, outinds, ininds, indlab, e0col in zip(e0_inds, output_inds, input_inds, inds_labels, e0_cols):
	if e0ind is None:
		modified_params = optimized_values.copy()
	else:
		modified_params = optimized_values.copy()
		modified_params[e0ind] = 0
		for o in outinds:
			modified_params[o] = 0
		for i in ininds:
			modified_params[i] = 0
		modified_bounds = param_bounds.copy()
		modified_bounds[e0ind] = (0, modified_bounds[e0ind][1])
		for o in outinds:
			modified_bounds[o] = (0, modified_bounds[o][1])
		for i in ininds:
			modified_bounds[i] = (0, modified_bounds[i][1])

	print('Modified parameters for ' + indlab)
	for labs,mp in zip(param_labels,modified_params):
		print(labs + ': ' + str(mp))
	
	# Create a new model instance with optimized params
	optimized_params_tensor = torch.tensor(modified_params, dtype=torch.float32,requires_grad=False)
	optimized_ou_params_tensor = torch.tensor(optimized_ou_params, dtype=torch.float32,requires_grad=False)
	optimized_model = CorticalMicrocircuit(optimized_params_tensor, modified_bounds, optimized_ou_params_tensor, params_bounds_ou)
	
	# Initial state
	y0 = torch.zeros(12)
	
	# Regenerate OU noise using optimized parameters (clamp to bounds for safety)
	if indlab == '-L23Pyr':
		ou_drive_pyr2_3 = torch.zeros_like(t_test)
	else:
		ou_drive_pyr2_3 = generate_ou_noise(
			optimized_model.mu_pyr2_3.clamp(*params_bounds_ou[0]),
			optimized_model.theta_pyr2_3.clamp(*params_bounds_ou[1]),
			optimized_model.sigma_pyr2_3.clamp(*params_bounds_ou[2]),
			t_test, dt)
	if indlab == '-L5Pyr':
		ou_drive_pyr5 = torch.zeros_like(t_test)
	else:
		ou_drive_pyr5 = generate_ou_noise(
			optimized_model.mu_pyr5.clamp(*params_bounds_ou[3]),
			optimized_model.theta_pyr5.clamp(*params_bounds_ou[4]),
			optimized_model.sigma_pyr5.clamp(*params_bounds_ou[5]),
			t_test, dt)
	if indlab == '-Stellate':
		ou_drive_st = torch.zeros_like(t_test)
	else:
		ou_drive_st = generate_ou_noise(
			optimized_model.mu_st.clamp(*params_bounds_ou[6]),
			optimized_model.theta_st.clamp(*params_bounds_ou[7]),
			optimized_model.sigma_st.clamp(*params_bounds_ou[8]),
			t_test, dt)
	if indlab == '-PV':
		ou_drive_PV = torch.zeros_like(t_test)
	else:
		ou_drive_PV = generate_ou_noise(
			optimized_model.mu_PV.clamp(*params_bounds_ou[9]),
			optimized_model.theta_PV.clamp(*params_bounds_ou[10]),
			optimized_model.sigma_PV.clamp(*params_bounds_ou[11]),
			t_test, dt)
	if indlab == '-SST':
		ou_drive_SST = torch.zeros_like(t_test)
	else:
		ou_drive_SST = generate_ou_noise(
			optimized_model.mu_SST.clamp(*params_bounds_ou[12]),
			optimized_model.theta_SST.clamp(*params_bounds_ou[13]),
			optimized_model.sigma_SST.clamp(*params_bounds_ou[14]),
			t_test, dt)
	if indlab == '-VIP':
		ou_drive_VIP = torch.zeros_like(t_test)
	else:
		ou_drive_VIP = generate_ou_noise(
			optimized_model.mu_VIP.clamp(*params_bounds_ou[15]),
			optimized_model.theta_VIP.clamp(*params_bounds_ou[16]),
			optimized_model.sigma_VIP.clamp(*params_bounds_ou[17]),
			t_test, dt)
	ou_drive = torch.stack([ou_drive_pyr2_3, ou_drive_pyr5, ou_drive_st, ou_drive_PV, ou_drive_SST, ou_drive_VIP], dim=1)
	optimized_model.set_external_drive(ou_drive)
	
	# Solve ODE with torchdiffeq using optimized parameters
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
	
	fig, axs = plt.subplots(6, 1, figsize=(6, 8), sharex=True)
	populations = [
		(v0, 'cornflowerblue', 'L2/3 Pyr.'),
		(v1, 'b', 'L5 Pyr.'),
		(v2, 'y', 'L4 Stellate'),
		(v3, 'forestgreen', 'PV'),
		(v4, 'crimson', 'SST'),
		(v5, 'darkorange', 'VIP')
	]

	for ax, (pop, color, label) in zip(axs, populations):
		ax.plot(t_trimmed, pop, c=color, label=label)
		ax.axis('off')
		
		# don't add scale bars if population is zeroed
		if ((indlab == '-L23Pyr') and (label == 'L2/3 Pyr.')): continue
		if ((indlab == '-L5Pyr') and (label == 'L5 Pyr.')): continue
		if ((indlab == '-Stellate') and (label == 'L4 Stellate')): continue
		if ((indlab == '-PV') and (label == 'PV')): continue
		if ((indlab == '-SST') and (label == 'SST')): continue
		if ((indlab == '-VIP') and (label == 'VIP')): continue
		
		x_range = t_trimmed[-1] - t_trimmed[0]
		y_range = max(pop) - min(pop)
		
		time_scale_unrounded = 0.1 * x_range
		time_order = math.floor(math.log10(abs(time_scale_unrounded)))
		time_scale = round(time_scale_unrounded, -time_order)
		
		psp_scale_unrounded = 0.2 * y_range
		psp_order = math.floor(math.log10(abs(psp_scale_unrounded)))
		psp_scale = round(psp_scale_unrounded, -psp_order)
		
		x_start = t_trimmed[0] + 0.05*x_range
		y_start = min(pop) + 0.05*y_range - psp_scale/2
		
		# Horizontal (time) scale bar
		if label == 'VIP':
			ax.plot([x_start, x_start + time_scale], [y_start, y_start], 'k', lw=4)
			ax.text(x_start + time_scale/2, y_start - 0.1*psp_scale, f'{time_scale}s', ha='center', va='top')
		
		# Vertical (PSP) scale bar
		ax.plot([x_start, x_start], [y_start, y_start + psp_scale], 'k', lw=4)
		ax.text(x_start - 0.02*(t_trimmed[-1]-t_trimmed[0]), y_start + psp_scale/2, f'{psp_scale}', ha='right', va='center')
	
	fig.tight_layout()
	plt.savefig("plots_removepops/microcircuit-output-subplots_"+indlab+"_"+str(Model_RANK)+".png", dpi=300, transparent=True)
	plt.close()

	# mean rates bar plot
	fig, ax = plt.subplots(1, 1, figsize=(12, 6))
	xvals = np.linspace(1,len(populations),len(populations))
	for x, (pop, color, label) in zip(xvals, populations):
		ax.bar(x, np.mean(pop), yerr=np.std(pop), ecolor='black', error_kw={'elinewidth':4,'capthick':4}, capsize=5, facecolor=color, edgecolor='k', linewidth=4, label=label)
		ax.legend(loc='upper left')

	ax.set_yscale('log')
	ax.set_ylabel('Mean Rate (Hz)')
	ax.set_xticks(xvals)
	ax.set_xticklabels([label for _,_,label in populations])
	fig.tight_layout()
	plt.savefig("plots_removepops/microcircuit-output-rates_"+indlab+"_"+str(Model_RANK)+".png", dpi=300, transparent=True)
	plt.close()

	# EEG output
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

	plt.figure(figsize=(10, 4))
	plt.plot(t_trimmed, EEG_signal, c='k')
	plt.axis('off')

	x_range = t_trimmed[-1] - t_trimmed[0]
	y_range = max(EEG_signal) - min(EEG_signal)

	time_scale_unrounded = 0.1 * x_range
	time_order = math.floor(math.log10(abs(time_scale_unrounded)))
	time_scale = round(time_scale_unrounded, -time_order)

	volt_scale_unrounded = 0.2 * y_range
	volt_order = math.floor(math.log10(abs(volt_scale_unrounded)))
	volt_scale = round(volt_scale_unrounded, -volt_order)

	x_start = t_trimmed[0] + 0.05*x_range
	y_start = min(EEG_signal) + 0.05*y_range - volt_scale # give it more buffer from the EEG signal

	# Horizontal
	plt.plot([x_start, x_start + time_scale], [y_start, y_start], 'k', lw=2)
	plt.text(x_start + time_scale/2, y_start - 0.1*volt_scale, f'{time_scale}s', ha='center', va='top')

	# Vertical
	plt.plot([x_start, x_start], [y_start, y_start + volt_scale], 'k', lw=2)
	plt.text(x_start - 0.02*(t_trimmed[-1]-t_trimmed[0]), y_start + volt_scale/2, f'{volt_scale}', ha='right', va='center')

	plt.savefig("plots_removepops/microcircuit-EEG-output_"+indlab+"_"+str(Model_RANK)+".png",dpi=300,transparent=True)
	plt.close()

	# Plot power spectrum
	fs = 1/(t_trimmed[1]-t_trimmed[0])
	nperseg = len(EEG_signal) // 64
	f, Pxx = welch(EEG_signal, fs=fs, nperseg=nperseg)
	ax_psd.loglog(f, Pxx, c=e0col, linewidth=3 if indlab=='Full' else 2)

ax_psd.set_xlim(right=np.max(f))
ax_psd.set_xlabel("Frequency (Hz)")
ax_psd.set_ylabel("Power")
fig_psd.tight_layout()
fig_psd.savefig("plots_removepops/microcircuit-PSD-output_"+str(Model_RANK)+".png",dpi=300,transparent=True)
plt.close()
