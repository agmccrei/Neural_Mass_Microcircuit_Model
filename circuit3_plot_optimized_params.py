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

os.makedirs("plots_analysis", exist_ok=True)

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':26})

seed = 1234
Num_best_models = 10 # Model rank to run/plot

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

param_labels = [r'$A_{L23Pyr→L23Pyr}$', r'$A_{L23Pyr→L5Pyr}$', r'$A_{L23Pyr→Stellate}$', r'$A_{L23Pyr→PV}$', r'$A_{L23Pyr→SST}$', r'$A_{L23Pyr→VIP}$', r'$a_{L23Pyr}$', r'$e0_{L23Pyr}$', r'$v0_{L23Pyr}$', r'$r_{L23Pyr}$', \
r'$A_{L5Pyr→L5Pyr}$', r'$A_{L5Pyr→Stellate}$', r'$A_{L5Pyr→PV}$', r'$A_{L5Pyr→SST}$', r'$a_{L5Pyr}$', r'$e0_{L5Pyr}$', r'$v0_{L5Pyr}$', r'$r_{L5Pyr}$', \
r'$A_{Stellate→L23Pyr}$', r'$A_{Stellate→L5Pyr}$', r'$A_{Stellate→Stellate}$', r'$A_{Stellate→PV}$', r'$A_{Stellate→SST}$', r'$a_{Stellate}$', r'$e0_{Stellate}$', r'$v0_{Stellate}$', r'$r_{Stellate}$', \
r'$B_{PV→L23Pyr}$', r'$B_{PV→L5Pyr}$', r'$B_{PV→Stellate}$', r'$B_{PV→PV}$', r'$B_{PV→SST}$', r'$B_{PV→VIP}$', r'$b_{PV}$', r'$e0_{PV}$', r'$v0_{PV}$', r'$r_{PV}$', \
r'$B_{SST→L23Pyr}$', r'$B_{SST→L5Pyr}$', r'$B_{SST→PV}$', r'$B_{SST→SST}$', r'$B_{SST→VIP}$', r'$b_{SST}$', r'$e0_{SST}$', r'$v0_{SST}$', r'$r_{SST}$', \
r'$B_{VIP→PV}$', r'$B_{VIP→SST}$', r'$B_{VIP→VIP}$', r'$b_{VIP}$', r'$e0_{VIP}$', r'$v0_{VIP}$', r'$r_{VIP}$', \
r'$N_{L23Pyr}$', r'$N_{L5Pyr}$', r'$N_{Stellate}$', r'$N_{PV}$', r'$N_{SST}$', r'$N_{VIP}$']

param_labels_ou = [r'$\mu_{OU,L23Pyr}$',r'$\Theta_{OU,L23Pyr}$',r'$\sigma_{OU,L23Pyr}$',
					r'$\mu_{OU,L5Pyr}$',r'$\Theta_{OU,L5Pyr}$',r'$\sigma_{OU,L5Pyr}$',
					r'$\mu_{OU,Stellate}$',r'$\Theta_{OU,Stellate}$',r'$\sigma_{OU,Stellate}$',
					r'$\mu_{OU,PV}$',r'$\Theta_{OU,PV}$',r'$\sigma_{OU,PV}$',
					r'$\mu_{OU,SST}$',r'$\Theta_{OU,SST}$',r'$\sigma_{OU,SST}$',
					r'$\mu_{OU,VIP}$',r'$\Theta_{OU,VIP}$',r'$\sigma_{OU,VIP}$'
					]

def dotplot(metric,xticklabels,filename,fs):
	dotsize=50
	transparency = 1
	xinds = np.arange(0,len(metric))
	fig_bands, ax_bands = plt.subplots(figsize=fs)
	for cind in range(len(metric)):
		metric_m = np.mean(metric[cind])
		metric_sd = np.std(metric[cind])
		xpositions = xinds[cind]+(np.random.random(len(metric[cind]))*0.4-0.2)
		ax_bands.scatter(xpositions,metric[cind],s=dotsize,facecolor='k',edgecolors='face',alpha=transparency)
		ax_bands.scatter(xpositions[0],metric[cind][0],s=dotsize,facecolor='r',edgecolors='face',alpha=transparency)
		ln1, = ax_bands.plot([xinds[cind]-0.25,xinds[cind]+0.25],[metric_m,metric_m],'k',alpha=1,linewidth=3)
		ln1.set_solid_capstyle('round')
	
	ax_bands.set_xlim(xinds[0]-0.5,xinds[-1]+0.5)
	ax_bands.set_yscale('log')
	ax_bands.set_ylabel('Optimized Magnitude')
	ax_bands.set_xticks(xinds)
	ax_bands.set_xticklabels(xticklabels, rotation = 45, ha="right")
	ax_bands.grid(False)
	ax_bands.spines['right'].set_visible(False)
	ax_bands.spines['top'].set_visible(False)
	fig_bands.tight_layout()
	fig_bands.savefig('plots_analysis/'+filename+'.png',dpi=300,transparent=True)
	plt.close()

params = [[] for _ in param_labels]
params_ou = [[] for _ in param_labels_ou]

for m in range(Num_best_models):
	model = CorticalMicrocircuit(params_init, param_bounds, params_init_ou, params_bounds_ou)
	model.load_state_dict(torch.load("best_models/optimized_microcircuit_"+str(m)+".pt"))
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
	
	for pind, param in enumerate(optimized_values):
		clipped_param = np.clip(param, param_bounds[pind][0], param_bounds[pind][1])
		params[pind].append(clipped_param)
	for pind, param in enumerate(optimized_ou_params):
		clipped_param = np.clip(param, params_bounds_ou[pind][0], params_bounds_ou[pind][1])
		params_ou[pind].append(clipped_param)

l23pyr_param_inds = [[0,10],[0,3]]
l5pyr_param_inds = [[10,18],[3,6]]
stellate_param_inds = [[18,27],[6,9]]
PV_param_inds = [[27,37],[9,12]]
SST_param_inds = [[37,46],[12,15]]
VIP_param_inds = [[46,53],[15,18]]
cellcount_param_inds = [[53,59]]

all_inds = [l23pyr_param_inds, l5pyr_param_inds, stellate_param_inds, PV_param_inds, SST_param_inds, VIP_param_inds, cellcount_param_inds]
filenames = ['opt_model_params_L23Pyr','opt_model_params_L5Pyr','opt_model_params_Stellate','opt_model_params_PV','opt_model_params_SST','opt_model_params_VIP','opt_model_params_counts']

for pair_inds,filename in zip(all_inds,filenames):
	params2plot = []
	paramlabels = []
	for count, inds in enumerate(pair_inds):
		if count == 0: # regular params list
			params2plot.extend(params[inds[0]:inds[1]])
			paramlabels.extend(param_labels[inds[0]:inds[1]])
		if count == 1: # ou
			params2plot.extend(params_ou[inds[0]:inds[1]])
			paramlabels.extend(param_labels_ou[inds[0]:inds[1]])
	
	if filename == 'opt_model_params_counts':
		params2plot = [[p*num_neurons_scaler for p in pa] for pa in params2plot]
		dotplot(params2plot,paramlabels,filename,(8, 6))
	else:
		dotplot(params2plot,paramlabels,filename,(12, 6))



