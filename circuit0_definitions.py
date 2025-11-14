import torch
torch.set_num_threads(4)
import torch.nn as nn

dt = 0.002

# Sigmoid function
def S(v, e0=2.5, v0=6.0, r=0.56):
    return 2 * e0 / (1 + torch.exp(r * (v0 - v)))

def generate_ou_noise(mu, theta, sigma, t, dt):
    noise = torch.randn(len(t)).detach()
    x = torch.zeros(len(t), device=mu.device)
    x[0] = mu
    for i in range(1, len(t)):
        dx = theta * (mu - x[i-1]) * dt + sigma * dt**0.5 * noise[i]
        x[i] = x[i-1] + dx
    return x

# Cortical microcircuit ODE system
class CorticalMicrocircuit(nn.Module):
    def __init__(self, params, bounds, params_ou, bounds_ou):
        super().__init__()
        self.params = nn.Parameter(params)
        self.bounds = bounds  # list of (min, max) tuples
        self.bounds_ou = bounds_ou
        self.dt = dt
        
        # Thalamic OU parameters (init values)
        self.mu_pyr2_3 = nn.Parameter(params_ou[0])
        self.theta_pyr2_3 = nn.Parameter(params_ou[1])
        self.sigma_pyr2_3 = nn.Parameter(params_ou[2])
        self.mu_pyr5 = nn.Parameter(params_ou[3])
        self.theta_pyr5 = nn.Parameter(params_ou[4])
        self.sigma_pyr5 = nn.Parameter(params_ou[5])
        self.mu_st = nn.Parameter(params_ou[6])
        self.theta_st = nn.Parameter(params_ou[7])
        self.sigma_st = nn.Parameter(params_ou[8])
        self.mu_PV = nn.Parameter(params_ou[9])
        self.theta_PV = nn.Parameter(params_ou[10])
        self.sigma_PV = nn.Parameter(params_ou[11])
        self.mu_SST = nn.Parameter(params_ou[12])
        self.theta_SST = nn.Parameter(params_ou[13])
        self.sigma_SST = nn.Parameter(params_ou[14])
        self.mu_VIP = nn.Parameter(params_ou[15])
        self.theta_VIP = nn.Parameter(params_ou[16])
        self.sigma_VIP = nn.Parameter(params_ou[17])

        self.external_drive = None  # Placeholder for OU input
    
    def set_external_drive(self, drive):
        self.external_drive = drive
    
    def forward(self, t, y):
        # Unpack parameters
        p = torch.stack([
            self.params[i].clamp(*self.bounds[i])
            for i in range(len(self.bounds))
        ])
        
        A_pyr2_3_pyr2_3, A_pyr2_3_pyr5, A_pyr2_3_st, A_pyr2_3_PV, A_pyr2_3_SST, A_pyr2_3_VIP, a_pyr2_3, e0_pyr2_3, v0_pyr2_3, r_pyr2_3, \
        A_pyr5_pyr5, A_pyr5_st, A_pyr5_PV, A_pyr5_SST, a_pyr5, e0_pyr5, v0_pyr5, r_pyr5, \
        A_st_pyr2_3, A_st_pyr5, A_st_st, A_st_PV, A_st_SST, a_st, e0_st, v0_st, r_st, \
        B_PV_pyr2_3, B_PV_pyr5, B_PV_st, B_PV_PV, B_PV_SST, B_PV_VIP, b_PV, e0_PV, v0_PV, r_PV, \
        B_SST_pyr2_3, B_SST_pyr5, B_SST_PV, B_SST_SST, B_SST_VIP, b_SST, e0_SST, v0_SST, r_SST, \
        B_VIP_PV, B_VIP_SST, B_VIP_VIP, b_VIP, e0_VIP, v0_VIP, r_VIP, \
        N_pyr2_3, N_pyr5, N_st, N_PV, N_SST, N_VIP = p
        
        Pyr2_3, Pyr5, Stellate, PV, SST, VIP, \
        dPyr2_3, dPyr5, dStellate, dPV, dSST, dVIP = y
        
        # Compute sigmoid activations with torch
        F_Pyr2_3 = S(Pyr2_3, e0_pyr2_3, v0_pyr2_3, r_pyr2_3)
        F_Pyr5 = S(Pyr5, e0_pyr5, v0_pyr5, r_pyr5)
        F_Stellate = S(Stellate, e0_st, v0_st, r_st)
        F_PV = S(PV, e0_PV, v0_PV, r_PV)
        F_SST = S(SST, e0_SST, v0_SST, r_SST)
        F_VIP = S(VIP, e0_VIP, v0_VIP, r_VIP)
        
        # Thalamic drive
        #t_idx = min(int((t / self.dt).item()), self.external_drive.shape[0] - 1)
        t_idx = int(torch.clamp(t / self.dt, 0, self.external_drive.shape[0] - 1).item())
        
        cortico_cortical_drive_pyr2_3 = self.external_drive[t_idx, 0]
        thalamic_drive_pyr5 = self.external_drive[t_idx, 1]
        thalamic_drive_st = self.external_drive[t_idx, 2]
        thalamic_drive_PV = self.external_drive[t_idx, 3]
        neuromodulatory_drive_SST = self.external_drive[t_idx, 4]
        cortico_cortical_drive_VIP = self.external_drive[t_idx, 5]
        
        # Derivatives
        dy0 = dPyr2_3
        dy1 = dPyr5
        dy2 = dStellate
        dy3 = dPV
        dy4 = dSST
        dy5 = dVIP
        
        dy6 = cortico_cortical_drive_pyr2_3 \
              + A_st_pyr2_3 * a_st * F_Stellate * (N_st/N_pyr2_3) \
              + A_pyr2_3_pyr2_3 * a_pyr2_3 * F_Pyr2_3 \
              - B_PV_pyr2_3 * b_PV * F_PV * (N_PV/N_pyr2_3) \
              - B_SST_pyr2_3 * b_SST * F_SST * (N_SST/N_pyr2_3) \
              - 2 * a_pyr2_3 * dPyr2_3 - a_pyr2_3**2 * Pyr2_3
        
        dy7 = thalamic_drive_pyr5 \
               + A_st_pyr5 * a_st * F_Stellate * (N_st/N_pyr5) \
               + A_pyr2_3_pyr5 * a_pyr2_3 * F_Pyr2_3 * (N_pyr2_3/N_pyr5) \
               + A_pyr5_pyr5 * a_pyr5 * F_Pyr5 \
               - B_PV_pyr5 * b_PV * F_PV * (N_PV/N_pyr5) \
               - B_SST_pyr5 * b_SST * F_SST * (N_SST/N_pyr5) \
               - 2 * a_pyr5 * dPyr5 - a_pyr5**2 * Pyr5
        
        dy8 = thalamic_drive_st \
               + A_pyr2_3_st * a_pyr2_3 * F_Pyr2_3 * (N_pyr2_3/N_st) \
               + A_pyr5_st * a_pyr5 * F_Pyr5 * (N_pyr5/N_st) \
               - B_PV_st * b_PV * F_PV * (N_PV/N_st) \
               - 2 * a_st * dStellate - a_st**2 * Stellate
               #- B_SST_st * b_SST * F_SST * (N_SST/N_st) \

        dy9 = thalamic_drive_PV \
              + A_pyr2_3_PV * a_pyr2_3 * F_Pyr2_3 * (N_pyr2_3/N_PV) \
              + A_pyr5_PV * a_pyr5 * F_Pyr5 * (N_pyr5/N_PV) \
              + A_st_PV * a_st * F_Stellate * (N_st/N_PV) \
              - B_PV_PV * b_PV * F_PV \
              - B_SST_PV * b_SST * F_SST * (N_SST/N_PV) \
              - B_VIP_PV * b_VIP * F_VIP * (N_VIP/N_PV) \
              - 2 * b_PV * dPV - b_PV**2 * PV
        
        dy10 = neuromodulatory_drive_SST \
              + A_pyr2_3_SST * a_pyr2_3 * F_Pyr2_3 * (N_pyr2_3/N_SST) \
              + A_pyr5_SST * a_pyr5 * F_Pyr5 * (N_pyr5/N_SST) \
              + A_st_SST * a_st * F_Stellate * (N_st/N_SST) \
              - B_PV_SST * b_PV * F_PV * (N_PV/N_SST) \
              - B_SST_SST * b_SST * F_SST \
              - B_VIP_SST * b_VIP * F_VIP * (N_VIP/N_SST) \
              - 2 * b_SST * dSST - b_SST**2 * SST
        
        dy11 = cortico_cortical_drive_VIP \
              + A_pyr2_3_VIP * a_pyr2_3 * F_Pyr2_3 * (N_pyr2_3/N_VIP) \
              - B_PV_VIP * b_PV * F_PV * (N_PV/N_VIP) \
              - B_SST_VIP * b_SST * F_SST * (N_SST/N_VIP) \
              - B_VIP_VIP * b_VIP * F_VIP \
              - 2 * b_VIP * dVIP - b_VIP**2 * VIP
              #+ A_pyr5_VIP * a_pyr5 * F_Pyr5 * (N_pyr5/N_VIP) \
              #+ A_st_VIP * a_st * F_Stellate * (N_st/N_VIP) \

        return torch.stack([dy0, dy1, dy2, dy3, dy4, dy5,
                            dy6, dy7, dy8, dy9, dy10, dy11])

# Optimizer variables
num_neurons_scaler = 1000 # order magnitude is too high otherwise and will barely change during optimization - all affect the simulation as proportions though, so shouldn't have to scale it back
params_init = torch.tensor([
    8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 100.0, 6.0, 2.5, 0.56,
    10.5, 10.5, 10.5, 10.5, 80.0, 6.0, 3.0, 0.6,
    8.0, 8.0, 8.0, 8.0, 8.0, 130.0, 6.0, 2.0, 0.7,
    3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 40.0, 3.5, 3.5, 0.6,
    3.0, 3.0, 3.0, 3.0, 3.0, 40.0, 3.5, 3.5, 0.6,
    3.0, 3.0, 3.0, 30.0, 2.5, 4.0, 0.4,
    10000/num_neurons_scaler, 10000/num_neurons_scaler, 10000/num_neurons_scaler, 1000/num_neurons_scaler, 1000/num_neurons_scaler, 500/num_neurons_scaler
], dtype=torch.float32)
param_bounds = [
    # A_pyr2_3 X 6,     a_pyr2_3,    e0_pyr2_3,  v0_pyr2_3,  r_pyr2_3,  N_pyr2_3
    (2.0, 60.0),(2.0, 60.0),(2.0, 60.0),(2.0, 60.0),(2.0, 60.0),(2.0, 60.0),     (10.0, 300.0), (1.0, 10.0), (0.5, 5.0), (0.2, 0.8),
    # A_pyr5 X 4,       a_pyr5,      e0_pyr5,    v0_pyr5,    r_pyr5,    N_pyr5
    (2.0, 60.0),(2.0, 60.0),(2.0, 60.0),(2.0, 60.0),     (10.0, 300.0), (1.0, 10.0), (0.5, 5.0), (0.2, 0.8),
    # A_st X 5,         a_st,        e0_st,      v0_st,      r_st,      N_st
    (1.0, 60.0),(1.0, 60.0),(1.0, 60.0),(1.0, 60.0),(1.0, 60.0),     (10.0, 300.0), (1.0, 10.0), (0.5, 5.0), (0.2, 0.8),
    # B_PV X 6,         b_PV,        e0_PV,      v0_PV,      r_PV,      N_PV
    (2.0, 60.0),(2.0, 60.0),(2.0, 60.0),(2.0, 60.0),(2.0, 60.0),(2.0, 60.0),   (8.0, 300.0),  (3.0, 15.0), (0.5, 8.0), (0.2, 1.0),
    # B_SST X 5,        b_SST,       e0_SST,     v0_SST,     r_SST,     N_SST
    (2.0, 60.0),(2.0, 60.0),(2.0, 60.0),(2.0, 60.0),(2.0, 60.0),   (8.0, 300.0),  (3.0, 15.0), (0.5, 8.0), (0.2, 0.8),
    # B_VIP X 3,        b_VIP,       e0_VIP,     v0_VIP,     r_VIP,     N_VIP
    (2.0, 60.0),(2.0, 60.0),(2.0, 60.0),    (8.0, 300.0),  (2.5, 10.0), (0.5, 8.0), (0.2, 0.8),
    (5000/num_neurons_scaler,20000/num_neurons_scaler),(5000/num_neurons_scaler,20000/num_neurons_scaler),(5000/num_neurons_scaler,20000/num_neurons_scaler),(500/num_neurons_scaler,5000/num_neurons_scaler),(500/num_neurons_scaler,5000/num_neurons_scaler),(250/num_neurons_scaler,5000/num_neurons_scaler)
]

params_init_ou = torch.tensor([15.0, 4.0, 0.4,
								15.0, 4.0, 0.4,
								15.0, 4.0, 0.4,
								15.0, 4.0, 0.4,
								0.0, 0.8, 0.0,
								2.0, 0.8, 0.4], dtype=torch.float32)
params_bounds_ou = [
    (0.1, 20.0),    # pyr2_3 mu
    (0.001, 5.0),    # pyr2_3 theta
    (0.001, 1.0),    # pyr2_3 sigma
    (0.1, 20.0),    # pyr5 mu
    (0.001, 5.0),    # pyr5 theta
    (0.001, 1.0),    # pyr5 sigma
    (0.1, 20.0),    # st mu
    (0.001, 5.0),    # st theta
    (0.001, 1.0),    # st sigma
    (0.1, 20.0),    # PV mu
    (0.001, 5.0),    # PV theta
    (0.001, 1.0),    # PV sigma
    (0.0, 0.1),    # SST mu
    (0.001, 2.0),    # SST theta
    (0.0, 0.1),    # SST sigma
    (0.0, 5.0),    # VIP mu
    (0.001, 2.0),    # VIP theta
    (0.001, 1.0)    # VIP sigma
]
