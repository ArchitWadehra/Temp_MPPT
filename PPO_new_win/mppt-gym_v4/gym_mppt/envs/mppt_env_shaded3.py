import gym
from gym import spaces
import numpy as np
import random
from functools import reduce
import operator

class Panel(object):

    def __init__(self):
        self.TK = 273 # Kelvin temperature
        self.Tr1 = 40  # Reference temperature in degree fahrenheit
        # self.S = 100  # Solar radiation in mW / sq.cm
        self.ki = 0.00023  # in A / K
        self.Iscr = 3.75  # SC Current at ref.temp. in A
        self.Irr = 0.000021  # in A
        self.k = 1.38065e-23  # Boltzmann constant
        self.q = 1.6022e-19  # charge of an electron
        self.A = 2.15 # ideality factor
        self.Eg0 = 1.166 # band gap energy
        self.alpha = 0.473
        self.beta = 636
        # panel composed of Np parallel modules each one including Ns photovoltaic cells connected
        self.Np = 1
        self.Ns = 36
        self.Rs = 1e-5
        self.Rsh = 1e5

    def calc_pv(self, G, T, vx, SH):
        # cell temperature
        Tcell = T + self.TK
        # cell reference temperature in kelvin
        Tr = ((self.Tr1 - 32) * (5 / 9)) + 273
        # band gap energy of semiconductor
        Eg = self.Eg0 - (self.alpha * Tcell * Tcell) / (Tcell + self.beta) * self.q
        # generated photocurrent
        Iph = (self.Iscr + self.ki * (Tcell - Tr)) * (G / 1000)
        # cell reverse saturation current
        Irs = self.Irr * ((Tcell / Tr) ** 3) * np.exp(self.q * Eg / (self.k * self.A) * ((1 / Tr) - (1 / Tcell)))
        # panel output current
        I = self.Np * Iph - self.Np * Irs * (np.exp(self.q / (self.k * Tcell * self.A) * vx * (1/SH) / self.Ns) - 1)
# #         I = 0

        Vt = self.k * Tcell * self.A / self.q 
        diff = float('inf')
        steps = 0
        
        while diff > 1e-3:
            steps += 1
            if steps > 100:
                break
            fv = I - (Iph*self.Np - Irs*self.Np*(np.exp(((vx/(self.Ns*SH)) + ((I*self.Rs)/self.Np))/Vt) - 1) - ((((vx*self.Np)/self.Ns) + (I*self.Rs))/self.Rsh))
#             fv = I - (Iph*self.Np - Irs*self.Np*(np.exp(((vx/(self.Ns*SH)) + ((I*self.Rs)/self.Np))/Vt) - 1))
            fdv = 1 + (self.Rs/self.Rsh) + ((np.exp(((vx/(self.Ns*SH)) + ((I*self.Rs)/self.Np))/Vt)*Irs*self.Rs)/(Vt))
#             fdv = 1                      + ((np.exp(((vx/(self.Ns*SH)) + ((I*self.Rs)/self.Np))/Vt)*Irs*self.Rs)/(Vt))
            # In = I - (fv./fdv)
            In = I - fv/fdv
            diff = abs(abs(In) - abs(I))
            I = In
        # panel output voltage
        V = vx # este es el Vg?
        # panel power
        P = vx * I
        
#         print(steps)

        return I,V,P


class Shaded(object):
    def __init__(self):
#         self.G = [100, 1000]  # read irradiance # Solar radiation in mW / sq.cm / Esto es la irradiancia para panel cuando [sombredo,sin_sombra]
        #self.T = 25  # read temperature # ojo con kelvin 273
        #self.SH = [4, 10, 7, 10, 10, 10]  # Shaded modules
        self.Mp = 10  # Modules in parallel
        self.Ng = [40, 38, 22]  # Parallel-connected series assemblies
        self.Iscr_sh = 0.375

    def data(self, Irradiancia_maxima, T, V, SH):
        self.G = [Irradiancia_maxima//10, Irradiancia_maxima]
        pv = Panel()
        Ipv = []

        for j in range(len(self.Ng)):
            IUN_i = pv.calc_pv(self.G[1], T, V, SH[j*2])[0]
            ISH_i = pv.calc_pv(self.G[0], T, V, SH[j*2+1])[0]
            Ipv.append(max(0, IUN_i, ISH_i))

        IT = Ipv[0] * self.Ng[0] + Ipv[1] * self.Ng[1] + Ipv[2] * self.Ng[2]
        VT = V
        PT = IT*VT
        return IT, VT, max(0.0, PT) 


class MpptEnvShaded_3(gym.Env):
    metadata = {}

    def __init__(self):
#         self._Pmaxnpy = np.load(r"C:\Users\archi\Python Code\BTP\PPO_new_win\mppt-gym_v4\gym_mppt\envs\Pmax.npy")
#         self._Vmaxnpy = np.load(r"C:\Users\archi\Python Code\BTP\PPO_new_win\mppt-gym_v4\gym_mppt\envs\Vmax.npy")
        self._Templst = np.load(r"C:\Users\archi\Python Code\BTP\PPO_new_win\mppt-gym_v4\gym_mppt\envs\temp.npy", allow_pickle=True)
        self._Irrlst = np.load(r"C:\Users\archi\Python Code\BTP\PPO_new_win\mppt-gym_v4\gym_mppt\envs\irr.npy", allow_pickle=True)
        
        self._Max_steps = 5000
        self.action_space = spaces.Box(low = -5.0, high = 5.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low = np.array([0.0, 0.0, 0.0, -1 * np.inf, -1 * np.inf, -1 * np.inf, -1 * np.inf, -1 * np.inf]), high = np.array([212.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]), shape=(8,), dtype=np.float32)
        
        self.V_min = 0.
        self.V_max = 210.

        self.I_min = 0.
        self.I_max = 375.

        self.P_min = 0.
        self.P_max = 53848.

        self.DeltaV_min = -5.
        self.DeltaV_max = 5.

        self.DeltaP_min = -self.P_max * 0.5
        self.DeltaP_max = self.P_max * 0.5

        self.DeltaI_min = -self.I_max
        self.DeltaI_max = self.I_max

        self.T_min = 0.
        self.T_max = 75.
        
        self.full_st_min = [self.V_min, self.I_min, self.P_min, self.DeltaV_min, self.DeltaI_min, self.DeltaP_min,
                            self.DeltaP_min, self.DeltaP_min, self.T_min]
        self.full_st_max = [self.V_max, self.I_max, self.P_max, self.DeltaV_max, self.DeltaI_max, self.DeltaP_max,
                            self.DeltaP_max, self.DeltaP_max, self.T_max]
        
        self.max_stateValue = 1.
        self.min_stateValue = -1.
        
    def _get_obs(self):
        return self.current_state
    
    def _get_info(self):
#         return {"Steps" : self._Steps, "Voltage" : self._Voltage, "Current" : self._Current, "Power" : self._Power, "NormPower" : (self._Power/self._Pmaxnpy[self._SH[0]][self._SH[2]][self._SH[4]]), "Temperature" : self._Temp, "Irradiance" : self._Irr, "Shading" : self._SH}
        return {"Steps" : self._Steps, "Voltage" : self._Voltage, "Current" : self._Current, "Power" : self._Power, "NormPower" : (self._Power/self._Pmax), "Temperature" : self._Temp, "Irradiance" : self._Irr, "Shading" : self._SH}
    
    def reset(self):
        self._Steps = 0
#         self._Temp = 25. 
#         self._Temp = random.randint(10, 50)
        self._Temp = 30.
#         self._Irr = 1000.
#         self._Irr = random.randint(100, 1000)
        self._Irr = 600
        a = random.sample([1,2,3,4,5,6,7,8,9,10],1)[0]
        b = random.sample([1,2,3,4,5,6,7,8,9,10],1)[0]
        c = random.sample([1,2,3,4,5,6,7,8,9,10],1)[0]
        self._SH = [a, 10, b, 10, c, 10]
        
#         self._Pmax, self._Vmax = self.simulation(self._Irr, self._Temp, self._SH)
        
        pv = Shaded()
        self._Voltage = 0.
        self._Current, _, _ = pv.data(self._Irr, self._Temp, self._Voltage, self._SH)
        self._Power = 0.
        self._deltaVoltage = 0.
        self._deltaCurrent = 0.
        self._deltaPower = 0.
        
#         info = self._get_info()
        
        self.current_state = np.array([float(self._Voltage), float(self._Current), float(self._Power), float(self._deltaVoltage), float(self._deltaCurrent), float(self._deltaPower), float((self._deltaPower/(self._deltaVoltage+1e-5))) , float(self._Temp)], dtype=np.float32)
        
        observation = self._get_obs()
    
        # return observation, info
        return observation
    
    def step(self, action):
#         if self._Temp != self.info["Temperature"] or self._Irr != self.info["Irradiance"] or self._Steps == 0:
#             self._Pmax, self._Vmax = self.simulation(self._Irr, self._Temp, self._SH)
#         print(self._Pmax, self._Vmax)
        if self._Steps == 0:
            self._Pmax, self._Vmax = self.simulation(self._Irr, self._Temp, self._SH)
            
        oldV = self._Voltage
        oldC = self._Current
        oldP = self._Power
        
        self._Steps += 1
        self._Voltage += action
        self._Voltage = min(max(0.0, self._Voltage), 212.0)
        
#         self._Irr = self._Irrlst[self._Steps]
#         self._Temp = self._Templst[self._Steps]

        pv = Shaded()
        
        self._Current, self._Voltage, self._Power = pv.data(self._Irr, self._Temp, self._Voltage, self._SH)
        self._Voltage = np.round(self._Voltage, 3)
        self._Power = np.round(self._Power, 5)
        
        self._deltaVoltage = self._Voltage - oldV
        self._deltaCurrent = self._Current - oldC
        self._deltaPower = self._Power - oldP
        
        done = bool(self._Steps>=self._Max_steps)
        reward = self.reward_function()
        info = self._get_info()
        
        self.current_state = np.array([float(self._Voltage), float(self._Current), float(self._Power), float(self._deltaVoltage), float(self._deltaCurrent), float(self._deltaPower), float((self._deltaPower/(self._deltaVoltage+1e-5))) , float(self._Temp)], dtype=np.float32).reshape((8,))
        
        self.states_mask = [1] * 8
        self.state_dim = np.sum(self.states_mask)
        state = []
        for i, m in enumerate(self.states_mask):
            if m == 1:
                state.append(self.current_state[i])

        state = np.reshape(
            np.hstack(state),
            (self.state_dim,))
        
        n_state = self.normalizing_state(state)
        
        return n_state, float(reward), done, info

    def reward_function(self):
#         r = - np.abs(self._Vmaxnpy[self._SH[0]][self._SH[2]][self._SH[4]] - self._Voltage) / 100
        r = - np.abs(self._Vmax - self._Voltage) / 100

#         if self._Power < self._Pmaxnpy[self._SH[0]][self._SH[2]][self._SH[4]] * 0.1:
        if self._Power < self._Pmax * 0.1:
            r = -1

        while isinstance(r, np.ndarray):
            r = r[0]
        
        return r
    
    def normalizing_state(self, state):

        self.discrete_bins = 1024
        
        st = []
        cont = 0
        for i, m in enumerate(self.states_mask):
            if m == 1:
                st.append((2 * (state[cont] - self.full_st_min[i]) / (
                        self.full_st_max[i] - self.full_st_min[i])) - 1)
                cont += 1

        st = np.array(st)
        st[st < self.min_stateValue] = self.min_stateValue
        st[st > self.max_stateValue] = self.max_stateValue

        st = (st + 1) * self.discrete_bins / 2

        st = st.astype(np.uint)

        st = st.astype(np.float32)

        st = (st * 2 / self.discrete_bins) - 1

        return st.astype(np.float32)
        
    def simulation(self, irr, temp, sh):
        pv = Shaded()
        i = 0
        maxpower = -1.
        maxvoltage = -1.
        for i in range(210):
            i += 1
            _, _, power = pv.data(irr, temp, i, sh)
            if power > maxpower:
                maxpower = power
                maxvoltage = i
        return maxpower, maxvoltage
    
    def render(self, mode):
        self._SH = mode[:6]
        self._Temp = mode[6]
        self._Irr = mode[7]
        return self._get_obs()

    def close(self):
        pass