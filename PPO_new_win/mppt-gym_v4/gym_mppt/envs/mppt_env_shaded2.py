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
#         I = 0

#         Vt = self.k * Tcell * self.A / self.q 
#         diff = float('inf')
#         while diff > 10e-4:
#             fv = I - (Iph*self.Np - Irs*self.Np*(np.exp(((vx/(self.Ns*SH)) + ((I*self.Rs)/self.Np))/Vt) - 1) - ((((vx*self.Np)/self.Ns) + (I*self.Rs))/self.Rsh))
# #             fv = I - (Iph*self.Np - Irs*self.Np*(np.exp(((vx/(self.Ns*SH)) + ((I*self.Rs)/self.Np))/Vt) - 1))
#             fdv = 1 + (self.Rs/self.Rsh) + ((np.exp(((vx/(self.Ns*SH)) + ((I*self.Rs)/self.Np))/Vt)*Irs*self.Rs)/(Vt))
# #             fdv = 1                      + ((np.exp(((vx/(self.Ns*SH)) + ((I*self.Rs)/self.Np))/Vt)*Irs*self.Rs)/(Vt))
#             # In = I - (fv./fdv)
#             In = I - fv/fdv
#             diff = abs(abs(In) - abs(I))
#             I = In
        
        # panel output voltage
        V = vx # este es el Vg?
        # panel power
        P = vx * I

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


class MpptEnvShaded_2(gym.Env):
    metadata = {}

    def __init__(self):
        # self._Pmaxnpy = np.load('Pmax.npy')
        self._Pmaxnpy = np.load(r"C:\Users\archi\Python Code\BTP\PPO_new\mppt-gym_v4\gym_mppt\envs\Pmax.npy")
#         self._Vocnpy = np.load("/home/archi/Code2/PPO-LSTM/mppt-gym_v4/gym_mppt/envs/Voc.npy")
        self._Vmaxnpy = np.load(r"C:\Users\archi\Python Code\BTP\PPO_new\mppt-gym_v4\gym_mppt\envs\Vmax.npy")
        self._Choice = np.load(r"C:\Users\archi\Python Code\BTP\PPO_new\mppt-gym_v4\gym_mppt\envs\choice.npy", allow_pickle=True)

        self._Max_steps = 1000
        self.action_space = spaces.Box(low = -5.0, high = 5.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low = np.array([0.0, 0.0, -1 * np.inf]), high = np.array([212.0, np.inf, np.inf]), shape=(3,), dtype=np.float32)

    def _get_obs(self):
        return self.current_state
    
    def _get_info(self):
        return {"Steps" : self._Steps, "Current" : self._Current, "Temperature" : self._Temp, "Irradiance" : self._Irr, "Shading" : self._SH}
        
    def reset(self):
        self._Voltage = 0.
        self._Power = 0.
        self._deltaPower = 0.
        self.current_state = np.array([self._Voltage, self._Power, self._deltaPower], dtype=np.float32)
        observation = self._get_obs()

        self._Steps = 0
        self._Current = 0.
        self._Temp = 25. 
        self._Irr = 1000.
#         a = random.sample([1,2,3,4,5,6,7,8,9,10],1)[0]
#         b = random.sample([1,2,3,4,5,6,7,8,9,10],1)[0]
#         c = random.sample([1,2,3,4,5,6,7,8,9,10],1)[0]
#         self._SH = [a, 10, b, 10, c, 10]
        x = random.random()
        if x < 0.2:
            self._SH = random.choice(self._Choice[0])
        elif x < 0.5:
            self._SH = random.choice(self._Choice[1])
        else:
            self._SH = random.choice(self._Choice[2])
#         self._ExploreCheckList = [0]*(self._Vocnpy[a][b][c] - 1)
        info = self._get_info()

        # return observation, info
        return observation
    
    def step(self, action):
        self._Steps += 1
        
#         self._Irr += 1
#         if self._Steps <= 250:
#             self._Temp += 0.2
#         else:
#             self._Temp -= 0.2

#         if self._Steps <= 500:
# #             self._Irr = 1000.
#             self._Temp += 0.01
#             self._Temp = round(self._Temp, 2)
#         elif self._Steps <= 1000:
# #             self._Irr = 1200.
#             self._Temp = 30.
#             self._Temp = round(self._Temp, 2)
#         elif self._Steps <= 1500:
# #             self._Irr = 800
#             self._Temp -= 0.02
#             self._Temp = round(self._Temp, 2)
#         elif self._Steps <= 2000:
# #             self._Irr = 1200.
#             self._Temp = 20.
#             self._Temp = round(self._Temp, 2)
#         elif self._Steps <= 2500:
# #             self._Irr = 1000.
#             self._Temp += 0.01
#             self._Temp = round(self._Temp, 2)

        self._Voltage += action
        self._Voltage = min(max(0.0, self._Voltage), 212.0)
        oldP = self._Power

        pv = Shaded()
        
        self._Current, self._Voltage, self._Power = pv.data(self._Irr, self._Temp, self._Voltage, self._SH)
        self._Voltage = np.round(self._Voltage, 3)
        self._Power = np.round(self._Power, 5)

        self._deltaPower = self._Power - oldP
        
        done = bool(self._Steps>=self._Max_steps)
        reward = self.reward_function(action)
        info = self._get_info()
        
        self.current_state = np.array([self._Voltage, self._Power, self._deltaPower], dtype=np.float32).reshape((3,))
        observation = self._get_obs()
        
        return observation, float(reward), done, info

    def reward_function(self, action):
        reward = 0

# #         tempVoltage = self._Voltage-0.001
#         tempVoltage = self._Voltage
#         if len(self._ExploreCheckList) > int((tempVoltage//10)):            # Exploration reward 
#             if self._ExploreCheckList[int(tempVoltage//10)] == 0:
#                 self._ExploreCheckList[int(tempVoltage//10)] = 1
#                 reward = reward + (self._Max_steps/len(self._ExploreCheckList))

        # if(math.prod(self._ExploreCheckList)) == 1:
        # if(reduce(operator.mul, self._ExploreCheckList, 1)) == 1:           # Power Reward (Only when exporation finished)
        #     reward = reward + ((self._Power/self._Pmaxnpy[self._SH[0]][self._SH[2]][self._SH[4]])**2)

        # reward = reward + ((self._Power/self._Pmaxnpy[self._SH[0]][self._SH[2]][self._SH[4]])**2)

        # if self._Steps > (self._Max_steps//5) and abs(action) > 1.0:        # For adding stability in PPO (only steps/5 available for exploration)
        #     reward = reward - 1

        if abs(self._Voltage - self._Vmaxnpy[self._SH[0]][self._SH[2]][self._SH[4]]) <= 2.5:
            reward = reward + ((self._Power/self._Pmaxnpy[self._SH[0]][self._SH[2]][self._SH[4]])**4)
        
        return np.round(reward, 5)
#         return np.round(self._Power/50000, 5)
    
    def render(self, mode):
        self._SH = mode
        return self._get_obs()

    def close(self):
        pass