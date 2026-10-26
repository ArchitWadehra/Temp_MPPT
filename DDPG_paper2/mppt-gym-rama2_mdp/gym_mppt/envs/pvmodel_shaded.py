import numpy as np


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
        
        Vt = self.k * Tcell * self.A / self.q 
    
        steps = 0
        diff = float('inf')
        while diff > 1e-3:
            if steps > 100:
                break
            steps += 1
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

        return I,V,P