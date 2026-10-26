#from pvmodel_shaded import Panel
from gym_mppt.envs.pvmodel_shaded import Panel 

class Shaded(object):
    def __init__(self):
#         self.G = [100, 1000]  # read irradiance # Solar radiation in mW / sq.cm / Esto es la irradiancia para panel cuando [sombredo,sin_sombra]
        #self.T = 25  # read temperature # ojo con kelvin 273
        #self.SH = [4, 10, 7, 10, 10, 10]  # Shaded modules
        self.Mp = 10  # Modules in parallel
        self.Ng = [40, 38, 22]  # Parallel-connected series assemblies
        self.Iscr_sh = 0.375



    def data(self, Irradiancia_maxima, T, V,SH):
        self.G = [Irradiancia_maxima//10, Irradiancia_maxima]
        # G_ar = np.array(G) # Solar radiation in mW/sq.cm

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