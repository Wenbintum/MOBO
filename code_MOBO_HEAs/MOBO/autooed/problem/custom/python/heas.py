import numpy as np
from autooed.problem.problem import Problem
from utils.surface_dev import BruteForceSurface
from utils.regression import load_GCN
from utils.bayesian import expected_improvement, append_to_file, random_comp, opt_acquisition
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from time import time
import pickle
import numpy as np
import torch
import iteround
from scipy import constants

import os
import checkpoint
CHECKPOINTPATH = checkpoint.__path__[0]+"/checkpoint.state"
#CHECKPOINTPATH = os.path.abspath("checkpoint.state")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

kB=constants.physical_constants['Boltzmann constant'][0]

atomic_radius = {
    "Au": 1.442,
    "Os": 1.352,
    "Pd": 1.375,
    "Pt": 1.387,
    "Ru": 1.338,
    "Cu": 1.278,
    "Re": 1.375,
    "Ag": 1.445,
    "Rh": 1.345,
    "Ir": 1.357,

    "Co": 1.251,
    "Fe": 1.241,
    "Ni": 1.246,
    "Mn": 1.35,
    "Cr": 1.249
}

bulk_modulus = {
    "Au": 220,
    "Os": 462,
    "Pd": 180,
    "Pt": 230,
    "Ru": 220,
    "Cu": 140,
    "Re": 370,
    "Ag": 100,
    "Rh": 380,
    "Ir": 320,

    "Co": 180,
    "Fe": 170,
    "Ni": 180,
    "Mn": 120,
    "Cr": 160
}

melting_points = {
    "Au": 1338,
    "Os": 3306,
    "Pd": 1827,
    "Pt": 1045,
    "Ru": 2527,
    "Cu": 1358,
    "Re": 3459,
    "Ag": 1235,
    "Rh": 2236,
    "Ir": 2719,

    "Co": 1768,
    "Fe": 1811,
    "Ni": 1728,
    "Mn": 1519,
    "Cr": 2180
}

molar_volume = {

    "Co": 6.62,
    "Fe": 7.0923,
    "Ni": 6.5888,
    "Mn": 7.3545,
    "Cr": 7.2317

}

mixing_enthalpy = {
 ('Au', 'Os') : 18,
 ('Au', 'Pd') : 0,
 ('Au', 'Pt') : 4,
 ('Au', 'Ru') : 15,
 ('Au', 'Cu') : -9,
 ('Au', 'Re') : 20,
 ('Au', 'Ag') : -6,
 ('Au', 'Rh') : 7,
 ('Au', 'Ir') : 13,
 ('Os', 'Pd') : 8,
 ('Os', 'Pt') : 0,
 ('Os', 'Ru') : 0,
 ('Os', 'Cu') : 10,
 ('Os', 'Re') : -1,
 ('Os', 'Ag') : 28,
 ('Os', 'Rh') : 2,
 ('Os', 'Ir') : -1,
 ('Pd', 'Pt') : 2,
 ('Pd', 'Ru') : 6,
 ('Pd', 'Cu') : -14,
 ('Pd', 'Re') : 6,
 ('Pd', 'Ag') : -7,
 ('Pd', 'Rh') : 2,
 ('Pd', 'Ir') : 6,
 ('Pt', 'Ru') : -1,
 ('Pt', 'Cu') : -12,
 ('Pt', 'Re') : -4,
 ('Pt', 'Ag') : -1,
 ('Pt', 'Rh') : -2,
 ('Pt', 'Ir') : 0,
 ('Ru', 'Cu') : 7,
 ('Ru', 'Re') : -1, 
 ('Ru', 'Ag') : 23,
 ('Ru', 'Rh') : 1,
 ('Ru', 'Ir') : -1,
 ('Cu', 'Re') : 18,
 ('Cu', 'Ag') : 2,
 ('Cu', 'Rh') : -2,
 ('Cu', 'Ir') : 0,
 ('Re', 'Ag') : 38,
 ('Re', 'Rh') : 1,
 ('Re', 'Ir') : -3,
 ('Ag', 'Rh') : 10,
 ('Ag', 'Ir') : 16,
 ('Rh', 'Ir') : 1,
 ('Re', 'Fe') : 0,
 ('Re', 'Co') : 2,
 ('Ru', 'Fe') : -5,
 ('Ru', 'Co') : -1,
 ('Fe', 'Co') : -1,
 ('Fe', 'Ni') : -2,
 ('Fe', 'Mn') : 0,
 ('Co', 'Ni') : 0,
 ('Co', 'Mn') : -5,
 ('Mn', 'Ni') : -8,
}


def sE(x): #no kB for now
    return 1+x/2-np.log(x)+np.log(1-np.exp(-x))-x/2*(1+np.exp(-x))/(1-np.exp(-x))

class hea:

    def __init__(self,elems,comps=None):
        if comps is None:
            comps=np.ones(len(elems))

        non_zeros = np.where(np.array(comps)!=0)
        self.elems = np.array(elems)[non_zeros]
        self.comps = np.array(comps)[non_zeros]/np.sum(np.array(comps)[non_zeros])


        self.S_id=-np.sum(self.comps*np.log(self.comps))
        self.radii=np.array([atomic_radius[elem] for elem in self.elems])
        self.Tm=np.sum(self.comps*np.array([melting_points[elem] for elem in self.elems]))

#! bulk modulos in Pa
        self.bms=np.array([bulk_modulus[elem]*1e9 for elem in self.elems]) # in Pa

#! volumes in m^3
        self.volumes=4.0/3.0*np.pi*self.radii**3*1e-30 # in m^3
        self.delta=np.sqrt(np.sum(self.comps*(1.0-self.radii/np.sum(self.comps*self.radii))**2))

    def H_hat(self):
        H_hat = 0
        coeff = 0
        for idx_i, element_i in enumerate(self.elems):
            for idx_j, element_j in enumerate(self.elems):
                if element_j != element_i:

                    Hij = mixing_enthalpy[(element_i,element_j)] if (element_i,element_j) in mixing_enthalpy else mixing_enthalpy[(element_j,element_i)] 
                    
                    H_hat+= self.comps[idx_i]*self.comps[idx_j]*Hij
                    coeff+= self.comps[idx_i]*self.comps[idx_j]

        return H_hat/coeff
                    
    def xe(self,T):
        return 4.12*self.delta*np.sqrt(np.sum(self.comps*self.bms)*np.sum(self.comps*self.volumes)/(kB*T))
    
#! enthalpy in kj/mol. 
    def xc(self, T):

        h_hat = self.H_hat()

        _xc = 0

        for idx_i, element_i in enumerate(self.elems):
            for idx_j, element_j in enumerate(self.elems):
                if element_j != element_i:

                    Hij = mixing_enthalpy[(element_i,element_j)] if (element_i,element_j) in mixing_enthalpy else mixing_enthalpy[(element_j,element_i)] 

                    _xc += self.comps[idx_i]*self.comps[idx_j]*(Hij - h_hat)**2
            
        return 2*np.sqrt(np.sqrt(_xc)*1000/constants.N_A/(kB*T)) 
    
    def S_corr(self,T=None):
        if T is None:
            T=self.Tm
        #! for benchmark xe only
        #return self.S_id+sE(self.xe(T)) # Note: no kB included, this is properly S_corr/k_B
        #! consider both xe and xc constributions
        if self.xe(T) + self.xc(T) == 0:
            return self.S_id
        else:
            return self.S_id+sE(self.xe(T)+self.xc(T))
    
    def S_corr_Indicator(self, T=None):

        return self.S_corr(T=T)*8.314*10  #for the purpose of same magnitude
    
    def S_ideal_Indicator(self, T=None):

        return self.S_id*8.314*10  #for the purpose of same magnitude

    def H_mix_Total(self):
        
        H_mix = 0

        for idx_i, element_i in enumerate(self.elems):
            for idx_j, element_j in enumerate(self.elems):
                if element_j != element_i:

                    Hij = mixing_enthalpy[(element_i,element_j)] if (element_i,element_j) in mixing_enthalpy else mixing_enthalpy[(element_j,element_i)]
                    H_mix+=4*Hij*self.comps[idx_i]*self.comps[idx_j]

        return H_mix


def comp2act(temp, E_opt, eU, pt_act,
        adsorbates, ads_atoms, sites, coordinates, height,
        regressor, feat_type,n_neighrbos, facet, size, displace_e, scale_e):
    comp = temp
    surf_size = size

    # Construct surface
    surface = BruteForceSurface(comp, adsorbates, ads_atoms, sites, coordinates, height,
                                    regressor, 'graphs', 2, 'fcc111', surf_size, displace_e, scale_e)

    # Determine gross energies of surface sites
    surface.get_gross_energies()

    # Get net adsorption energy distributions upon filling the surface
    surface.get_net_energies()

    # Get activity of the net distribution of *OH adsorption energies
    activity = surface.get_activity(G_opt=E_opt, eU=eU, T=298.15, j_d=1)
    
    # Print sampled composition
    f_str = ' '.join(f"{k}({v + 1e-5:.2f})" for k, v in comp.items())
    print(f_str)
    

    return activity/pt_act * 100

    #"f{f_str A = {activity / pt_act * 100:.0f}} %'

def clip_negative(X, xl, xu):
    X = np.clip(X, xl, xu)
    X = np.array(X)/np.sum(X)
    X = iteround.saferound(X, 2)
    return X

def cal_price_xD(comps, elements):
    element_price = { 
      "Cu": 0.239,
      "Ru": 473.000,
      "Rh": 12350.000,
      "Pd": 1719.000,
      "Ag": 23.964,
      "Re": 43.148,
      "Os": 55509.442,
      "Ir": 4790.000,
      "Pt": 1069.000,
      "Au": 1824.151}
    element_dict = dict(zip(elements, comps))
    return sum([element_price[_element]*element_dict[_element] for _element in elements])/element_price["Pt"]*100

def temp_comps(compositions, elements):
    all_elements = ['Ag', 'Ir', 'Pd', 'Pt', 'Ru', 'Au', 'Os', 'Cu', 'Re', 'Rh']
    all_dict = dict.fromkeys(all_elements, 0)
    sub_dict = dict(zip(elements, compositions))
    for key_i in sub_dict:
        if key_i in all_dict:
            all_dict[key_i] = sub_dict[key_i]
    return all_dict

def cal_activity_xD(compositions, elements, surf_size=100):
    temp = temp_comps(compositions, elements)

    ads_atoms = ['O','H']
    adsorbates = ['OH','O']
    sites = ['ontop','fcc']
    coordinates = [([0,0,0],[0.65,0.65,0.40]), None]
    height = np.array([2,1.3])
    displace_e = [0.0, 0.0]
    scale_e = [1, 0.5]

    with open(CHECKPOINTPATH, "rb") as input:
        trained_state = pickle.load(input)

    #load optimized hyperparameters
    kwargs = {'n_conv_layers': 4,    # number of gated graph convolution layers
              'n_hidden_layers': 0,  # number of hidden layers
              'conv_dim': 22,  
              'act': 'relu'          # activation function in hidden layers.
             }

    regressor = load_GCN(kwargs,trained_state=trained_state).to(DEVICE)
    # Define slab size (no. of atom x no. of atoms)
    surf_size = (surf_size, surf_size)
    # Define optimal OH adsorption energy (relative to Pt(111))
    E_opt = 0.100  # eV
    # Define potential at which to evaluate the activity
    eU = 0.820  # eV
    j_ki = np.exp(-(np.abs(-E_opt) - 0.86 + eU) / (8.617e-5 * 298.15))
    pt_act = 2 / 3 * np.sum(1 / (1 + 1 / j_ki))
    activity = comp2act(temp, E_opt=E_opt, eU = eU, pt_act = pt_act,
                       adsorbates=adsorbates, ads_atoms=ads_atoms, sites=sites,
                       coordinates=coordinates, height=height, regressor=regressor, feat_type="graph",
                       n_neighrbos=2, facet="fcc111", size=surf_size, displace_e=displace_e,
                       scale_e=scale_e)

    return activity



#define base class for tasks
class HEAS_base_3OB(Problem):
    config = {}
    elements = []
    def evaluate_objective(self, x):

        elements = self.elements
        xi = 1 - sum(x)
        X = np.append(x, xi)
        comps = iteround.saferound(X, 2)

        #ensure valid samples
        tol = 0.05
        if np.sum(comps[:self.config["n_var"]]) < 0 - 1*(tol) or \
           np.sum(comps[:self.config["n_var"]]) > 1 + 1*(tol):
            raise ValueError("Suggested points are not valid")
        if np.any(np.array(comps) < - 1e-4):
            comps = clip_negative(comps, self.config["var_lb"], self.config["var_ub"])
            print(f"Clip negative comps: {comps}")

        if self.config["n_var"] == 4:
            surf_size = 100
        elif self.config["n_var"] == 5:
            surf_size = 150
        elif self.config["n_var"] == 6:
            surf_size = 200
        elif self.config["n_var"] == 9:
            surf_size = 250
        else:
            raise ValueError("Unknown nums. of elements")

        f1 = -cal_activity_xD(comps, elements, surf_size=surf_size)
        f2 = -hea(elements,comps).S_ideal_Indicator()
        f3 = cal_price_xD(comps, elements)

        print("Activity:", f"{-f1 + 1e-5:.0f}" + "%", 
              "Cost:",     f"{f3  + 1e-5:.0f}" + "%",
              "Entropy:",  f"{-f2/10  + 1e-5:.1f}"
              )

        return f1, f2, f3

    def evaluate_constraint(self, x):

        xi = 1 - sum(x)
        g1 = -xi
        g2 = xi - 0.9

        return g1, g2



class HEAS_base_ideal_entropy(Problem):
    config = {}
    elements = []

    def evaluate_objective(self, x):

        elements = self.elements
        xi = 1 - sum(x)
        X = np.append(x, xi)
        comps = iteround.saferound(X, 2)

        #ensure valid samples
        tol = 0.05
        if np.sum(comps[:self.config["n_var"]]) < 0 - 1*(tol) or \
           np.sum(comps[:self.config["n_var"]]) > 1 + 1*(tol):
            raise ValueError("Suggested points are not valid")
        if np.any(np.array(comps) < - 1e-4):
            comps = clip_negative(comps, self.config["var_lb"], self.config["var_ub"])
            print(f"Clip negative comps: {comps}")

        if self.config["n_var"] == 4:
            surf_size = 100
        elif self.config["n_var"] == 5:
            surf_size = 150
        elif self.config["n_var"] == 6:
            surf_size = 200
        elif self.config["n_var"] == 9:
            surf_size = 250
        else:
            raise ValueError("Unknown nums. of elements")

        f1 = -cal_activity_xD(comps, elements, surf_size=surf_size)
        f2 = -hea(elements,comps).S_ideal_Indicator()

        print("Activity:", f"{-f1 + 1e-5:.0f}" + "%", 
              "Entropy:",  f"{-f2/10  + 1e-5:.1f}"
              )

        return f1, f2

    def evaluate_constraint(self, x):

        xi = 1 - sum(x)
        g1 = -xi
        g2 = xi - 0.9

        return g1, g2



class HEAS_base(Problem):
    config = {}
    elements = []

    def evaluate_objective(self, x):

        elements = self.elements
        xi = 1 - sum(x)
        X = np.append(x, xi)
        comps = iteround.saferound(X, 2)

        #ensure valid samples
        tol = 0.05
        if np.sum(comps[:self.config["n_var"]]) < 0 - 1*(tol) or \
           np.sum(comps[:self.config["n_var"]]) > 1 + 1*(tol):
            raise ValueError("Suggested points are not valid")
        if np.any(np.array(comps) < - 1e-4):
            comps = clip_negative(comps, self.config["var_lb"], self.config["var_ub"])
            print(f"Clip negative comps: {comps}")

        if self.config["n_var"] == 4:
            surf_size = 100
        elif self.config["n_var"] == 5:
            surf_size = 150
        elif self.config["n_var"] == 6:
            surf_size = 200
        elif self.config["n_var"] == 9:
            surf_size = 250
        else:
            raise ValueError("Unknown nums. of elements")

        f1 = -cal_activity_xD(comps, elements, surf_size=surf_size)
        f2 = cal_price_xD(comps, elements)

        print("Activity:", f"{-f1 + 1e-5:.0f}" + "%", 
              "Cost:",     f"{f2  + 1e-5:.0f}" + "%")

        return f1, f2

    def evaluate_constraint(self, x):

        xi = 1 - sum(x)
        g1 = -xi
        g2 = xi - 0.9

        return g1, g2



#example 1:
#bio-objective optimization on cost and activity for AgIrPdPtRu
class HEAS_AgIrPdPtRu(HEAS_base):

      config = {
        'type': 'continuous',
        'n_var': 4, 
        'n_obj': 2, 
        'n_constr' : 2,
        'var_lb': 0,
        'var_ub': 0.9,
               }
      elements = ['Ag', 'Ir', 'Pd', 'Pt', 'Ru']  

      def __init__(self):
          super().__init__()
          
#example 2:
#bio-objective optimization on ideal mixing entropy and activity for AgAuCuIrOsPdPtReRhRu
class HEAS_AgAuCuIrOsPdPtReRhRu_ideal_entropy(HEAS_base_ideal_entropy):
      config = {
        'type': 'continuous',
        'n_var': 9,
        'n_obj': 2,
        'n_constr' : 2,
        'var_lb': 0,
        'var_ub': 0.9,
               }
      elements = ['Ag', 'Au', 'Cu', 'Ir', 'Os', 'Pd', 'Pt', 'Re', 'Rh', 'Ru']

      def __init__(self):
          super().__init__()


#example 3:
#tri-objective optimization on cost, ideal mixing entropy and activity for AgIrPdPtRu
class HEAS_AgIrPdPtRu_3OB(HEAS_base_3OB):

      config = {
        'type': 'continuous',
        'n_var': 4,
        'n_obj': 3,
        'n_constr' : 2,
        'var_lb': 0,
        'var_ub': 0.9,
               }
      elements = ['Ag', 'Ir', 'Pd', 'Pt', 'Ru']

      def __init__(self):
          super().__init__()
