from tblite.ase import TBLite
import os.path
import time
import numpy as np
from ase.io import read
from ase import Atoms
from mace.calculators import MACECalculator
import argparse
import torch

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--start',)
parser.add_argument('-e', '--end',)
parser.add_argument('-d', '--device',)

args = parser.parse_args()

start = int(args.start)
end = int(args.end)
device = args.device

torch.cuda.set_device(int(device.split(":")[-1]))

#print(os.getcwd() + "/{}-{}/0/".format(start, end))

fname = 'rdy_namd'
a=True
calculator = MACECalculator(model_path='/home/lia/rat/MACE_models/PBE0_TZVP_run-123_stagetwo.model',device=device, enable_cueq=True)
#calculator = MACECalculator(model_path='/home/lia/rat/MACE_B2PLYP/maceDFT01_run-123_stagetwo.model', device=device)
#calculator2 = TBLite(method="GFN2-xTB", charge=1)
factor = 23.0621 #eV to kcal/mol

#elements = ['C','C','C','C','H','H','H','H','H','H','H','H','H','O','H','H']
atoms = Atoms('CCCCHHHHHHHHHOHH')

while a:
    for i in range(start,end):
        filepath = os.getcwd() + "/{}/0/".format(i)
        if os.path.isfile(filepath + fname):
#            print("calc in {}".format(filepath))
            coords = []
            with open(filepath + "qmmm_0.input") as f:
                f.readline()
                for line in f:
                    coords.append(line.split())
            atoms.set_positions(coords)
            atoms.calc = calculator
        # NAMD needs energies in kcal/mol
            energy = np.array([atoms.get_total_energy()]).astype(np.float32) * factor
        # NAMD needs *FORCES* in kcal/mol/angstrons
        # The conversion factor is -1*627.509469/0.529177 = -1185.82151
            forces = np.hstack((atoms.get_forces(),np.zeros(atoms.get_forces().shape[0]).reshape(16,1))) * factor
            with open(filepath + 'qmmm_0.input.result', "wb") as f:
                np.savetxt(f, energy, fmt='%f')
                np.savetxt(f, forces, delimiter=' ', fmt='%f')
            os.remove(filepath + fname)
            open(filepath + 'rdy_mace','w').close()

            if os.path.isfile(filepath + "stop"):
                quit()
        elif os.path.isfile(filepath + "stop"):
            quit()
        time.sleep(0.00005)
