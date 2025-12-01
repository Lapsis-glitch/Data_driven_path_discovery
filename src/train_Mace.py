#from xtb.ase.calculator import XTB#This needs to be imported first, otherwise errors occur due to various incompatibilites, see https://github.com/tblite/tblite/issues/110
from tblite.ase import TBLite
import warnings
warnings.filterwarnings("ignore")
import os.path
import os
os.chdir('/home/rat/mace')
import time
import numpy as np
from mace.calculators import mace_anicc, mace_off
from ase import build
from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from tqdm import tqdm
# from xtb.ase.calculator import XTB
from aseMolec import extAtoms as ea
from mace.cli.run_train import main as mace_run_train_main
import sys
import logging
from mace.cli.eval_configs import main as mace_eval_configs_main
from aseMolec import pltProps as pp
from ase.io import read
import matplotlib.pyplot as plt
from aseMolec import extAtoms as ea
import numpy as np
from IPython.utils import io

from ase.io import read, write
from tqdm import tqdm
from tblite.ase import TBLite
from IPython.utils import io
import warnings

warnings.filterwarnings("ignore")
from mace.cli.run_train import main as mace_run_train_main
from mace.cli.eval_configs import main as mace_eval_configs_main
import sys
import logging
import os


from aseMolec import pltProps as pp
from ase.io import read
import matplotlib.pyplot as plt
from aseMolec import extAtoms as ea
import numpy as np


def train_mace(config_file_path):
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", config_file_path]
    mace_run_train_main()


def eval_mace(configs, model, output):
    sys.argv = ["program", "--configs", configs, "--model", model, "--output", output, "--device", "cuda"]
    mace_eval_configs_main()

def plot_RMSEs(db, labs, plotpath):
    ea.rename_prop_tag(db, 'MACE_energy', 'energy_mace') #Backward compatibility
    ea.rename_prop_tag(db, 'MACE_forces', 'forces_mace') #Backward compatibility

    plt.figure(figsize=(9,6), dpi=100)
    plt.subplot(1,3,1)
    pp.plot_prop(ea.get_prop(db, 'bind', '_dft', True).flatten(), \
                 ea.get_prop(db, 'bind', '_mace', True).flatten(), \
                 title=r'Energy $(\rm eV/atom)$ ', labs=labs, rel=False)
    plt.subplot(1,3,2)
    pp.plot_prop(ea.get_prop(db, 'info', 'energy_dft', True).flatten(), \
                 ea.get_prop(db, 'info', 'energy_mace', True).flatten(), \
                 title=r'Energy $(\rm eV/atom)$ ', labs=labs, rel=False)
    plt.subplot(1,3,3)
    pp.plot_prop(np.concatenate(ea.get_prop(db, 'arrays', 'forces_dft')).flatten(), \
                 np.concatenate(ea.get_prop(db, 'arrays', 'forces_mace')).flatten(), \
                 title=r'Forces $\rm (eV/\AA)$ ', labs=labs, rel=False)
    plt.tight_layout()
    plt.savefig(plotpath)
    return


# Split the training data into train + test
db = read('data_/0.xyz', index=':')
for i in range(32):
    db = db + read('data/{}.xyz'.format(i), index=':')
db = read('data/C.xyz', index=':') + read('data/H.xyz', index=':') + read('data/O.xyz', index=':') + db
for at in db[:3]:
    at.info['config_type'] = 'IsolatedAtom'
print("Number of configs in database: ", len(db))
np.random.seed(42)
test = np.random.randint(3, len(db), int(len(db) * 0.1), dtype=int)
train = [x for i, x in enumerate(np.arange(0, len(db), 1, dtype=int)) if i not in test]
xtb_calc = TBLite(method="GFN2-xTB", charge=1, max_iterations=300)

for at in tqdm(db[3:]):  # showcase: first 15 frames
    at.calc = xtb_calc
    with io.capture_output() as captured:
        at.info['energy_xtb'] = at.get_potential_energy()
        at.arrays['forces_xtb'] = at.get_forces()

write('data/xtb_train.xyz', [db[i] for i in train[3:]])
write('data/xtb_test.xyz', [db[i] for i in test])

s_at = read('data/C.xyz', index=':')
s_at = s_at + read('data/O.xyz', index=':')
s_at = s_at + read('data/H.xyz', index=':')
xtb_calc = TBLite(method="GFN2-xTB")
for at in tqdm(s_at):
    at.info['config_type'] = 'IsolatedAtom'
    at.calc = xtb_calc
    with io.capture_output() as captured:
        at.info['energy_xtb'] = at.get_potential_energy()
        at.arrays['forces_xtb'] = at.get_forces()
write('data/xtb_s_at.xyz', s_at)



train_mace("/home/rat/Nancy_D/SNAR_v3/MACE/config-02.yml")


## evaluate the training set
eval_mace(configs="/home/rat/Nancy_D/SNAR_v3/MACE//dft_train.xyz",
          model="/home/rat/Nancy_D/SNAR_v3/MACE/MACE_models/mace_B3LYP_SNAR_it1_stagetwo_compiled.model",
          output="/home/rat/Nancy_D/SNAR_v3/MACE//solvent_train.xyz")

#evaluate the test set
eval_mace(configs="/home/rat/Nancy_D/SNAR_v3/MACE//dft_test.xyz",
          model="/home/rat/Nancy_D/SNAR_v3/MACE/MACE_models/mace_B3LYP_SNAR_it1_stagetwo_compiled.model",
          output="/home/rat/Nancy_D/SNAR_v3/MACE//solvent_test.xyz")





train_data = read('/home/rat/Nancy_D/SNAR_v3/MACE//solvent_train.xyz', ':')
test_data = train_data[:3]+read('/home/rat/Nancy_D/SNAR_v3/MACE/solvent_test.xyz', ':') #append the E0s for computing atomization energy errors

plot_RMSEs(train_data, labs=['DFT-B2PLYP', 'MACE'], plotpath = '/home/rat/Nancy_D/SNAR_v3/MACE/plot_B2PLYP_train.png')
plot_RMSEs(test_data, labs=['DFT-B2PLYP', 'MACE'], plotpath = '/home/rat/Nancy_D/SNAR_v3/MACE/plot_B2PLYP_test.png')