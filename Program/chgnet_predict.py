from chgnet.model.model import CHGNet
from pymatgen.core import Structure
import numpy as np
from ase.io import read, write
import sys
from chgnet.model import StructOptimizer,CHGNetCalculator
from ase import Atoms

infile = sys.argv[1]

in_atom = read(infile, ":")

out_atom = []

for atom in in_atom:
    # nep_calc = CHGNetCalculator()
    # atom.calc = nep_calc

    position = atom.get_positions()
    cell = atom.get_cell()
    pbc = atom.get_pbc()
    symbols = atom.get_chemical_symbols()

    oatom = Atoms(symbols,
                     positions=position,
                     cell=cell,
                     pbc=pbc)

    chgnet = CHGNet.load()
    structure = Structure.from_ase_atoms(atom)
    prediction = chgnet.predict_structure(structure)

    pre_energy = prediction['e']    # eV/atom
    pre_forces = prediction['f']    # eV/A
    pre_stress = prediction['s']    # GPa
    pre_magmom = prediction['m']    # mu_B

    oatom.set_array('magmoms', pre_magmom)
    oatom.set_array('forces', pre_forces)
    oatom.info['energy'] = pre_energy
    oatom.info['stress'] = pre_stress

    out_atom.append(oatom)

write(f'pre_CHGNet_{infile}', out_atom)

exit()

structure = Structure.from_file(infile)
prediction = chgnet.predict_structure(structure)

for key, unit in [("energy", "eV/atom"),
    ("forces", "eV/A"),
    ("stress", "GPa"),
    ("magmom", "mu_B")]:
    print(f"CHGNet-predicted {key} ({unit}):\n{prediction[key[0]]}\n")
print(np.sum(prediction["m"]))

