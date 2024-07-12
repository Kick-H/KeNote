'''
Extract the training set from the VASP output file "vasprun.xml".
Notice:
    Here the 'stress' of VASP (unit: eV/Å^3) is converted to 'virial' (unit: GPa)
    The 'energy' used is 'free_energy', in order to correspond to the shell file.
Run example:
    $ python vasp2xyz.py ${indoc}
Contributors:
    Zezhu Zeng
    Yuwen Zhang
    Ke Xu
'''

import os, sys
import numpy as np
from ase.io import read, write
from ase import Atoms, Atom
from tqdm import tqdm


def file_to_list(file_name):
    with open(file_name, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


def local_label(file_name, label, mode=1):
    with open(file_name, 'r') as f:
        result = []
        for nline, line in enumerate(f):
            if label in line:
                if mode == 1:
                    result.append(line)
                if mode == 2:
                    result.append(nline)
    return result


def get_spin(file_name, natoms):
    spin = []
    lines = file_to_list(file_name)
    mag_line_numb = local_label(file_name, "magnetization (x)", 2)
    spin_data = lines[mag_line_numb[-1]+4:mag_line_numb[-1]+4+natoms]
    for iline, line in enumerate(spin_data):
        data = list(map(float, line.split()))[-1]
        spin.append(np.array(data))
    return np.array(spin)


def Convert_atoms(atom):
    # 1 eV/Å^3 = 160.21766 GPa
    xx,yy,zz,yz,xz,xy = -atom.calc.results['stress']*atom.get_volume() # *160.21766 
    atom.info['virial'] = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
    atom.calc.results['energy'] = atom.calc.results['free_energy']
    del atom.calc.results['stress']
    del atom.calc.results['free_energy']


def find_vasprun(start_path='.'):
    result = []
    for root, dirs, files in os.walk(start_path):
        if 'vasprun.xml' in files:
            result.append(root)
    return result


file_list = find_vasprun(start_path=sys.argv[1])

cnum = 0     # total number of configuration
atoms_list, err_list = [], []
for dir_name in tqdm(file_list):
    fxml = os.path.join(dir_name, "vasprun.xml")
    fout = os.path.join(dir_name, "OUTCAR")
    try:
        atoms = read(fxml.strip('\n'), index=":")
        spindata = get_spin(fout,atoms[0].get_global_number_of_atoms())
    except:
        err_list.append(fxml)
        continue
    for ai in range(len(atoms)):
        Convert_atoms(atoms[ai])
        try:
            atoms[ai].set_array('magmoms', spindata)
        except:
            err_list.append(fxml)
            continue
        atoms_list.append(atoms[ai])
    cnum += len(atoms)

write('train.xyz', atoms_list, format='extxyz')
print('The total number of configurations is: {} \n'.format(cnum))

if err_list:
    print("The list of failed calculation files is as follows.")
    for err_dirname in err_list:
        print(err_dirname)
else:
    print("All calculations are successful!")
