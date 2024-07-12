import matplotlib
import matplotlib.pyplot as plt
import os
import re
from typing import List, Dict, Union, Any
import numpy as np


if __name__ == "__main__":


    nstr = "highspin"
    nstr = "lowspin"

    from ase.io import read, write

    vasp_xyz = read(f"{nstr}.xyz", ":")
    chgnet_xyz = read(f"pre_CHGNet_{nstr}.xyz", ":")

    total_vene, total_cene = [], []

    for va, ca in zip(vasp_xyz, chgnet_xyz):

        vene = va.get_total_energy()
        vfor = va.get_forces()
        vmag = va.get_magnetic_moment()
        # vstr = va.get_stress()
        # vstr = va.get_stresses()

        cene = ca.get_total_energy()
        cfor = ca.get_forces()
        cmag = ca.get_magnetic_moment()
        # cstr = ca.get_stress()

        total_vene.append(vene)
        total_cene.append(cene)

        try:
            total_vfor = np.concatenate((total_vfor, vfor), axis=0)
            total_cfor = np.concatenate((total_cfor, cfor), axis=0)
            total_vmag = np.concatenate((total_vmag, vmag), axis=0)
            total_cmag = np.concatenate((total_cmag, cmag), axis=0)
        except:
            total_vfor = vfor
            total_cfor = cfor
            total_vmag = vmag
            total_cmag = cmag


    total_vene = np.array(total_vene)
    total_cene = np.array(total_cene)

    np.savetxt(f"energy_{nstr}.out", np.column_stack([total_cene,total_vene]))
    np.savetxt(f"mament_{nstr}.out", np.column_stack([total_cmag,total_vmag]))
    np.savetxt(f"force_{nstr}.out", np.concatenate((total_cfor,total_vfor), axis=1))


    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)

    data = np.loadtxt(f"energy_{nstr}.out")
    nclo = int(data.shape[1]/2)
    targe = data[:, :nclo].reshape(-1)
    predi = data[:, nclo:].reshape(-1)
    pids = np.abs(targe - predi) < 1000
    targe = targe[pids]
    predi = predi[pids]
    units = ['energy', 'eV', 1]

    data_min = np.min([np.min(targe),np.min(predi)])
    data_max = np.max([np.max(targe),np.max(predi)])
    data_min -= (data_max-data_min)*0.1
    data_max += (data_max-data_min)*0.1
    RMSE = np.sqrt(((predi - targe) ** 2).mean())
    plt.plot([data_min, data_max], [data_min, data_max], c="grey", lw=2,
             label=f"RMSE:{1000*RMSE:.4f} {units[1]}")
    plt.xlim([data_min, data_max])
    plt.ylim([data_min, data_max])
    color = f"C{units[2]}"
    plt.plot(targe, predi, '.', color=color, ms=5)
    plt.xlabel(f'DFT {units[0]} ({units[1]})')
    plt.ylabel(f'CHGNet {units[0]} ({units[1]})')
    print(" >>", f'{1000*RMSE:.4f}', units[1])

    plt.subplot(1,3,2)

    data = np.loadtxt(f"force_{nstr}.out")
    nclo = int(data.shape[1]/2)
    targe = data[:, :nclo].reshape(-1)
    predi = data[:, nclo:].reshape(-1)
    pids = np.abs(targe - predi) < 1000
    targe = targe[pids]
    predi = predi[pids]
    units = ['force', 'eV/A', 2]

    data_min = np.min([np.min(targe),np.min(predi)])
    data_max = np.max([np.max(targe),np.max(predi)])
    data_min -= (data_max-data_min)*0.1
    data_max += (data_max-data_min)*0.1
    RMSE = np.sqrt(((predi - targe) ** 2).mean())
    plt.plot([data_min, data_max], [data_min, data_max], c="grey", lw=2,
             label=f"RMSE:{1000*RMSE:.4f} {units[1]}")
    plt.xlim([data_min, data_max])
    plt.ylim([data_min, data_max])
    color = f"C{units[2]}"
    plt.plot(targe, predi, '.', color=color, ms=5)
    plt.xlabel(f'DFT {units[0]} ({units[1]})')
    plt.ylabel(f'CHGNet {units[0]} ({units[1]})')
    print(" >>", f'{1000*RMSE:.4f}', units[1])

    plt.subplot(1,3,3)

    data = np.loadtxt(f"mament_{nstr}.out")
    nclo = int(data.shape[1]/2)
    targe = data[:, :nclo].reshape(-1)
    predi = data[:, nclo:].reshape(-1)
    pids = np.abs(targe - predi) < 1000
    targe = targe[pids]
    predi = predi[pids]
    units = ['mament', 'mu_B', 3]

    data_min = np.min([np.min(targe),np.min(predi)])
    data_max = np.max([np.max(targe),np.max(predi)])
    data_min -= (data_max-data_min)*0.1
    data_max += (data_max-data_min)*0.1
    RMSE = np.sqrt(((predi - targe) ** 2).mean())
    plt.plot([data_min, data_max], [data_min, data_max], c="grey", lw=2,
             label=f"RMSE:{1000*RMSE:.4f} {units[1]}")
    plt.xlim([data_min, data_max])
    plt.ylim([data_min, data_max])
    color = f"C{units[2]}"
    plt.plot(targe, predi, '.', color=color, ms=5)
    plt.xlabel(f'DFT {units[0]} ({units[1]})')
    plt.ylabel(f'CHGNet {units[0]} ({units[1]})')
    print(" >>", f'{1000*RMSE:.4f}', units[1])

    plt.tight_layout()
    plt.savefig("compare.png", dpi=300)
