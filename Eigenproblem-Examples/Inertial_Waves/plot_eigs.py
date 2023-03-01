import numpy as np
import os
import matplotlib.pyplot as plt

Ek = -4e0
fig,ax=plt.subplots()
markers = ['x','+','1','2','3','4']
labels = ['32x32','64x64','128x128','256x128','512x64']
for ii in range(5):
    Ekstr="{:.2f}".format(Ek)
    istr = "{:d}".format(ii)
    data_dir = "Inertial_Waves_Ek" + Ekstr + '_'+istr
    exists  = os.path.isdir(data_dir)
    if (exists):
        slepceigs = np.load(data_dir+'/eigenvalues.npy')
        ax.scatter(slepceigs.real,slepceigs.imag,label=labels[ii],marker=markers[ii],alpha=0.6,s=(ii+1)*50)

ax.set_xlabel('Real')
ax.set_ylabel('Imag')
ax.legend()
fig.savefig('Eigs_compare.png', dpi=600)

