import numpy as np
import os
import matplotlib.pyplot as plt

#search for already finished cases.
Re = 6.0
M = -6.75
modify = 0e0 #pure dipole
fig,ax=plt.subplots()
markers = ['x','+','1','2'] #,'1','2','3','4']
#labels = ['64x32','64x64','128x32','128x64','256x32','256x64']
labels=['64x64','128x64','256x64','512x64']
N=16

for ii in range(3*N+1):
    kk = ii
    Mstr="{:.2f}".format(M)
    myM = 10e0**M
    Restr="{:.2f}".format(Re)
    modstr="{:.2f}".format(modify)
    kstr="{:d}".format(kk)
    data_dir = "ModelS_Fit_Re" + Restr + "_M" + Mstr + "_mod"+modstr+"_k"+kstr
    exists  = os.path.isdir(data_dir)
    if (exists):
        slepceigs = np.load(data_dir+'/eigenvalues.npy')
        if ii==0:
            ax.scatter(slepceigs.real,slepceigs.imag,label=labels[0],marker=markers[0],color='black')
        elif ii==1:
            ax.scatter(slepceigs.real,slepceigs.imag,label=labels[1],marker=markers[1],color='black')
            color = iter(plt.cm.rainbow(np.linspace(0, 1, N)))
        elif ((ii>1) and (ii<17)):
            ax.scatter(slepceigs.real,slepceigs.imag,marker=markers[1],color=next(color))
        elif ii==17:
            ax.scatter(slepceigs.real,slepceigs.imag,label=labels[2],marker=markers[2],color='black')
            color = iter(plt.cm.rainbow(np.linspace(0, 1, N)))
        elif ((ii>17) and (ii<33)):
            ax.scatter(slepceigs.real,slepceigs.imag,marker=markers[2],color=next(color))
        elif ii==33:
            ax.scatter(slepceigs.real,slepceigs.imag,label=labels[3],marker=markers[3],color='black')
            color = iter(plt.cm.rainbow(np.linspace(0, 1, N)))
        else:
            ax.scatter(slepceigs.real,slepceigs.imag,marker=markers[3],color=next(color))

ax.set_xlabel('Real')
ax.set_ylabel('Imag')
ax.legend()
fig.savefig('Eigs_compare.png', dpi=600)

