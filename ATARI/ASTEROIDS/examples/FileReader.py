import numpy as np

def readSammyPar(file):
    with open(file, 'r') as contents:
        txt = contents.read()
        # header = txt.split('RESONANCES')[0]
        resonances = np.array([[float(col) for col in res_txt.split()] for res_txt in txt.split('RESONANCES')[1].split('\n')[1:]])

    # Only take positive resonance energies:
    resonances = resonances[resonances[:,0] > 0.0,:]

    E      = resonances[:,0]
    Gg     = resonances[:,1]
    Gn     = resonances[:,2]
    SGType = np.int_(resonances[:,-1]) - 1

    return E, Gn, Gg, SGType

