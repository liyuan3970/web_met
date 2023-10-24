import struct
import numpy as np
#path = r'./202008132040.dat'

nx = 259
ny = 327
nz = 10

def readwind(path,nx, ny, nz):
    u = np.zeros((nx, ny, nz))
    v = np.zeros((nx, ny, nz))
    w = np.zeros((nx, ny, nz))
    with open(path,'rb') as f:
        f.seek(444)
        windsize = 3*nx*ny*nz
        wind = f.read(windsize)
        #wind = list(struct.unpack('=2540790B', f.read(windsize)))
       # print(type(wind))
        item = 0


        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    pwind = list(struct.unpack('=3B',wind[item:item+3]))
                    #print(pwind,item)
                    item = item + 3
                    if (pwind[0] != 255 and pwind[0] > 3):
                        u[i][j][k] = (pwind[0] - 129.) / 2.
                    else:
                        u[i][j][k] = 0.
                    if (pwind[1] != 255 and pwind[1] > 3):
                        v[i][j][k] = (pwind[1] - 129.) / 2.
                    else:
                        v[i][j][k] = 0.
                    if (pwind[2] != 255 and pwind[2] > 3):
                        w[i][j][k] = (pwind[2] - 129.) / 2.
                    else:
                        w[i][j][k] = 0.

        return (u,v,w)


#readwind(path,nx, ny, nz)