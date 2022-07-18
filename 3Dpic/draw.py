from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
def plot_opague_cube(x=10, y=20, z=30, dx=40, dy=50, dz=60):
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')

    xx=np.linspace(x, x+dx, 2)
    print(xx)
    yy=np.linspace(y, y+dy, 2)
    zz=np.linspace(z, z+dz, 2)
    xx2, yy2=np.meshgrid(xx, yy)
    ax.plot_surface(xx2, yy2, np.full_like(xx2, z))
    ax.plot_surface(xx2, yy2, np.full_like(xx2, z+dz))
    # yy2, z2=np.meshgrid(yy, zz)
    # ax.plot_surface(np.full_like(yy2, x), yy2, z2)
    # ax.plot_surface(np.full_like(yy2, x+dx), yy2, z2)
    # xx2, zz2=np.meshgrid(xx, zz)
    # ax.plot_surface(xx2, np.full_like(yy2,y), z2)
    # ax.plot_surface(xx2, np.full_like(yy2,y+dy), z2)
    plt.show()
    plt.title("Cube")

plot_opague_cube()