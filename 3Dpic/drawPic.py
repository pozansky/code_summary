import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from FillBetween3d import fill_between_3d
def plot_linear_cub(x, y, z, dx, dy, dz, color="blue"):
    fig = plt.figure()
    ax = Axes3D(fig)
    kwargs = {'color': color}
    xx = [x, x, x+dx, x+dx, x]
    yy = [y, y+dy, y+dy, y, y]
    # 长方体上面的面
    ax.plot3D(xx, yy, [z] * 5, **kwargs)
    # ax.plot3D(xx, yy, [z+10] * 5, **kwargs)
    ax.plot3D(xx, yy, [z + dz] * 5, **kwargs)

    # ax.plot3D(xx, yy, [z + 50] * 5)
    ax.plot3D([x, x], [y, y], [z, z + dz], **kwargs)
    # # 一条斜线
    # ax.plot3D([x, x], [y, y+dy], [z, z + dz], **kwargs)
    # # 长斜线
    # ax.plot3D([x, x+dx], [y, y + dy], [z, z + dz], **kwargs)
    # 下部长斜线
    # ax.plot3D([x, x + dx], [y, y + dy], [z, z+dz/2])
    ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)

    ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z + dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y, y], [z, z + dz], **kwargs)



    set1 = [[x, x], [y, y], [z, z + dz/2]]  # x, y, z coordinates of the first line (x1, y1, z1)
    set2 = [[x, x], [y+dy, y + dy], [z, z + dz/2]]
    set3 = [[x+dx/2, x+dx/2], [y+dy, y+dy], [z, z + dz/2]]  # x, y, z coordinates of the first line (x1, y1, z1)
    set4 = [[x+dx, x+dx], [y, y], [z, z + dz/2]]
    ax.plot(*set1, lw=4)
    ax.plot(*set2, lw=4)
    ax.plot(*set3, lw=4)
    ax.plot(*set4, lw=4)
    fill_between_3d(ax, *set1, *set2, *set3, *set4, mode=1, c="C0")
    # ax.add_collection3d(plt.fill_between(*set1, *set2), zs=100, zdir='y')


    # ax.fill_between(*set1, *set2)

    plt.show()


plot_linear_cub(0, 0, 0, 100, 100, 100)