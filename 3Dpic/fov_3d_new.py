import cv2
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, Point
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st

# calculate the iou between the roi and the fov
def get_iou(l1, l2):
    P = Polygon([Point(p) for p in l1])
    Q = Polygon([Point(p) for p in l2])
    return P.intersection(Q).area / P.area

# calculate the rotation matrix between two normal vector
def get_transform(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.dot(v1, v2))
    return cv2.Rodrigues(axis * angle)

# calculate the point between a plane and a line
def get_intersection(plane, line_pt, line_vec):
    b = np.dot(plane[:-1], line_pt) + plane[-1]
    print("line_pt", line_pt)
    a = np.dot(plane[:-1], line_vec)
    print("line_vec", line_vec)
    t = -b / a
    return line_pt + t * line_vec

# param of roi
# ex = 1640
# ey = 2560
# ez = 1117

html_temp = """
            <div style="background-color:#2B6695;padding:10px">
            <h2 style="color:white;text-align:center;">3D IMAGE</h2>
            </div>

            """
st.markdown(html_temp, unsafe_allow_html=True)

ex = st.number_input("please input ex", 0, 10000)
ey = st.number_input("please input ey", 0, 10000)
ez = st.number_input("please input ez", 0, 10000)
# focal distance
fl = st.number_input("please input fl", 0, 10000)

tx = 100
ty = 111
tz = ez
t = np.array((tx, ty, tz))
# vector of sight line of the camera
sight = (ex, ey/2, -ez * 1.4)

# fl = 16
unit = 3.45e-3
cx = unit * 4096
cy = unit * 3000

# get the 4 points of the ccd
trans = get_transform((0, 0, 1), sight)[0]
p1 = np.array((cx/2, cy/2, -fl))
print("p1", p1)
p2 = np.array((cx/2, -cy/2, -fl))
p3 = np.array((-cx/2, -cy/2, -fl))
p4 = np.array((-cx/2, cy/2, -fl))

plane = np.array((0, 0, 1, 0))

p1x = get_intersection(plane, t, np.dot(trans, p1))
print("p1x", p1x)
p2x = get_intersection(plane, t, np.dot(trans, p2))
p3x = get_intersection(plane, t, np.dot(trans, p3))
p4x = get_intersection(plane, t, np.dot(trans, p4))


rect = np.array([[0, 0], [ex, 0], [ex, ey], [0, ey], [0, 0]])
rect2 = np.array([p1x, p2x, p3x, p4x, p1x])[:, :2]


def plot_origin_cuboid():
    # 红框
    # (0, 0) -> (1640, 0)
    ax.plot3D([x, x+ex], [y, y], [z+ez, z+ez], **kwargs2)
    # (1640, 0) -> (1640, 2560)
    ax.plot3D([x+ex, x+ex], [y, y+ey], [z+ez, z+ez], **kwargs2)
    # (1640, 2560) -> (0, 2560)
    ax.plot3D([x+ex, x], [y+ey, y+ey], [z+ez, z+ez], **kwargs2)
    # (0, 2560) -> (0, 0)
    ax.plot3D([x, x], [y+ey, y], [z+ez, z+ez], **kwargs2)

    # 底面
    ax.plot3D([x, x+ex], [y, y], [z, z], **kwargs2)
    ax.plot3D([x+ex, x+ex], [y, y+ey], [z, z], **kwargs2)
    ax.plot3D([x+ex, x], [y+ey, y+ey], [z, z], **kwargs2)
    ax.plot3D([x, x], [y+ey, y], [z, z], **kwargs2)
    # 四条线
    ax.plot3D([x, x], [y, y], [z, z+ez], **kwargs2)
    ax.plot3D([x+ex, x+ex], [y, y], [z, z+ez], **kwargs2)
    ax.plot3D([x+ex, x+ex], [y+ey, y+ey], [z, z+ez], **kwargs2)
    ax.plot3D([x, x], [y+ey, y+ey], [z, z+ez], **kwargs2)

# 投影面填充颜色
def fill_groud():
    xxx = [p1x[0], p2x[0], p3x[0], p4x[0]]
    yyy = [p1x[1], p2x[1], p3x[1], p4x[1]]
    zzz = [z,z,z,z]
    verts = [list(zip(xxx, yyy, zzz))]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.5))

def plot_rays():
    ax.plot3D([x, p1x[0]], [x, p1x[1]], [x+ez, p1x[2]], **kwargs3)
    ax.plot3D([x, p2x[0]], [x, p2x[1]], [x+ez, p2x[2]], **kwargs3)
    ax.plot3D([x, p3x[0]], [x, p3x[1]], [x+ez, p3x[2]], **kwargs3)
    ax.plot3D([x, p4x[0]], [x, p4x[1]], [x+ez, p4x[2]], **kwargs3)

# 绘制投影面
# 蓝框
def plot_projection_surface():
    ax.plot3D([p1x[0], p2x[0]], [p1x[1], p2x[1]], [z, z], **kwargs1)
    # (1640, 0) -> (1640, 2560)
    ax.plot3D([p2x[0], p3x[0]], [p2x[1], p3x[1]], [z, z], **kwargs1)
    # (1640, 2560) -> (0, 2560)
    ax.plot3D([p3x[0], p4x[0]], [p3x[1], p4x[1]], [z, z], **kwargs1)
    # (0, 2560) -> (0, 0)
    ax.plot3D([p4x[0], p1x[0]], [p4x[1], p1x[1]], [z, z], **kwargs1)



if __name__ == '__main__':
    x = 0
    y = 0
    z = 0
    fig = plt.figure()
    ax = Axes3D(fig)
    # 设置颜色
    kwargs1 = {'color': "blue"}
    kwargs2 = {'color': "red"}
    kwargs3 = {'color': "green"}
    plot_origin_cuboid()
    plot_rays()
    plot_projection_surface()
    fill_groud()
    # 设置坐标系大小
    ax.set_xlim3d(0, ex * 1.1)
    ax.set_ylim3d(0, ey * 1.1)
    ax.set_zlim3d(0, ez * 1.1)
    # 坐标系自适应
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    st.pyplot(plt)
    plt.show()

