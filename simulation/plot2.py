import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np


data_firststep = np.load("./outputs/gfn/pre_train3.npy")
data_orthogonal = np.load("./outputs/gfn/final_result3.npy")
data_variance = np.load("./outputs/gfn/variance3.npy")

fig, ax_kwargs = plt.subplots(figsize=(6, 6))
ax_kwargs.plot(data_firststep[2,],data_firststep[3,],data_orthogonal[0,],data_orthogonal[1,])
lambda_, v = np.linalg.eig(data_variance)
angle = np.rad2deg(np.arccos(v[0,0]))
ell_radius_x = np.sqrt(-lambda_[0])
ell_radius_y = np.sqrt(-lambda_[1])
ellip = Ellipse((data_orthogonal[0,29],data_orthogonal[1,29]),width = ell_radius_x * 2, height = ell_radius_y * 2, facecolor = 'gray', angle = angle)
ax_kwargs.add_patch(ellip)
#ax_kwargs.legend(labels = ['initial step','orthogonal score'],loc=1)
plt.ylabel('D_2')
plt.xlabel('D_1')
plt.savefig("./pic/para3.png")
