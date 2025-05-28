# boxplot

import matplotlib.pyplot as plt
import numpy as np

# x = np.array([[2.0882,1.6243,2.2224,2.6844],[2.1756,1.8689,2.5167,2.7778],[2.4906,2.0845,1.8613,3.1641],[0.8728,0.9495,1.1828,0.6894]])
x = np.array([[2.0882,1.6243,2.2224,2.6844],[2.1756,1.8689,2.5167,2.7778],[2.4906,2.0845,1.8613,3.1641],[5.5839,0.7794,1.4185,0.4297]])
x = x.T
plt.boxplot(x,patch_artist=True)
plt.ylabel('Values of C_D',fontsize=16)
plt.xticks(ticks=[1,2,3,4],labels=["semiPDE","Benchmark 1","Benchmark 2","Benchmark 3"],fontsize=12)
plt.savefig("./pic/boxplot_pic.png")