import matplotlib.pyplot as plt

def read(name):
    xx, y = [],[]
    
    with open(name, 'r') as f:
        for x in f.readlines():
            p = [x.strip() for x in x.strip().split('\t') if x != '']

            xx.append(int(p[0]))
            y.append(float(p[1]))

    return xx, y

names = ['Device to Device Bandwidth.o', 'Host to Device Bandwidth.o', 'Device to Host Bandwidth.o']
fig, axs = plt.subplots(3)
for i,name in enumerate(names):
    x, y = read(name)
    axs[i].plot(x, y)
    axs[i].set_title(name)


for ax in axs.flat:
    ax.set(xlabel = 'Bytes', ylabel = 'MB/s')

fig.tight_layout()
plt.show()