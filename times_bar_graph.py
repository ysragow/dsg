import matplotlib.pyplot as plt
import numpy as np
from json import load

color_map = {
    'qd': 'tab:orange',
    'init': 'tab:green',
    'opt': 'tab:red',
    'format': 'tab:blue'
}


def bar_plot(path, base, out_file, block_size=True):
    """
    Plot a graph
    :param path: path to the json file to be working with
    :param base: the split factor or block size to keep constant
    :param out_file: file to save the graph to
    :param block_size: whether the base is a block size (True) or a split factor (False)
    """
    sort_d = lambda d: sorted(list(d.keys()), key=int)
    base = str(base)
    
    with open(path, "r") as file:
        data = load(file)
    if block_size:
        x_axis = sort_d(data)
    else:
        x_axis = sort_d(data[base])

    sections = color_map.keys()
    weight_counts = dict([(sect, []) for sect in sections])
    for x in x_axis:
        times = data[x][base] if block_size else data[base][x]
        for sect in sections:
            weight_counts[sect].append(times[sect])
    for sect in sections:
        weight_counts[sect] = np.array(weight_counts[sect])
    
    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(len(x_axis))
    if not block_size:
        new_x_axis = []
        for x in x_axis:
            e = '{' + str(len(x) - 1) + '}'
            pref = '' if (x[0] == '1') else (x[0] + '\\times')
            new_x_axis.append(f'${pref}10^{e}$')
        x_axis = new_x_axis
        fig.set_figwidth(10)
        fig.set_figheight(5)

    for sect, weight_count in weight_counts.items():
        p = ax.bar(x_axis, weight_count, width, label=sect, bottom=bottom, color=color_map[sect])
        bottom += weight_count

    ax.set_title(base)
    # ax.legend(loc="upper left")

    plt.savefig(out_file)
    plt.close()
