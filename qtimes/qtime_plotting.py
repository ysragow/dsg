import matplotlib.pyplot as plt
from json import load
from numpy import array

tpch_sizes = [10000000, 5000000, 2000000, 1000000, 500000, 200000, 100000, 50000, 20000, 10000]
man_sizes = [10 * size for size in tpch_sizes]

color_map = {
    'Qd-tree': 'tab:orange',
    'P-Qd 5': 'tab:red',
    'P-Qd 10': 'tab:green',
    'P-Qd 20': 'tab:purple',
    'Baseline': 'tab:blue'
}

for folder in ('man/10',):
    is_tpch = folder[0:4] == 'tpch'
    SMALL_SIZE = 14
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)
    plt.xscale('log')
    plt.ylabel('Seconds')
    plt.xlabel('Block Size')
    stuff_to_plot = {'P-Qd 5': [], 'P-Qd 10': [], 'P-Qd 20': [], 'Qd-tree': [], 'Baseline': []}
    sizes = tpch_sizes if is_tpch else man_sizes
    with open(folder + '/qd.json', "r") as file:
        qd_data = load(file)['pooled']
    with open(folder + '/baseline.json', "r") as file:
        base_data = load(file)['pooled']
    for size in sizes:
        with open(f'{folder}/{size}.json') as file:
            size_data = load(file)['pooled']
        ssize = str(size)
        stuff_to_plot['Qd-tree'].append([size, qd_data[ssize]['10']])
        found_size = size if ((size > 5000000) or is_tpch) else (2 * size)
        stuff_to_plot['Baseline'].append([found_size, base_data['1-' + ssize]['10']])
        for factor in (5, 10, 20):
            stuff_to_plot[f'P-Qd {factor}'].append([size, size_data[f'{factor}-{size}']['10']])
    for line in stuff_to_plot.keys():
        x, y = array(stuff_to_plot[line]).T
        y = y / 10
        plt.plot(x, y, label=line, color=color_map[line])
    plt.legend(loc='upper left')
    plt.title("Manufactured data, 10% Queries")
    # plt.set_figwidth(10)
    # plt.set_figheight(10)
    plt.savefig(folder + '/plot.png')
    plt.clf()
    
        
        
        
    
