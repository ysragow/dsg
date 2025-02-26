import matplotlib.pyplot as plt
from json import load, loads
from numpy import array

def load_data(f):
    suffix = '.txt' if AVG_BLOCKS else '.json'
    with open(f + suffix, 'r') as file:
        s = file.read()
    if not AVG_BLOCKS:
        return loads(s)['pooled']
    s = s.split('\n')
    totals = {}
    for i in s:
        sp = i.split(' ')
        params = sp[-4]
        totals[params] = totals.get(params, 0) + int(sp[-7])
    num_params = len(totals.keys())
    assert (len(s) % num_params) == 0
    output = {}
    for params in totals.keys():
        output[params] = {'10': totals[params] / (len(s) // num_params)}
    return output

tpch_sizes = [10000000, 5000000, 2000000, 1000000, 500000, 200000, 100000, 50000, 20000, 10000]
man_sizes = [10 * size for size in tpch_sizes]

AVG_BLOCKS = True


color_map = {
    'Qd-tree': 'tab:orange',
    'P-Qd 5': 'tab:red',
    'P-Qd 10': 'tab:green',
    'P-Qd 20': 'tab:purple',
    'Baseline': 'tab:blue'
}

for folder in ('../qblocks/tpch/3',):
    is_tpch = ('tpch' in folder)
    SMALL_SIZE = 14
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)
    plt.xscale('log')
    if AVG_BLOCKS:
        plt.ylabel('Average # of Blocks Accessed')
        plt.yscale('log', base=2)
    else:
        plt.ylabel('Seconds')
    plt.xlabel('Block Size')
    stuff_to_plot = {'P-Qd 5': [], 'P-Qd 10': [], 'P-Qd 20': [], 'Qd-tree': []}
    if not AVG_BLOCKS:
        stuff_to_plot['Baseline']= []
        base_data = load_data(folder + '/baseline')
    sizes = tpch_sizes if is_tpch else man_sizes
    qd_data = load_data(folder + '/qd')
    for size in sizes:
        size_data = load_data(f'{folder}/{size}')
        ssize = str(size)
        stuff_to_plot['Qd-tree'].append([size, qd_data[ssize]['10']])
        found_size = size if ((size > 5000000) or is_tpch) else (2 * size)
        if not AVG_BLOCKS:
            stuff_to_plot['Baseline'].append([found_size, base_data['1-' + ssize]['10']])
        for factor in (5, 10, 20):
            stuff_to_plot[f'P-Qd {factor}'].append([size, size_data[f'{factor}-{size}']['10']])
    for line in stuff_to_plot.keys():
        x, y = array(stuff_to_plot[line]).T
        y = y / (1 if AVG_BLOCKS else 10)
        plt.plot(x, y, label=line, color=color_map[line])
    plt.legend(loc='upper left')
    plt.title("TPC-H Data, TPC-H Query Template 3")
    # plt.set_figwidth(10)
    # plt.set_figheight(10)
    plt.savefig(folder + '/plot.png')
    plt.clf()
    
        
        
        
    
