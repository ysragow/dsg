from warnings import filterwarnings
from matplotlib import pyplot as plt
from json import load
import numpy as np

filterwarnings('ignore')


def transform(d):
    return np.array(list(d.items())).astype(float).T


if __name__ == '__main__':
    for i in range(3):
        size = (10**9) // (2**i)
        b = 'bandwidth'
        with open('{}s/dat{}{}.json'.format(b, chr(97 + i), b)) as file:
            data = transform(load(file)['parallel']['8'])
        if i == 0:
            label = '1570 MB'
        elif i == 1:
            label = '780 MB'
        else:
            label = '390 MB'
        plt.plot(data[0], data[1], label=label)
    for i in range(3):
        size = (10**9) // (2**i)
        b = 'bandwidth'
        with open('{}s2/dat{}{}.json'.format(b, chr(97 + i), b)) as file:
            data = transform(load(file)['parallel']['256'])
        if i == 0:
            label = '49.1 MB'
        elif i == 1:
            label = '24.5 MB'
        else:
            label = '12.3 MB'
        plt.plot(data[0], data[1], label=label)
    plt.title('Bandwidth of Reading 2 Files')
    plt.xlabel('Processes')
    plt.ylabel('Gigabytes/Second')
    plt.legend(loc='upper left')
    plt.show()
    plt.savefig('images/2bandwidth2_image.png')
    plt.close()
    for fname in ('query_times', 'bandwidth'):
        with open(fname + '.json', 'r') as file:
            query_times = load(file)
        data_list = {}
        if fname == 'query_times':
            if 'regular' in query_times.keys():
                data_re = transform(query_times['regular'])
                data_list['regular'] = {'regular': data_re}
                del query_times['regular']
            ylabel = 'Seconds'
            outname = 'images/q_times_images/'
            loc = 'upper center'
        else:
            ylabel = 'Gigabytes/Second'
            outname = 'images/bandwidth_images/'
            loc = 'upper right'
        for method in query_times.keys():
            dic1 = query_times[method]
            dic2 = {}
            for partition_count in dic1.keys():
                for process_count in dic1[partition_count].keys():
                    dic2[process_count] = dic2.get(process_count, dict())
                    dic2[process_count][partition_count] = dic1[partition_count][process_count]
            data = dict([(key, transform(dic2[key])) for key in dic2.keys()])
            data_list[method] = data
        for key in data_list.keys():
            data = data_list[key]
            for label in data.keys():
                if True:  # label in ('regular', '10', '20', '30', '1'):
                    plt.plot(data[label][0], data[label][1], label=label)
            plt.xscale('log', base=2)
            plt.xlabel('Block Size (Rows)')
            plt.ylabel(ylabel)
            plt.legend(loc=loc)
            plt.show()
            print(outname + key + '_image.png')
            plt.savefig(outname + key + '_image.png')
            plt.close()
