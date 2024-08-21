from warnings import filterwarnings
from params import name, partitions, verbosity, timestamps, processes, query
from parallel import parallel_read, pooled_read, regular_read
from json import load, dump
from os.path import getsize

filterwarnings('ignore')


def remap(dic, f):
    """
    :param dic: A dictionary
    :param f: A function mapping a key-value pair to a new key
    :return: A new dictionary mapping keys to the output of f
    """
    return dict([(key, f(key, dic[key])) for key in dic.keys()])


def bandwidth(direc):
    with open(direc + '/query_times.json', 'r') as f:
        d1 = load(f)
    sizes = {}
    reg_data = d1['regular']
    del d1['regular']
    for method in ('parallel', 'pooled'):
        method_data = d1[method]
        sizes[method] = {}
        for part in method_data.keys():
            with open('{}/{}/files.json'.format(direc, part), 'r') as f:
                indexed_files = load(f)
            total_size = 0
            for pfile in indexed_files:
                total_size += getsize(pfile)
            sizes[method][part] = total_size
    output = remap(d1, lambda k1, v1: remap(v1, lambda k2, v2: remap(v2, lambda _, t: sizes[k1][k2] / t)))
    with open('{}/{}bandwidth.json'.format(direc, direc), 'w') as f:
        dump(output, f)


if __name__ == '__main__':

    parallel_dict = {}
    pooled_dict = {}
    regular_dict = {}
    for partition_count in partitions:
        folder = '{}/{}'.format(name, partition_count)
        with open(folder + '/files.json') as file:
            files = load(file)
        parallel_times_dict = {}
        pooled_times_dict = {}
        for process_count in processes:
            print("Partitions: {}   Processes: {}".format(partition_count, process_count))
            pa_time = parallel_read(query, files, process_count, scan=True, timestamps=timestamps, verbose=verbosity)
            parallel_times_dict[process_count] = pa_time
            po_time = pooled_read(query, files, process_count, scan=True, timestamps=timestamps)
            pooled_times_dict[process_count] = po_time
        parallel_dict[partition_count] = parallel_times_dict
        pooled_dict[partition_count] = pooled_times_dict
        re_time = regular_read(query, files, scan=True, timestamps=timestamps)
        regular_dict[partition_count] = re_time
    overall_dict = {'regular': regular_dict, 'pooled': pooled_dict, 'parallel': parallel_dict}
    with open(name + '/query_times.json', 'w') as file:
        dump(overall_dict, file)
    bandwidth(name)
