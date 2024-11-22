from warnings import filterwarnings
from params import name, partitions, verbosity_2, timestamps, processes, queries, query_objects, query_types, scan
from metaparams import read
from parallel import parallel_read, pooled_read, regular_read
from json import load, dump
from numpy import datetime64
from pyarrow import scalar
from datetime import datetime
import os
import subprocess
from sys import argv


filterwarnings('ignore')


def drop_caches():
    print("Dropping caches", end='\r')
    subprocess.run('drop_caches')
    print("Done", end='\r')


def run_all(f, files, args, kwargs, drop=False):
    """
    Run a large number of queries
    :param f: function to run for querying
    :param files: files to query
    :param args: args to f
    :param kwargs: kwargs to f
    :param drop: whether to drop caches after each query, or only after all queries are run
    :return:
    """
    total = 0
    for j in range(len(queries)):
        q = []
        print(queries[j])
        # Eliminate the non-numerical predicates
        for dnf in queries[j]:
            if type(dnf[2]) in (float, int):
                q.append(dnf)
            elif type(dnf[2]) == datetime64:
                date_str = str(dnf[2])
                arg_2 = '%Y-%m-%d'
                if len(date_str) != 10:
                    arg_2 += ' %H:%M:%S'
                q.append((dnf[0], dnf[1], scalar(datetime.strptime(date_str, arg_2)))
        print("Filters:", q)
        q = [q]
        query_files = files[j]
        # print(f.__repr__().split(' ')[1] + ' with args ' + str([q, query_files] + args) + 'and kwargs ' + str(kwargs))
        total += f(q, query_files, *args, **kwargs)
        if drop:
            drop_caches()
    if not drop:
        drop_caches()
    return total


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
    if 'regular' in query_types:
        reg_data = d1['regular']
        del d1['regular']
    for method in d1.keys():
        method_data = d1[method]
        sizes[method] = {}
        for part in method_data.keys():
            with open('{}/{}/files.json'.format(direc, part), 'r') as f:
                indexed_files = load(f)
            total_size = 0
            for pfiles in indexed_files:
                for pfile in pfiles:
                    total_size += os.path.getsize(pfile)
            sizes[method][part] = total_size
    output = remap(d1, lambda k1, v1: remap(v1, lambda k2, v2: remap(v2, lambda _, t: sizes[k1][k2] / t)))
    with open('{}/{}bandwidth.json'.format(direc, direc), 'w') as f:
        dump(output, f)


def main(verbosity=False):

    from params import queries, name

    offset = 0
    query = queries[0]
    drop_caches = False
    for i in range(len(argv)):
        if (i == 1) and (argv[1][0] == '-'):
            if 'v' in argv[1]:
                verbosity = True
            if 'n' in argv[1]:
                name = argv[2]
                offset += 1
            if 's' in argv[1]:
                query[0] = (query[0][0], query[0][1], int(argv[2 + offset]))
                query[1] = (query[1][0], query[1][1], int(argv[3 + offset]))
                queries = [query]
            if 'd' in argv[1]:
                drop_caches = True
    parallel_dict = {}
    pooled_dict = {}
    regular_dict = {}
    for partition_count in partitions:
        folder = '{}/{}'.format(name, partition_count)
        with open(folder + '/files.json') as file:
            files_list = load(file)
        parallel_times_dict = {}
        pooled_times_dict = {}
        if verbosity:
            print("Partitions: {}".format(partition_count))
        kwarg = {'scan': scan, 'timestamps': timestamps, 'verbose': verbosity}
        for process_count in processes:
            arg = [process_count]
            if 'parallel' in query_types:
                parallel_times_dict[process_count] = run_all(parallel_read, files_list, arg, kwarg, drop_caches)
            if 'pooled' in query_types:
                pooled_times_dict[process_count] = run_all(pooled_read, files_list, arg, kwarg, drop_caches)
        parallel_dict[partition_count] = parallel_times_dict
        pooled_dict[partition_count] = pooled_times_dict
        if 'regular' in query_types:
            regular_dict[partition_count] = run_all(regular_read, files_list, [], kwarg, drop_caches)
    if not scan:
        return
    overall_dict = {}
    if 'regular' in query_types:
        overall_dict['regular'] = regular_dict
    if 'parallel' in query_types:
        overall_dict['parallel'] = parallel_dict
    if 'pooled' in query_types:
        overall_dict['pooled'] = pooled_dict
    # overall_dict = {'regular': regular_dict, 'pooled': pooled_dict, 'parallel': parallel_dict}
    with open(name + '/query_times.json', 'w') as file:
        dump(overall_dict, file)
    # bandwidth(name)


if __name__ == '__main__':
    main()
