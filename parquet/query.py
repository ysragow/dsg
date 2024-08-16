from warnings import filterwarnings
from params import name, partitions, verbosity, timestamps, processes, query
from parallel import parallel_read, pooled_read, regular_read
from json import load, dump

filterwarnings('ignore')

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
