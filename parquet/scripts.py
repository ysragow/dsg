from sys import argv
from json import load, dump
from params import queries
from metaparams import read


def get_average(n, v=False):
    """
    Takes the average over n runs of query.py and writes it to query_times.json
    :param n: the number of times to run query.py
    """

    import query
    from numpy import sqrt

    first = True
    second = False
    data = None
    variance = {}
    name = query.name
    print(name + '/query_times.json')
    for i in range(n):
        with open('query.py', 'r') as q:
            query.main(v)
        with open(name + '/query_times.json', 'r') as q:
            new_data = load(q)
        if first:
            data = new_data
            first = False
            second = True
        else:
            for q_type in data.keys():
                q_data = data[q_type]
                if second:
                    variance[q_type] = {}
                vq_data = variance[q_type]
                nq_data = new_data[q_type]
                for num_part in q_data.keys():
                    if q_type == 'regular':
                        if second:
                            vq_data[num_part] = q_data[num_part] ** 2
                        vq_data[num_part] += (nq_data[num_part] ** 2)
                        q_data[num_part] += nq_data[num_part]
                        # print('Total for {} with {} partitions: {}'.format(q_type, num_part, q_data[num_part]))
                        if i == (n - 1):
                            q_data[num_part] /= n
                            vq_data[num_part] /= n
                            vq_data[num_part] -= (q_data[num_part] ** 2)
                            vq_data[num_part] = sqrt(vq_data[num_part])
                    else:
                        p_data = q_data[num_part]
                        np_data = nq_data[num_part]
                        if second:
                            vq_data[num_part] = {}
                        vp_data = vq_data[num_part]
                        for num_proc in p_data.keys():
                            if second:
                                vp_data[num_proc] = p_data[num_proc] ** 2
                            vp_data[num_proc] += (np_data[num_proc] ** 2)
                            p_data[num_proc] += np_data[num_proc]
                            # print('Total for {} with {} partitions and {} processes: {}'.format(q_type, num_part, num_proc, p_data[num_proc]))
                            if i == (n - 1):
                                p_data[num_proc] /= n
                                vp_data[num_proc] /= n
                                vp_data[num_proc] -= (p_data[num_proc] ** 2)
                                vp_data[num_proc] = sqrt(vp_data[num_proc])
            if second:
                second = False
    print('Writing')
    with open(name + '/query_times.json', 'w') as q:
        dump(data, q)
    with open(name + '/variances.json', 'w') as q:
        dump(variance, q)


def get_min(j, path=None):
    best = []
    minimum = None
    if path is None:
        path = []
        with open(j, 'r') as file:
            data = load(file)
    else:
        data = j
    if type(data) == dict:
        for key in data.keys():
            sub_min, sub_best = get_min(data[key], path + [key])
            if minimum is None:
                minimum = sub_min
                best = sub_best
            elif sub_min < minimum:
                minimum = sub_min
                best = sub_best
        return minimum, best
    elif type(data) in (float, int):
        return data, path
    else:
        raise Exception('Invalid data. {} is not a dictionary or a number'.format(data))


def get_test(files_1, files_2, save_data=False, print_data=0, verbosity=False, func='regular', num_proc=10):
    """
    Run sanity tests by running the same query on two sets of files
    :param files_1: The first set of files
    :param files_2: The second set of files
    :param save_data: Whether to save the outputs to 'data_1.parquet' and 'data_2.parquet'
    :param print_data: How many lines to print of the output of each file
    :param verbosity: Whether to print timestamps in the running of pooled_read
    :param func: Which function to use to read the data (one of regular, parallel, or pooled)
    :param num_proc: How many processes to use.  Default 10
    """

    from parallel import pooled_read, parallel_read, regular_read
    from pyarrow.parquet import write_table
    import pyarrow as pa

    filters = [queries[0]]
    kwargs = {'scan': False, 'verbose': verbosity}
    f = None
    if func == 'regular':
        f = regular_read
        argmnts = [filters, files_1]
    elif func == 'parallel':
        f = parallel_read
        argmnts = [filters, files_1, num_proc]
    elif func == 'pooled':
        f = pooled_read
        argmnts = [filters, files_1, num_proc]
    else:
        raise Exception("func must be one of 'regular', 'parallel', or 'pooled'")
    # print(func + ' read with args ' + str(argmnts) + ' and kwargs ' + str(kwargs))
    output_1_og = f(*argmnts, **kwargs)
    argmnts[1] = files_2
    output_2_og = f(*argmnts, **kwargs)
    if save_data:
        write_table(output_1_og, 'data_1.parquet')
        write_table(output_2_og, 'data_2.parquet')
    if print_data != 0:
        if read == 'pyarrow':
            output_1 = output_1_og.to_pandas()
            output_2 = output_2_og.to_pandas()
            print('Data from source 1:')
            print(output_1.head(print_data))
            print('Data from source 2:')
            print(output_2.head(print_data))
            del output_1
            del output_2
        elif read == 'fastparquet':
            print('Data from source 1:')
            print(output_1_og.head(print_data))
            print('Data from source 2:')
            print(output_2_og.head(print_data))
    if read == 'fastparquet':
        output_1_og = output_1_og.reset_index(drop=True)
        output_2_og = output_2_og.reset_index(drop=True)
        output_1_og = pa.Table.from_pandas(output_1_og)
        output_2_og = pa.Table.from_pandas(output_2_og)
    output_1_og = output_1_og.sort_by('A')
    output_2_og = output_2_og.sort_by('A')
    output_1 = output_1_og.to_pandas()
    output_2 = output_2_og.to_pandas()
    is_equal = output_1.equals(output_2)
    if is_equal:
        print("They are equal")
    else:
        print("They are not equal")
        print("Differences:")
        print(output_1.compare(output_2))


def get_read_all(path, rfunc=None, filters=None):
    """
    Read all files at this path
    :param path: path to the directory containing the files to be read
    :param rfunc: function with which to read parquet files
    """

    from pandas import concat
    from time import time

    if rfunc is None:
        from pyarrow.parquet import read_table
        rfunc = read_table
    elif rfunc == 'parquet':
        from fastparquet import ParquetFile
        rfunc = ParquetFile

    print("Starting...")
    start_time = time()
    with open(path + '/files.json') as f:
        files = load(f)[0]
    data = []
    for f in files:
        data.append(rfunc(f).to_pandas())
    print("Concatenating...")
    output = concat(data)
    end_time = time()
    print("Found {} rows in {} seconds".format(output.shape[0], end_time - start_time))


def get_get(path, *keys):
    with open(path, 'r') as f:
        data = load(f)
    for key in keys:
        data = data[key]
    return data


if __name__ == '__main__':
    if len(argv) < 2:
        raise Exception("Which function do you want to call?")
    elif argv[1] == 'min':
        if len(argv) != 3:
            raise Exception("This function takes 1 argument")
        print(get_min(argv[2]))
    elif argv[1] == 'mean':
        if len(argv) not in (3, 4):
            raise Exception("This function takes 1 or 2 arguments argument")
        verbose = False
        if len(argv) == 4:
            if argv[3] == '-v':
                verbose = True
        get_average(int(argv[2]), verbose)
    elif argv[1] == 'test':
        if len(argv) < 4:
            raise Exception('This function takes 2 or more arguments')
        args = []
        all_kwargs = {}
        kwarg_list = []
        for i in range(2):
            with open(argv[2 + i] + '/files.json') as file:
                args.append(load(file)[0])
        if len(argv) > 4:
            assert argv[4][0] == '-', 'You cannot have more than 2 arguments without specifying them first'
            duplicated = False
            for c in argv[4]:
                if c == '-':
                    continue
                elif c == 'v':
                    all_kwargs['verbosity'] = True
                elif c == 's':
                    all_kwargs['save_data'] = True
                elif c in 'nfp':
                    if c in kwarg_list:
                        duplicated = True
                    kwarg_list.append(c)
                else:
                    raise Exception('Unrecognized parameter: ' + c)
            assert not duplicated, 'Duplicated argument'
            assert len(kwarg_list) + 5 == len(argv), 'Number of arguments given does not match number of parameters requiring arguments'
            arg_dict = {'n': 'num_proc', 'p': 'print_data', 'f': 'func'}
            for i in range(len(kwarg_list)):
                all_kwargs[arg_dict[kwarg_list[i]]] = argv[i + 5] if kwarg_list[i] == 'f' else int(argv[i + 5])
        get_test(*args, **all_kwargs)
    elif argv[1] == 'read':
        if len(argv) not in (3, 4):
            raise Exception("This function takes 1 or 2 arguments")
        if len(argv) == 4:
            if argv[3] == '-f':
                get_read_all(argv[2], 'parquet')
            else:
                raise Exception(argv[3] + ' is not a valid parameter')
        else:
            get_read_all(argv[2])
    elif argv[1] == 'get':
        if len(argv) < 3:
            raise Exception("This function takes 1 or more arguments")
        print(get_get(*argv[2:]))
    else:
        print('Not a valid function')


