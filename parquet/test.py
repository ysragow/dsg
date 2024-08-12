from warnings import filterwarnings
from parallel import pooled_read, parallel_read, regular_read, query_to_filters
from parquet import generate_two_column, generate_query, get_matching_files
import json

filterwarnings('ignore')

if __name__ == '__main__':
    name = 'test'
    v = False
    t = True
    n = 20
    s = 10
    parallel_times = {}
    pooled_times = {}
    regular_times = {}

    for j in range(n-1):
        p = s * (j + 1)
        print("Generating data with {} partitions...".format(p))
        generate_two_column(1000000000, name, p)
        print("Done")
        query = generate_query(1/s, 999999999, 0)
        # print(query)
        files = get_matching_files(query, name)
        # files = ['test/27.parquet', 'test/28.parquet', 'test/29.parquet', 'test/30.parquet', 'test/31.parquet', 'test/32.parquet', 'test/33.parquet', 'test/34.parquet', 'test/35.parquet', 'test/36.parquet']
        # print(files)
        query = query_to_filters(query)
        # query = [('A', '<', 40495520.61431494), ('A', '>=', 30495520.71431494)]
        # print(query)

        # print("Parallel Reads")
        # parallel_read(query, files, 10, timestamps=t, verbose=v)
        # parallel_read(query, files, 5, timestamps=t, verbose=v)
        # parallel_read(query, files, 2, timestamps=t, verbose=v)
        # parallel_read(query, files, 1, timestamps=t, verbose=v)

        # print("Pooled Reads")
        # pooled_read(query, files, 10, timestamps=t)
        # pooled_read(query, files, 5, timestamps=t)
        # pooled_read(query, files, 4, timestamps=t)
        # pooled_read(query, files, 2, timestamps=t)
        # pooled_read(query, files, 1, timestamps=t)

        print("Parallel Scans")
        parallel_list = [None]*(n+1)
        for i in range(n):
            parallel_list[i+1] = parallel_read(query, files, i+1, timestamps=t, verbose=v, scan=True)
        parallel_times[j+2] = parallel_list

        print("Pooled Scans")
        pooled_list = [None]*(n+1)
        for i in range(n):
            pooled_list[i+1] = pooled_read(query, files, i+1, scan=True, timestamps=t)
        pooled_times[j+2] = pooled_list

        print("Regular Scan")
        regular_times[j+2] = regular_read(query, files, timestamps=t, scan=True)
        # name = 'test1'
        # generate_two_column(100000000, name, 9)
        # query = generate_query(1 / 10, 99999999, 0, blower=50)
        # files = get_matching_files(query, name)
        # print(pooled_read(query, files, 10, timestamps=t))
        # print(pooled_read(query, files, 1, timestamps=t))
    print("Pooled Times:")
    print(pooled_times)
    print("Parallel Times:")
    print(parallel_times)
    print("Regular Times:")
    print(regular_times)
    po = json.dumps(pooled_times)
    pa = json.dumps(parallel_times)
    re = json.dumps(regular_times)
    with open('times/pooled.json', 'w') as file:
        file.write(po)
    with open('times/parallel.json', 'w') as file:
        file.write(pa)
    with open('times/regular.json', 'w') as file:
        file.write(re)

