from parallel import pooled_read, parallel_read, regular_read, query_to_filters
from parquet import generate_two_column, generate_query, get_matching_files

s = 700

if __name__ == '__main__':
    name = 'test'
    # print("Generating data...")
    # generate_two_column(100000000, name, 90)
    # print("Done")
    query = generate_query(1/10, 99999999, 0)
    # print(query)
    files = get_matching_files(query, name)
    # print(files)
    query = query_to_filters(query)
    v = False
    t = False
    
    # print("Parallel Reads")
    # parallel_read(query, files, 10, timestamps=t, verbose=v)
    # parallel_read(query, files, 5, timestamps=t, verbose=v)
    # parallel_read(query, files, 2, timestamps=t, verbose=v)
    # parallel_read(query, files, 1, timestamps=t, verbose=v)
    
    # print("Parallel Scans")
    # pooled_read(query, files, 10, timestamps=t)
    # pooled_read(query, files, 5, timestamps=t)
    # pooled_read(query, files, 4, timestamps=t)
    # pooled_read(query, files, 2, timestamps=t)
    # pooled_read(query, files, 1, timestamps=t)
    
    # print("Pooled Reads")
    parallel_read(query, files, 10, timestamps=t, verbose=v, scan=True)
    # parallel_read(query, files, 5, timestamps=t, verbose=v, scan=True)
    # parallel_read(query, files, 2, timestamps=t, verbose=v, scan=True)
    # parallel_read(query, files, 1, timestamps=t, verbose=v, scan=True)
    
    # print("Pooled Scans")
    pooled_read(query, files, 10, scan=True, timestamps=t)
    # pooled_read(query, files, 5, scan=True, timestamps=t)
    # pooled_read(query, files, 4, scan=True, timestamps=t)
    # pooled_read(query, files, 2, scan=True, timestamps=t)
    # pooled_read(query, files, 1, scan=True, timestamps=t)
    
    # print("Regular Scan")
    regular_read(query, files, timestamps=t, scan=True)
    # name = 'test1'
    # generate_two_column(100000000, name, 9)
    # query = generate_query(1 / 10, 99999999, 0, blower=50)
    # files = get_matching_files(query, name)
    # print(pooled_read(query, files, 10, timestamps=t))
    # print(pooled_read(query, files, 1, timestamps=t))
