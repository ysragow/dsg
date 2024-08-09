from parallel import pooled_read, parallel_read, regular_read
from parquet import generate_two_column, generate_query, get_matching_files

s = 700

if __name__ == '__main__':
    name = 'test'
    print("Generating data...")
    generate_two_column(100000000, name, 90)
    print("Done")
    query = generate_query(1/10, 99999999, 0)
    print(query)
    files = get_matching_files(query, name)
    print(files)
    print("Parallel Reads")
    v = False
    # parallel_read(query, files, 10, timestamps=True, verbose=v)
    # parallel_read(query, files, 5, timestamps=True, verbose=v)
    # parallel_read(query, files, 2, timestamps=True, verbose=v)
    # parallel_read(query, files, 1, timestamps=True, verbose=v)
    print("Pooled Reads")
    pooled_read(query, files, 10, timestamps=True)
    pooled_read(query, files, 5, timestamps=True)
    pooled_read(query, files, 4, timestamps=True)
    pooled_read(query, files, 2, timestamps=True)
    pooled_read(query, files, 1, timestamps=True)
    print("Regular Read")
    regular_read(query, files, timestamps=True)
    # name = 'test1'
    # generate_two_column(100000000, name, 9)
    # query = generate_query(1 / 10, 99999999, 0, blower=50)
    # files = get_matching_files(query, name)
    # print(pooled_read(query, files, 10, timestamps=True))
    # print(pooled_read(query, files, 1, timestamps=True))
