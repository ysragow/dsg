import numpy as np
from qd.qd_table import tbl_to_parquet
from fastparquet import ParquetFile, write
import os
# from pandas import StringDtype


def sql_to_names(s):
    lines = s.lower().split(',\n')
    lines = [l.replace('\t', '') for l in lines]
    lines = [l.replace('  ', ' ') for l in lines]
    lines = [l.replace('  ', ' ') for l in lines]
    lines = [l.replace('  ', ' ') for l in lines]
    lines = [l.replace('  ', ' ') for l in lines]
    lines = [l.replace('  ', ' ') for l in lines]
    lines = [l.replace('  ', ' ') for l in lines]
    lines = [l[1:] if l[0] == ' ' else l for l in lines]
    split_lines = [l.split(' ', 1) for l in lines]
    names = []
    types = {}
    for name, dtype in split_lines:
        names.append(name)
        dtype = dtype.replace(' not null', '')
        if dtype == 'integer':
            types[name] = np.int64
        elif dtype == 'date':
            types[name] = np.datetime64
        elif 'decimal' in dtype:
            types[name] = np.float64
        elif 'char' in dtype:
            types[name] = int(dtype.split('(')[1][:-1])
    # print(names, '\n', types)
    return names, types


def rename(df, column):
    df.rename(columns={column: column.split('_')[1]}, inplace=True)


def kwargs(c, t):
    return {'left_on': key(c, t), 'right_on': key(c, c), 'how': 'inner'}



def key(c, t=None):
    keys = {'o': 'order', 'p': 'part', 's': 'supp', 'n': 'nation', 'r': 'region', 'c': 'cust'}
    return ('' if t is None else t + '_') + keys[c] + 'key'

# n_names = ['n_nationkey', 'n_name', 'n_regionkey', 'n_comment']
# n_types = {'n_nationkey': np.int64,
#            'n_name': 25,
#            'n_regionkey': np.int64,
#            'n_comment': 152}
#
# r_names = ['r_regionkey', 'r_name', 'r_comment']
# r_types = {'r_regionkey': np.int64,
#            'r_name': 25,
#            'r_comment': 152}

n_s = '''N_NATIONKEY  INTEGER NOT NULL,
                            N_NAME       CHAR(25) NOT NULL,
                            N_REGIONKEY  INTEGER NOT NULL,
                            N_COMMENT    VARCHAR(152)'''

r_s = '''R_REGIONKEY  INTEGER NOT NULL,
R_NAME CHAR(25) NOT NULL,
R_COMMENT    VARCHAR(152)'''

p_s = '''P_PARTKEY     INTEGER NOT NULL,
                          P_NAME        VARCHAR(55) NOT NULL,
                          P_MFGR        CHAR(25) NOT NULL,
                          P_BRAND       CHAR(10) NOT NULL,
                          P_TYPE        VARCHAR(25) NOT NULL,
                          P_SIZE        INTEGER NOT NULL,
                          P_CONTAINER   CHAR(10) NOT NULL,
                          P_RETAILPRICE DECIMAL(15,2) NOT NULL,
                          P_COMMENT     VARCHAR(23) NOT NULL'''

s_s = '''S_SUPPKEY     INTEGER NOT NULL,
                             S_NAME        CHAR(25) NOT NULL,
                             S_ADDRESS     VARCHAR(40) NOT NULL,
                             S_NATIONKEY   INTEGER NOT NULL,
                             S_PHONE       CHAR(15) NOT NULL,
                             S_ACCTBAL     DECIMAL(15,2) NOT NULL,
                             S_COMMENT     VARCHAR(101) NOT NULL'''

ps_s = '''PS_PARTKEY     INTEGER NOT NULL,
                             PS_SUPPKEY     INTEGER NOT NULL,
                             PS_AVAILQTY    INTEGER NOT NULL,
                             PS_SUPPLYCOST  DECIMAL(15,2)  NOT NULL,
                             PS_COMMENT     VARCHAR(199) NOT NULL'''

c_s = '''C_CUSTKEY     INTEGER NOT NULL,
                             C_NAME        VARCHAR(25) NOT NULL,
                             C_ADDRESS     VARCHAR(40) NOT NULL,
                             C_NATIONKEY   INTEGER NOT NULL,
                             C_PHONE       CHAR(15) NOT NULL,
                             C_ACCTBAL     DECIMAL(15,2)   NOT NULL,
                             C_MKTSEGMENT  CHAR(10) NOT NULL,
                             C_COMMENT     VARCHAR(117) NOT NULL'''

o_s = '''O_ORDERKEY       INTEGER NOT NULL,
                           O_CUSTKEY        INTEGER NOT NULL,
                           O_ORDERSTATUS    CHAR(1) NOT NULL,
                           O_TOTALPRICE     DECIMAL(15,2) NOT NULL,
                           O_ORDERDATE      DATE NOT NULL,
                           O_ORDERPRIORITY  CHAR(15) NOT NULL,
                           O_CLERK          CHAR(15) NOT NULL,
                           O_SHIPPRIORITY   INTEGER NOT NULL,
                           O_COMMENT        VARCHAR(79) NOT NULL'''

l_s = '''L_ORDERKEY    INTEGER NOT NULL,
                             L_PARTKEY     INTEGER NOT NULL,
                             L_SUPPKEY     INTEGER NOT NULL,
                             L_LINENUMBER  INTEGER NOT NULL,
                             L_QUANTITY    DECIMAL(15,2) NOT NULL,
                             L_EXTENDEDPRICE  DECIMAL(15,2) NOT NULL,
                             L_DISCOUNT    DECIMAL(15,2) NOT NULL,
                             L_TAX         DECIMAL(15,2) NOT NULL,
                             L_RETURNFLAG  CHAR(1) NOT NULL,
                             L_LINESTATUS  CHAR(1) NOT NULL,
                             L_SHIPDATE    DATE NOT NULL,
                             L_COMMITDATE  DATE NOT NULL,
                             L_RECEIPTDATE DATE NOT NULL,
                             L_SHIPINSTRUCT CHAR(25) NOT NULL,
                             L_SHIPMODE     CHAR(10) NOT NULL,
                             L_COMMENT      VARCHAR(44) NOT NULL'''

table_list = list(['tpch_data/' + s + '.parquet' for s in (
    'nation',
    'customer',
    'lineitem',
    'orders',
    'part',
    'partsupp',
    'region',
    'supplier',
)])


if __name__ == '__main__':
    n_names, n_types = sql_to_names(n_s)
    r_names, r_types = sql_to_names(r_s)
    p_names, p_types = sql_to_names(p_s)
    s_names, s_types = sql_to_names(s_s)
    ps_names, ps_types = sql_to_names(ps_s)
    c_names, c_types = sql_to_names(c_s)
    o_names, o_types = sql_to_names(o_s)
    l_names, l_types = sql_to_names(l_s)
    all_tuples = [
        (n_names, n_types, 'nation.tbl'),
        (r_names, r_types, 'region.tbl'),
        (p_names, p_types, 'part.tbl'),
        (s_names, s_types, 'supplier.tbl'),
        (ps_names, ps_types, 'partsupp.tbl'),
        (c_names, c_types, 'customer.tbl'),
        (o_names, o_types, 'orders.tbl'),
        (l_names, l_types, 'lineitem.tbl'),
    ]
    for t_names, t_types, t_path in all_tuples:
        path_name = 'tpch_data/' + t_path
        new_path_name = path_name[:-3] + 'parquet'
        if os.path.exists(new_path_name):
            os.remove(new_path_name)
        tbl_to_parquet(path_name, t_names, t_types)
    l_f = ParquetFile('tpch_data/lineitem.parquet').to_pandas()
    o_f = ParquetFile('tpch_data/orders.parquet').to_pandas()
    print("Joining l and o...")
    lo_f = l_f.merge(o_f, **kwargs('o', 'l'))
    lo_f.drop(key('o', 'o'), axis=1, inplace=True)
    print("Size:", lo_f.shape)
    # lo_alt = l_f.join(o_f, on=key('o'), how='left', rsuffix='r')
    # lo_out1 = lo_alt[lo_alt[key('o') + 'r'].isnull()]
    # lo_out2 = lo_out1.dropna(axis=1, how='all')
    # lo_alt.drop(key('o') + 'l', axis=1, inplace=True)
    # write('tpch_data/dropped_l.parquet', lo_out2)
    del l_f
    del o_f
    p_f = ParquetFile('tpch_data/part.parquet').to_pandas()
    s_f = ParquetFile('tpch_data/supplier.parquet').to_pandas()
    ps_f = ParquetFile('tpch_data/partsupp.parquet').to_pandas()
    print('Joining p, s, and ps...')
    print('p', p_f.columns)
    print('s', s_f.columns)
    print('ps', ps_f.columns)
    sp1_f = ps_f.merge(p_f, **kwargs('p', 'ps'))
    sp_f = sp1_f.merge(s_f, **kwargs('s', 'ps'))
    sp_f.drop(key('s', 's'), axis=1, inplace=True)
    sp_f.drop(key('p', 'p'), axis=1, inplace=True)
    del p_f
    del s_f
    del ps_f
    sp_f['ps'] = sp_f[key('p', 'ps')] + sp_f[key('s', 'ps')]
    lo_f['ps'] = lo_f[key('p', 'l')] + lo_f[key('s', 'l')]
    print('Joining lo to sp...')
    lops_f = lo_f.merge(sp_f, left_on='ps', right_on='ps', how='inner')
    lops_f.drop('ps', axis=1, inplace=True)
    # lops_f.drop('psl', axis=1, inplace=True)
    lops_f.drop(key('p', 'ps'), axis=1, inplace=True)
    lops_f.drop(key('s', 'ps'), axis=1, inplace=True)
    print("Size:", lops_f.shape)
    del sp_f
    del lo_f
    c_f = ParquetFile('tpch_data/customer.parquet').to_pandas()
    print('Joining c...')
    clops_f = lops_f.merge(c_f, **kwargs('c', 'o'))
    clops_f.drop(key('c', 'c'), axis=1, inplace=True)
    print("Size:", clops_f.shape)
    del lops_f
    del c_f
    n_f = ParquetFile('tpch_data/nation.parquet').to_pandas()
    print('Joining n...')
    nclops_f = clops_f.merge(n_f, **kwargs('n', 'c'))
    nclops_f.drop(key('n', 'n'), axis=1, inplace=True)
    print("Size:", nclops_f.shape)
    del clops_f
    del n_f
    r_f = ParquetFile('tpch_data/region.parquet').to_pandas()
    print('Joining r...')
    rnclops_f = nclops_f.merge(r_f, **kwargs('r', 'n'))
    rnclops_f.drop(key('r', 'r'), axis=1, inplace=True)
    print("Size:", nclops_f.shape)
    del nclops_f
    del r_f
    write('tpch_data/tpch.parquet', rnclops_f)



