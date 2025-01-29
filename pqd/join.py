from pqd.split import PNode
from qd.qd_algorithms import index, table_gen
from qd.qd_predicate_subclasses import intersect, Numerical, Operator, pred_gen
from qd.qd_query import Query
from numpy import argsort as np_argsort
from fastparquet import ParquetFile, write
from pyarrow import parquet as pq, Table
from pandas import concat
from json import dump
import os


# Functions for file_gen

def list_sig(l):
    """
    Make a hashable signature for a list
    :param l: a list
    :return: The list sorted and transformed into a tuple
    """
    k = list(l)
    k.sort()
    return tuple(k)


def factors(n):
    # Get all factors of a number n
    # Start by getting prime factorization
    primes = {}  # one plus the number of times each prime appears in the prime factorization
    i = 2
    assert n > 0, "n needs to be greater than 0"
    assert n == int(n), "n needs to be an integer"
    while n != 1:
        if (i * (n // i)) == n:
            n = n // i
            primes[i] = primes.get(i, 1) + 1
        else:
            i += 1

    all_primes = list(primes.keys())
    # Set up the assignment process
    num_factors = 1
    p_switches = {}

    for p in all_primes:
        p_switches[p] = num_factors
        num_factors *= primes[p]

    # Get every factor
    output = []
    for i in range(num_factors):
        new_num = 1
        for p in all_primes:
            for _ in range((i // p_switches[p]) % primes[p]):
                new_num *= p
        output.append(new_num)
    return output


# LList and associated functions

class LList:
    def __init__(self, obj_list):
        self.rep = {}
        prev_obj = None
        obj = obj_list[0]
        self.first = obj
        for i in range(len(obj_list) - 1):
            next_obj = obj_list[i + 1]
            self.rep[obj] = (prev_obj, next_obj)
            prev_obj = obj
            obj = next_obj
        self.rep[obj] = (prev_obj, None)
        self.last = obj
        self.size = len(obj_list)

    def __len__(self):
        return self.size

    def __iter__(self):
        output = self.first
        while output is not None:
            yield output
            output = self.rep[output][1]

    def remove(self, obj):
        prev_obj, next_obj = self.rep[obj]
        if prev_obj is None:
            self.first = next_obj
        else:
            self.rep[prev_obj] = (self.rep[prev_obj][0], next_obj)
        if next_obj is None:
            self.last = prev_obj
        else:
            self.rep[next_obj] = (prev_obj, self.rep[next_obj][1])
        self.size -= 1
        del self.rep[obj]

    def pop(self):
        # Remove and return the top object
        output = self.last
        self.remove(self.last)
        return output


def argsort(obj_list, f):
    """
    Argsort a function and output a linked list
    :param obj_list: A list of hashable objects
    :param f: a function
    :return: The list of objects, sorted by the function
    """
    return list([obj_list[i] for i in np_argsort([f(obj) for obj in obj_list])])


# The PFile Class

class PFile:
    def __init__(self, file_list, size, is_overflow):
        """
        :param file_list: List of files
        :param size: Number of rows
        :param is_overflow: Whether this is an overflow file
        """
        self.file_list = file_list
        self.split_factor = 1
        self.size = size
        self.made = False

        # Initialize the queries (set of all queries that access this group of files),
        # and the relevant columns (dict mapping numeric columns accessed by this group of files to their bounds)
        self.queries = {}
        self.relevant_columns = set()
        # self.all_preds = ()

        self.path = 'None'
        self.indices = {}
        for file in file_list:
            self.indices[file] = (0,0)

    def __str__(self):
        output = '{'
        output += f"Files: {self.file_list}, "
        output += f"Size: {self.size}, "
        output += f"Path: {self.path}"
        output += '}'
        return output

    def __repr__(self):
        return str(self)

    def add_queries(self, q_ids, q_objs):
        """
        Add queries to the current set of queries
        :param q_ids: a list of query ids
        :param q_objs: a list of query objects associated with those ids
        """
        # self.queries = self.queries.union(q_ids)
        for i in range(len(q_ids)):
            q_id = q_ids[i]
            p_list = filter(lambda x: x.column.numerical & (not x.comparative), q_objs[i].list_preds())
            q_obj = Query(p_list, q_objs[i].table)
            self.queries[q_id] = q_obj
            for p in q_obj.list_preds():
                # We only care about columns that 1. we haven't seen before 2. are numerical and 3. non-comparative
                if p.column.numerical & (not p.comparative) & (p.column.name not in self.relevant_columns):
                    self.relevant_columns.add(p.column.name)
            # self.all_preds = sum(self.relevant_columns.values(), start=())

    def test_merge(self, other):
        """
        Test what would happen if this pfile merged with a different pfile
        :param other: a different pfile
        :return: A dictionary mapping query ids to the number of new files this would add to that query
        """
        output = {}
        for q in self.queries:
            if q not in other.queries:
                output[q] = other.split_factor
        for q in other.queries:
            if q not in self.queries:
                output[q] = self.split_factor
        return output

    def merge(self, other):
        """
        Merge with another pfile
        :param other: a different pfile
        :return: the result of test_merge on the other file, before the merger
        """
        output = self.test_merge(other)
        self.file_list += other.file_list
        for q, q_obj in other.queries.items():
            if q not in self.queries:
                self.queries[q] = q_obj
        self.split_factor += other.split_factor
        self.relevant_columns = self.relevant_columns.union(other.relevant_columns)
        return output


# Decorators for file_gen functions

def remove_index(func):
    """
    Decorator for file_gen functions to remove the index column if it exists
    :param func: a file_gen function of the pqd class, taking in self, file_path, obj_dict
    :return: A file_gen function, decorated to remove the index
    """
    def f(self, file_path, obj_dict):
        func(self, file_path, obj_dict)
        if 'index' in ParquetFile(file_path).columns:
            print(f"Removing index from {file_path}...", end='\r')
            pq.write_table(pq.read_table(file_path).drop('index'), file_path)
    return f


def rg_approx(func):
    """
    Decorator for file_gen functions to remove the index column if it exists
    :param func: a file_gen function of the pqd class, taking in self, file_path, obj_dict
    :return: A file_gen function, decorated to change the rg size and then change it back
    """
    def f(self, file_path, obj_dict):
        if self.approx_rg_size:
            total_size = sum(map(lambda x: x.shape[0], obj_dict.values()))
            temp_rg_size = self.rg_size
            min_rg_size = int(1 + (temp_rg_size * total_size / self.block_size))
            if self.limit_rg_usage:
                total_obj_size = sum(map(lambda x: ParquetFile(x).count(), obj_dict.keys()))
                split_factor = int(0.5 + (total_obj_size / total_size))
                print("Found Split Factor:", split_factor)
                self.rg_size = max(int(-(-total_size // split_factor)), min_rg_size)
            else:
                self.rg_size = min_rg_size
            func(self, file_path, obj_dict)
            self.rg_size = temp_rg_size
        else:
            func(self, file_path, obj_dict)
    return f


# The PQD Class

class PQD:
    def __init__(self, root_path, table, workload, block_size, split_factor, row_group_size=1000000, dp_factor=100, verbose=False, approx_rg_size=False, limit_rg_usage=False):
        """
        Initialize a PQD layout
        :param root_path: path to the root file, regardless of whether it exists
        :param a table object: used for indexing (necessary for resetting queries)
        :param workload: The workload upon which we will be creating this
        :param block_size: The block size (block_size <= # of rows per file <= 2 * block size)
        :param split_factor: How many ways to split the data
        :param row_group_size: Size of row groups that the data is made into
        :param dp_factor: Granularity of the dynamic programming by row count in file_gen_3
        :param verbose: whether to print stuff in initialization
        :param approx_rg_size: whether to approximate the row group size based on the block size instead of doing exactly what you are told
        """

        # Save relevant stuff
        if verbose:
            print("Initializing PQD object...", end='\r')
        self.block_size = block_size
        split_path = root_path.split('/')
        self.name = '.'.join(split_path[-1].split('.')[:-1])
        self.path = '/'.join(split_path[:-1])
        self.split_factor = split_factor
        self.split_factors = None  # Used for the new algorithm.  After massaging the layout, this will be defined
        self.workload = workload
        self.qd_index = lambda query, v=False: index(query, root_path, table, verbose=v)
        self.q_gen = lambda p_list: Query(p_list, table)
        self.pred_gen = lambda pred: pred_gen(pred, table)
        self.approx_rg_size = approx_rg_size
        self.limit_rg_usage = limit_rg_usage

        # For file_gen_3
        self.rg_size = row_group_size
        self.dp_factor = dp_factor
        self.rg_factors = argsort(factors(self.rg_size), lambda x: (self.rg_size - x)) # Contains every factor of the rg size
        self.opt_arrangements = {} # The optimal arrangement of a given set of objs within a file, so it doesn't have to be computed multiple times

        self.files = None  # This will change when we successfully run one of the "make_files" functions
        self.layout = None  # This will change when we successfully run one of the "make_layout" functions
        # self.file_dict = None # This will change when we successfully run one of the "make_layout" functions
        self.files_func = None  # Indicates which make_files function was used to make the files
        self.layout_func = None  # Indicates which make_layout function was used to make the files

        # The index
        #   - Queries learn which objs to access from the tree
        #   - Based on these objs, the index instructs on which files to read
        self.index = {}

        # Get a list of all files by indexing an empty query
        self.files_list = []

        # Initialize table_dict, table_q_dict, and the index
        self.table_dict = {}  # dict mapping file names to corresponding table objects
        self.table_q_dict = {}  # dict mapping file names to list of query ids in the workload which access it
        self.table_q_num_dict = {} # dict mapping file names to list of query ids on the workload which would access it in a row group skipping context
        self.total_size = 0
        all_objs = index(Query([], table), root_path, table)
        for obj in all_objs:
            self.total_size += ParquetFile(obj).count()
            self.index[obj] = []
            self.table_dict[obj] = table_gen(obj)
            self.table_q_dict[obj] = []
            self.table_q_num_dict[obj] = []

            # Don't bother dealing with files that are empty
            if self.table_dict[obj].size != 0:
                self.files_list.append(obj)

        # Index your workload on the tree
        self.indices = {}  # map of query ids to their files
        self.query_ids = {}  # maps query ids to queries
        num_columns = []
        for column in table.columns.values():
            if column.numerical:
                num_columns.append(column.name)
        id = 0
        for q in workload.queries:
            objs = index(q, root_path, table)
            self.indices[id] = objs
            self.query_ids[id] = q
            for obj in objs:
                if obj not in self.table_q_dict:
                    print(f"A contradiction has been found!  Query {q} accesses file {obj}, even though an empty query does not.")
                self.table_q_dict[obj].append(id)

            # Index it in the row group skipping way: without comparative preds or non-numerical columns
            num_q = Query(filter(lambda x: x.column.numerical & (not x.comparative), q.list_preds()), table)
            for obj in filter(lambda x: self.intersect(num_q, x, num_columns), all_objs):
                self.table_q_num_dict[obj].append(id)

            # # Remove any queries which are not in the table_q_num_dict
            # for obj in objs:
            #     if id not in self.table_q_num_dict[obj]:
            #         self.table_q_dict[obj].remove(id)

            # Increment the id
            id += 1

        # Assert that every leaf is queried
        for obj in all_objs:
            assert len(self.table_q_dict[obj]) > 0, f"The object {obj} is not queried"

        # Setup for all make_layout
        self.abstract_block_size = block_size * split_factor
        self.eff_size_dict = {}  # maps file names to their effective sizes (the size mod twice the abs blk size)
        for file in self.files_list:
            self.eff_size_dict[file] = self.table_dict[file].size % (2 * self.abstract_block_size)
        if verbose:
            print("Done.                                                        ")

    # General helper functions
    def layout_made(self):
        return self.layout is not None

    def files_made(self):
        return self.files is not None

    def get_sizes(self):
        if not self.layout_made():
            raise Exception("No layout has been made!")

    def get_queries(self, obj_list):
        """
        Extract the query ids from a list of objects
        :param obj_list: Iterable containing objects
        :return: The set of queries associated with these objects
        """
        output = set()
        for obj in obj_list:
            for q in self.table_q_dict[obj]:
                output.add(q)
        return output

    def rank_match(self, obj_list, obj, q_dict=None):
        """
        Decide how well an object matches up against a list of objects
        :param obj_list: A list of objects
        :param obj: The object, to test against the rest
        :param q_dict: A dictionary mapping query ids to a number.  Default maps every query to 1
        :return: A score for how well they align - smaller is better
        """
        all_qs = self.get_queries(obj_list)

        rank = 0
        for q in self.table_q_dict[obj]:
            if q not in all_qs:
                if q_dict is None:
                    rank += 1
                else:
                    rank += q_dict[q]
        return rank

    # Make layout functions
    def make_layout_qd(self):
        self.layout = [PFile([f], ParquetFile(f).count(), False) for f in self.qd_index(self.q_gen([]))]

    def make_layout_1(self, take_top=True, split_factor=None):
        """
        Create self.layout
        Algo specs:
            - 1. Split things with sizes greater than 2 * block_size * split_factor into full files
            - 2. Then, take remaining pieces and sort them by (size / workload)
            - 3. Place ones with largest (size * workload) first
            - 4. Place ones with highest percentage of matching queries until above ABS (if no more blocks exist, halt)
            - 5. Then, add largest blocks whose workload is a subset of the total workload on this file
            - 6. Once no more such files exist or we cannot add another without going over (2 * ABS), return to step 4
        :return: self.layout
        """
        temp_split_factor = self.split_factor
        if split_factor is not None:
            self.split_factor = 1

        # Initialize the layout list.  Assign this to self.layout at the end
        layout = []
        # file_dict = {}

        # Make the linked list
        size_llist = LList(argsort(self.files_list, lambda x: self.eff_size_dict[x] * len(self.table_q_dict[x])))

        # Enter the while loop, which does not cease until every obj is assigned
        while len(size_llist) > 0:
            # Initialize variables, including first obj
            if take_top:
                obj = size_llist.pop()
            else:
                obj = size_llist.first
                size_llist.remove(size_llist.first)
            current_file = [obj]  # the files that will be in this set
            total_size = self.eff_size_dict[obj]
            # file_dict[obj] = []
            queries_accessed = set(self.table_q_dict[obj])  # contains the ids of the queries that access this file

            # Initialize npq_dict: the dict mapping objs to a set of their queries not present in queries_accessed
            npq_dict = {}
            for obj_2 in size_llist:
                npq_dict[obj_2] = set()
                for q in self.table_q_dict[obj_2]:
                    if q not in queries_accessed:
                        npq_dict[obj_2].add(q)

            # Add more objects
            while total_size < self.abstract_block_size:
                # Get the new object and remove it from size_llist
                active_files = list([obj_2 for obj_2 in size_llist])
                if len(active_files) == 0:
                    break
                obj = argsort(active_files, lambda x: (len(npq_dict[x]) / len(self.table_q_dict[x])))[0]
                # file_dict[obj] = []
                size_llist.remove(obj)

                # Update layout with new object
                current_file.append(obj)
                total_size += self.eff_size_dict[obj]
                for q in self.table_q_dict[obj]:
                    queries_accessed.add(q)
                    for obj_2 in size_llist:
                        if q in npq_dict[obj_2]:
                            npq_dict[obj_2].remove(q)

            # Find all files whose queries are all in queries_accessed, sort them by size
            empty_files = []
            for obj in size_llist:
                if len(npq_dict[obj]) == 0:
                    empty_files.append(obj)
            empty_files = argsort(empty_files, lambda x: (1 / (self.table_dict[x].size + 1)))

            # Fill the file with relevant objects
            for obj in empty_files:
                # Skip the object if adding it would put us above block size
                if (total_size + self.eff_size_dict[obj]) > 2 * self.abstract_block_size:
                    continue

                # Add the current file, and change variables accordingly
                current_file.append(obj)
                size_llist.remove(obj)
                total_size += self.eff_size_dict[obj]

            # Finally, add the current file to the layout
            new_pfile = PFile(current_file, sum([self.eff_size_dict[obj] for obj in current_file]), False)
            # file_dict[obj].append(new_pfile)
            layout.append(new_pfile)

        # Assign the layout to self.layout
        self.layout = layout
        # self.file_dict = file_dict
        self.split_factor = temp_split_factor

    def make_layout_1a(self):
        self.make_layout_1(take_top=False)

    # Score functions for layout massaging
    def score_func_1a(self, split_factor, remainders, pfile1, pfile2):
        """
        Calculate the score
        * This one only uses the number of remaining files
        ** It blocks any merge that would result in going over the remainder
        :param split_factor: The split factor
        :param remainders: Negative the number of files accessed by each query, mod the split factor
        :param pfile1: A pfile object
        :param pfile2: Another pfile object
        :return: A score of how good merging these pfiles would be.  Higher is better.
        """
        score = 0
        test_dict = pfile1.test_merge(pfile2)
        for q, new_files in test_dict.items():
            # score -= split_factor * ((remainders[q] - new_files) // split_factor)
            # score += remainders[q] - ((remainders[q] - new_files) % split_factor)
            score += new_files
            if new_files > remainders[q]:
                return 0
        return score

    def score_func_1b(self, split_factor, remainders, pfile1, pfile2):
        """
        Calculate the score
        * This one only uses the number of remaining files
        ** It punishes going over the remainder by subtracting the split factor from the score
        :param split_factor: The split factor
        :param remainders: Negative the number of files accessed by each query, mod the split factor
        :param pfile1: A pfile object
        :param pfile2: Another pfile object
        :return: A score of how good merging these pfiles would be.  Higher is better.
        """
        score = 0
        test_dict = pfile1.test_merge(pfile2)
        for q, new_files in test_dict.items():
            score -= split_factor * ((remainders[q] - new_files) // split_factor)
            score += remainders[q] - ((remainders[q] - new_files) % split_factor)
        return score

    def intersect(self, q, file, relevant_columns):
        pq_file = ParquetFile(file)
        bounds = q.list_preds()
        for c_name in relevant_columns:
            bounds.append(self.pred_gen(f"{c_name} >= {pq_file.statistics['min'][c_name][0]}"))
            bounds.append(self.pred_gen(f"{c_name} <= {pq_file.statistics['max'][c_name][0]}"))
        return intersect(bounds)

    def score_func_2a(self, split_factor, remainders, pfile1, pfile2):
        """
        Calculate the score
        * This one uses the number of remaining files and the bounds
        ** Scores entirely based on how much it would reduce the percentage of files it needs to access in merged files
        ** It also blocks any merge that would result in going over the remainder
        :param split_factor: The split factor
        :param remainders: Negative the number of files accessed by each query, mod the split factor
        :param pfile1: A pfile object
        :param pfile2: Another pfile object
        :return: A score of how good merging these pfiles would be.  Higher is better.
        """
        score = 0
        total_qs = 0
        total_files_1 = len(pfile1.file_list)
        total_files_2 = len(pfile2.file_list)
        relevant_columns = pfile1.relevant_columns.union(pfile2.relevant_columns)
        test_dict = pfile1.test_merge(pfile2)
        for q, new_files in test_dict.items():
            total_qs += 1
            if new_files > remainders[q]:
                return 0
            total_intersected_1 = 0
            for obj in pfile1.file_list:
                total_intersected_1 += q in self.table_q_num_dict[obj]
            total_intersected_2 = 0
            for obj in pfile2.file_list:
                total_intersected_2 += q in self.table_q_num_dict[obj]
            in_1 = q in pfile1.queries
            in_2 = q in pfile2.queries
            new_match = (total_intersected_1 + total_intersected_2) / (total_files_1 + total_files_2)
            if in_1 & in_2:
                score += max(total_intersected_1 / total_files_1, total_intersected_2 / total_files_2) - new_match
            elif in_1:
                score += (total_intersected_1 / total_files_1) - new_match
            else:
                # Must be in pfile2
                score += (total_intersected_2 / total_files_2) - new_match
        return score

    # Layout post-processing functions
    def massage_layout(self, split_factor, score_func, verbose=False):
        """
        A greedy algorithm for pairing together objects
        - For each pfile, keep track of the queries which access it
        Modifies the layout
        :param split_factor: The number of parallel reads which can be performed simultaneously
        :param score_func: The function used to calculate the score
        :param verbose: whether to print things
        """
        if not self.layout_made():
            raise Exception("This function can only be called once the layout is made")
        index_all = [set(self.qd_index(q)) for q in self.workload.queries]
        for pfile in self.layout:
            new_queries = sum([list(self.table_q_dict[obj]) for obj in pfile.file_list], start=[])
            # print(new_queries)
            pfile.add_queries(new_queries, [self.workload.queries[q_id] for q_id in new_queries])

        # Make sure everything lines up
        for pfile in self.layout:
            for q in pfile.queries:
                assert len(index_all[q].intersection(pfile.file_list)) > 0, f"Query {self.workload.queries[q]} is listed as being in the pfile containing the files {pfile.file_list}, but it only indexes to {index_all[q]}."
        file_counts = []
        for i in range(len(index_all)):
            file_set = index_all[i]
            file_count = 0
            for pfile in self.layout:
                accessed = False
                for obj in pfile.file_list:
                    if obj in file_set:
                        assert i in pfile.queries, f"Query {self.workload[i]} maps to object {obj}, but is not in the queries assigned to access the pfile containing the files {pfile.file_list}."
                        if not accessed:
                            accessed = True
                            file_count += 1
            file_counts.append(file_count)

        # Initialize the remainders
        remainders = [(-s) % split_factor for s in file_counts]

        # Enter the while loop
        while remainders != [0]*len(index_all):
            if verbose:
                print("Remainders:", remainders)
            # Initialize the instruments used for finding the best pair
            best_score = 0
            best_pair = None
            for i in range(len(self.layout)):
                for j in range(i):
                    pfile1 = self.layout[i]
                    pfile2 = self.layout[j]

                    # Score the pair
                    score = score_func(split_factor, remainders, pfile1, pfile2)
                    if score > best_score:
                        best_pair = (i, j)
                        best_score = score
                    # if verbose:
                    #     print(f"Score for ({i}, {j}): {score}")

            # Break if nothing was found
            if best_score <= 0:
                break

            # Otherwise, continue.  Merge the pair
            i, j = best_pair
            best_dict = self.layout[i].merge(self.layout[j])

            # Subtract from the remainders
            for k in range(len(remainders)):
                if k in best_dict:
                    remainders[k] = (remainders[k] - best_dict[k]) % split_factor

            # Make a new layout
            new_layout = []
            for k in range(len(self.layout)):
                if k == j:
                    continue
                new_layout.append(self.layout[k])
            self.layout = new_layout

        # Define the split factors to be used when making files
        self.split_factors = [p.split_factor for p in self.layout]

    def make_files(self, folder_path, make_alg, verbose=False):
        """
        Generate files based on the current self.layout and make_alg
        :param folder_path: Folder for storing things in
        :param make_alg: which file_gen algorithm to use
        :param verbose: how much to print
        """
        # Set the split factor for each item in the layout
        if self.split_factors is None:
            self.split_factors = [self.split_factor] * len(self.layout)

        # Make the folders, if they don't exist
        file_template = folder_path + "/" + "{}/" + self.name + "{}.parquet"
        folder_template = folder_path + "/" + "{}"
        for i in range(max(self.split_factors)):
            folder = folder_template.format(i)
            if not os.path.exists(folder):
                os.makedirs(folder)

        total_og_size = 0  # size of input
        total_gen_size = 0  # size of output
        # Begin the for loop.  Load objects into memory.
        file_num = 0
        for k in range(len(self.layout)):
            split_factor = self.split_factors[k]
            pfile = self.layout[k]
            og_size = 0
            gen_size = 0
            # Initialize variables
            eff_dframes = {}  # Maps n to a dict mapping obj to the slice of the dframes[obj] going into f_path/n/pfile
            for i in range(split_factor):
                eff_dframes[i] = {}

            # Loop through all the objects in the pfile
            for obj in pfile.file_list:
                og_size += ParquetFile(obj).count()
                if verbose:
                    print(f"Loading dataframe for file {obj}...", end='\r')

                df = ParquetFile(obj).to_pandas()
                if verbose:
                    print(f"Loaded dataframe for file {obj} with {df.shape[0]} rows")
                df_index = 0  # Keep track of our index in the dataframe

                # Account for overflow
                for _ in range(self.table_dict[obj].size // (2 * self.block_size * split_factor)):
                    for i in range(split_factor):
                        file_chunk = df[df_index:df_index + (2 * self.block_size)]
                        chunk_file_name = file_template.format(i, file_num)
                        if verbose:
                            n_rows = file_chunk.shape[0]
                            print(f"Making overflow file {chunk_file_name} from file {obj} with {n_rows} rows and {-(-n_rows//self.rg_size)} row groups.")
                        make_alg(chunk_file_name, {obj: file_chunk})
                        gen_size += ParquetFile(chunk_file_name).count()
                        df_index += 2 * self.block_size
                    file_num += 1

                # Add to effective dframes
                eff_size = self.table_dict[obj].size % (2 * self.block_size * split_factor)
                if eff_size == 0:
                    print(f"The object {obj} has an effective size of 0")
                    continue
                ind_size = eff_size // split_factor  # size of each chunk
                cutoff = eff_size % split_factor  # cutoff for when to stop adding 1 to size
                for i in range(split_factor):
                    split_size = ind_size + (1 if i < cutoff else 0)
                    eff_dframes[i][obj] = df[df_index: df_index + split_size]
                    df_index += split_size

            # Create each file
            for i in range(split_factor):
                is_nonzero = False
                for dframe in eff_dframes[i].values():
                    is_nonzero = (dframe.shape[0] != 0)
                    if is_nonzero:
                        break
                if verbose and is_nonzero:
                    p_temp = "{} rows from file {}"
                    p_str = ", ".join([p_temp.format(eff_dframes[i][obj3].shape[0], obj3) for obj3 in eff_dframes[i].keys()])
                    print(f"Making file {file_template.format(i, file_num)} with {p_str}")
                if is_nonzero:
                    make_alg(file_template.format(i, file_num), eff_dframes[i])
                    gen_size += ParquetFile(file_template.format(i, file_num)).count()
            file_num += 1
            assert gen_size == og_size, f"{gen_size} rows were generated from an input of {og_size}"
            total_og_size += og_size
            total_gen_size += gen_size

        assert total_gen_size == total_og_size
        assert total_gen_size == self.total_size
        # Save the index
        with open(folder_path + "/index.json", "w") as file:
            dump(self.index, file)

    # file_gen functions for make_files
    @remove_index
    def file_gen_1(self, file_path, obj_dict):
        """
        Generate a file
        **This one does not pay attention to row groups**
        :param file_path: Folder for storing things in
        :param obj_dict: A dict mapping file names to pandas dataframes containing chunks of those files
        """
        # print(list(obj_dict.items()))

        # write(file_path, concat(obj_dict.values()).reset_index(drop=True))
        pq.write_table(Table.from_pandas(concat(obj_dict.values()).reset_index(drop=True)), file_path, row_group_size=self.rg_size)
        for obj in obj_dict.keys():
            self.index[obj].append(file_path)

    @remove_index
    def file_gen_1a(self, file_path, obj_dict):
        """
        Generate a file
        **This one gives each chunk its own row group**
        :param file_path: Folder for storing things in
        :param obj_dict: A dict mapping file names to pandas dataframes containing chunks of those files
        """

        objs = list(obj_dict.keys())
        total_size = 0
        rg_indices = []

        # Insert files into the index
        for obj in objs:
            self.index[obj].append(file_path)
            rg_indices.append(total_size)
            total_size += obj_dict[obj].shape[0]

        print(f"Row Group Indices for {file_path}:", rg_indices)

        write(file_path, concat([obj_dict[obj] for obj in objs]), row_group_offsets=rg_indices)

    @remove_index
    def file_gen_1b(self, file_path, obj_dict):
        """
        Generate a file
        **This one does not pay attention to row groups and resets the index**
        Literally just the original file_gen
        :param file_path: Folder for storing things in
        :param obj_dict: A dict mapping file names to pandas dataframes containing chunks of those files
        """
        write(file_path, concat(obj_dict.values()))
        for obj in obj_dict.keys():
            self.index[obj].append(file_path)


    @remove_index
    @rg_approx
    def file_gen_3a(self, file_path, obj_dict):
        """
        Generate a file
        **This one adds another column for ease of row group skipping**
        How it works
        - Try every ordering of the file chunks and see which one has queries accessing the fewest row groups
        - How to do efficiently?  Dynamic programming on every possible subset!
        - Make files in the optimal ordering
        This runs in exponential time, so only good for small number of files
        :param file_path: Folder for storing things in
        :param obj_dict: A dict mapping file names to pandas dataframes containing chunks of those files
        """

        objs = list(obj_dict.keys())

        llist_objs = LList(objs)
        total_size = 0
        min_obj_size = 0
        # all_qs = set() # The set of all queries present here
        num_rg = -((-total_size) // self.rg_size)

        for obj in objs:
            obj_size = obj_dict[obj].shape[0]
            total_size += obj_dict[obj].shape[0]
            if obj_size > 0:
                self.index[obj].append(file_path)
                # for q in self.table_q_dict[obj]:
                #     all_qs.add(q)
                if min_obj_size == 0:
                    min_obj_size = obj_size
                else:
                    min_obj_size = min(min_obj_size, obj_size)
            else:
                # We don't care about objects of size 0
                llist_objs.remove(obj)

        all_qs = self.get_queries(llist_objs)

        # If the list is truly empty, then don't write anything at all
        if len(llist_objs) == 0:
            return

        # If a layout for these files already exists, use that one instead of calculating a new one
        objs_sig = list_sig(llist_objs)
        if objs_sig in self.opt_arrangements:
            opt_order = self.opt_arrangements[objs_sig]
        else:
            # Otherwise, run this n^2 * 2^n time algorithm.  Start by initializing the new and old subsets/last pairs
            # SLP = subset/last pair: a tuple ((objs in a subset), last objs)
            max_score = num_rg * len(all_qs)

            q_dict = self.table_q_dict
            rg_size = self.rg_size
            
            # Initialize the bounds for every object so we don't have to do this in the for loop
            # c_maxes = {}
            # c_mins = {}
            num_columns = set()
            obj_bound_dict = {}
            for obj in objs:
                table = table_gen(obj)
                obj_bounds = {}
                for col in table.columns.values():
                    if col.numerical:
                        obj_bounds[col.name] = (col.min, col.max)
                        num_columns.add(col.name)

                        # Keep track of the mins and maxes so we can set defaults for the queries
                        # if col.name in c_maxes:
                        #     c_maxes[col.name] = max(c_maxes[col.name], col.max)
                        #     c_mins[col.name] = min(c_mins[col.name], col.min)
                        # else:
                        #     c_maxes[col.name] = col.max
                        #     c_mins[col.name] = col.min
                obj_bound_dict[obj] = obj_bounds
            # total_bounds = {}
            # for c in c_maxes.keys():
            #     total_bounds[c] = (c_mins[c], c_maxes[c])
            num_columns = list(num_columns)

            # Initialize the bounds for every query
            q_maxes_dict = {}
            q_mins_dict = {}
            for qid in all_qs:
                q = self.workload.queries[qid]
                q_maxes = {}
                q_mins = {}
                for pred in q.list_preds():
                    if (not pred.comparative) and (pred.column.numerical):
                        cname = pred.column.name
                        op = pred.op.symbol
                        if op in ("<", "<=", "="):
                            # This is a maximum
                            if cname not in q_maxes:
                                q_maxes[cname] = (pred.value, (op != '<'))
                            if pred.value < q_maxes[cname][0]:
                                q_maxes[cname] = (pred.value, (op != '<'))
                            if pred.value == q_maxes[cname][0]:
                                q_maxes[cname] = (pred.value, (op != '<') & q_maxes[cname][1])
                        if op in (">", ">=", "="):
                            # This is a minimum
                            if cname not in q_mins:
                                q_mins[cname] = (pred.value, (op != '>'))
                            if pred.value > q_mins[cname][0]:
                                q_mins[cname] = (pred.value, (op != '>'))
                            if pred.value == q_mins[cname][0]:
                                q_mins[cname] = (pred.value, (op != '>') & q_mins[cname][1])
                q_maxes_dict[qid] = q_maxes
                q_mins_dict[qid] = q_mins
                
            # Define the Ordering class
            class Ordering:
                def __init__(self, next_obj, prev=None):
                    if prev is None:
                        self.ordering = []
                        self.remainder = set(llist_objs)
                        self.sig = ()
                        self.last = None
                        self.score = 0
                        self.size = 0
                        self.last_rg_queries = set()
                        self.bounds = None  # Bounds of the current rg
                        return
                    self.last = next_obj

                    # Initialize the ordering and signature
                    self.ordering = prev.ordering.copy()
                    self.ordering.append(self.last)
                    self.sig = (list_sig(self.ordering), self.last)

                    # Initialize the remainder
                    self.remainder = prev.remainder.copy()
                    self.remainder.remove(self.last)

                    # Initialize the size
                    self.size = prev.size + obj_dict[next_obj].shape[0]

                    # Initialize the score
                    self.score = prev.score
                    self.last_rg_queries = set()

                    # Account for score accrued through new queries in the current row group, and make the new bounds
                    if prev.size % rg_size != 0:
                        if prev.bounds is None:
                            self.bounds = obj_bound_dict[next_obj].copy()
                        else:
                            self.bounds = {}
                            for c in num_columns:
                                self.bounds[c] = (min(prev.bounds[c][0], obj_bound_dict[next_obj][c][0]), max(prev.bounds[c][1], obj_bound_dict[next_obj][c][1]))
                        self.last_rg_queries = prev.last_rg_queries.copy()
                        for q in all_qs:
                            if q not in self.last_rg_queries:
                                # Test if adding this object would cause the query to intersect this
                                overlaps = True
                                for c, q_max in q_maxes_dict[q].items():
                                    new_min = self.bounds[c][0]
                                    if q_max[0] < new_min:
                                        overlaps = False
                                    elif (q_max[0] == new_min) and (not q_max[1]):
                                        overlaps = False
                                for c, q_min in q_mins_dict[q].items():
                                    new_max = self.bounds[c][1]
                                    if q_min[0] > new_max:
                                        overlaps = False
                                    elif (q_min[0] == new_max) and (not q_min[1]):
                                        overlaps = False
                                if overlaps:
                                    self.score += 1
                                    self.last_rg_queries.add(q)
                    else:
                        self.bounds = None

                    # Account for score accrued through making new row groups
                    new_rgs = (self.size // rg_size) - (prev.size // rg_size)
                    if new_rgs > 0:
                        self.bounds = obj_bound_dict[next_obj].copy()
                        self.last_rg_queries = set()
                        for q in all_qs:
                            # Test if adding this object would cause the query to intersect this
                            overlaps = True
                            for c, q_max in q_maxes_dict[q].items():
                                new_min = self.bounds[c][0]
                                if q_max[0] < new_min:
                                    overlaps = False
                                elif (q_max[0] == new_min) and (not q_max[1]):
                                    overlaps = False
                            for c, q_min in q_mins_dict[q].items():
                                new_max = self.bounds[c][1]
                                if q_min[0] > new_max:
                                    overlaps = False
                                elif (q_min[0] == new_max) and (not q_min[1]):
                                    overlaps = False
                            if overlaps:
                                self.score += new_rgs
                                self.last_rg_queries.add(q)

            old_slps = [Ordering(None, None)]

            for _ in range(len(llist_objs)):
                best_slps = {}  # Best Ordering object for each signature
                for slp in old_slps:
                    for obj in slp.remainder:
                        new_ordering = Ordering(obj, slp)
                        if new_ordering.sig in best_slps:
                            if best_slps[new_ordering.sig].score > new_ordering.score:
                                best_slps[new_ordering.sig] = new_ordering
                            else:
                                del new_ordering
                        else:
                            best_slps[new_ordering.sig] = new_ordering
                old_slps = best_slps.values()

            assert len(old_slps) == (len(llist_objs)), "Something went wrong somewhere"

            best_score = max_score
            opt_order = list(llist_objs)
            for ordering in old_slps:
                if ordering.score < best_score:
                    opt_order = ordering.ordering

        # Check that the ordering is valid
        assert set(opt_order) == set(obj_dict.keys())
        assert len(opt_order) == len(obj_dict.keys())

        # Write the file and finish
        ordered_chunks = list([obj_dict[obj] for obj in opt_order])
        print(f"Ordered by file_gen_3a.  Row group size is {self.rg_size}")
        pq.write_table(Table.from_pandas(concat(ordered_chunks).reset_index(drop=True)), file_path, row_group_size=self.rg_size)

    @remove_index
    @rg_approx
    def file_gen_3b(self, file_path, obj_dict):
        """
        Generate a file
        ** This one approximates the best way of ordering leaves
        - This depends on every obj being smaller than rg_size
        - This will be asserted
        :param file_path: Folder for storing things in
        :param obj_dict: A dict mapping file names to pandas dataframes containing chunks of those files
        """
        # rg_count = -(-self.block_size // self.rg_size)
        q_dict = {}
        objs = list(obj_dict.keys())
        total_size = 0

        # Get the frequency with which queries are seen
        for obj in objs:
            obj_size = obj_dict[obj].shape[0]
            assert obj_size <= self.rg_size, f"Object {obj} is too large to be using this method"
            total_size += obj_size
            for q in self.table_q_dict[obj]:
                q_dict[q] = 1 + q_dict.get(q, 0)
        q_list = list(q_dict.keys())

        # Arrange queries by the sum of the inverse frequency of their queries
        objs.sort(key=lambda x: sum((1/q_dict[q] for q in self.table_q_dict[x])))

        # Take the first (rg_count - 1) of those queries as the rg bound objects
        rg_count = -(-total_size // self.rg_size)
        rg_bounds = objs[0:rg_count - 1]
        rg_bounds.sort(key=lambda x: obj_dict[x].shape[0])
        objs = objs[rg_count - 1:]

        # Now, initiate the ordering, and enter the for loop
        ordering = []
        current_rg_size = 0
        for i in range(rg_count):
            rg_objs = [] # Do not rely on this to be ordered
            if i > 0:
                rg_objs.append(rg_bounds[i - 1])
            if i < (rg_count - 1):
                rg_objs.append(rg_bounds[i])
            else:
                # Empty out everything for the last one
                for obj in objs:
                    ordering.append(obj)
                break
            end_size = obj_dict[rg_bounds[i]].shape[0]

            # Enter the while loop to fill this row group
            while current_rg_size < (self.rg_size - end_size):
                # Develop the row group
                objs.sort(key=lambda x: self.rank_match(rg_objs, x))
                obj_found = False
                obj = None
                j = None
                assert len(objs) > 0, "Something went wrong here"
                for j in range(len(objs)):
                    obj = objs[j]
                    if (obj_dict[obj].shape[0] + current_rg_size) <= self.rg_size:
                        obj_found = True
                        break
                if not obj_found:
                    # If the end piece is too small and nothing fits, switch it with the best object here
                    obj = objs[0]
                    objs.append(rg_bounds[i])
                    rg_bounds[i] = obj
                    _ = objs.pop(0)
                    break
                rg_objs.append(obj)
                ordering.append(obj)
                current_rg_size += obj_dict[obj].shape[0]
                _ = objs.pop(j)

            # Add the end piece
            ordering.append(rg_bounds[i])
            current_rg_size += obj_dict[rg_bounds[i]].shape[0]

            # Reset the rg_size
            current_rg_size -= self.rg_size

        # Check that the ordering is valid
        assert set(ordering) == set(obj_dict.keys())
        assert len(ordering) == len(obj_dict.keys())

        # Write the file and finish
        order = map(lambda x: obj_dict[x], ordering)
        for obj in ordering:
            if obj_dict[obj].shape[0] > 0:
                self.index[obj].append(file_path)
        print(f"Ordered by file_gen_3b.  Row group size is {self.rg_size}")
        pq.write_table(Table.from_pandas(concat(order).reset_index(drop=True)), file_path, row_group_size=self.rg_size)

    @remove_index
    @rg_approx
    def file_gen_3c(self, file_path, obj_dict):
        """
        Generate a file
        ** This one approximates the best way of ordering leaves for when the number of row groups is
        close to the number of files
        * All this tries to do is order them so that adjacent ones are similar
        :param file_path: Folder for storing things in
        :param obj_dict: A dict mapping file names to pandas dataframes containing chunks of those files
        """
        rg_count = -(-self.block_size // self.rg_size)
        q_dict = {}
        objs = list(obj_dict.keys())

        # Get the frequency with which queries are seen
        for obj in objs:
            for q in self.table_q_dict[obj]:
                q_dict[q] = 1 + q_dict.get(q, 0)
        q_list = list(q_dict.keys())

        # Invert the frequencies
        for q in q_list:
            q_dict[q] = 1 / q_dict[q]

        # Initialize the ordering
        ordering = []

        # Start with the best object
        obj = min(map(lambda x: sum(map(lambda y: q_dict[y], self.table_q_dict[x]))))
        ordering.append(obj)
        objs.remove(obj)

        # Keep adding objects until nothing is left
        while len(objs) > 0:
            obj = min(map(lambda x: self.rank_match([obj], x, q_dict=q_dict)))
            ordering.append(obj)
            objs.remove(obj)

        # Check that the ordering is valid
        assert set(ordering) == set(obj_dict.keys())
        assert len(ordering) == len(obj_dict.keys())

        # Make the file
        order = map(lambda x: obj_dict[x], ordering)
        for obj in ordering:
            if obj_dict[obj].shape[0] > 0:
                self.index[obj].append(file_path)
        print(f"Ordered by file_gen_3c.  Row group size is {self.rg_size}")
        pq.write_table(Table.from_pandas(concat(order).reset_index(drop=True)), file_path, row_group_size=self.rg_size)

    def file_gen_3d(self, file_path, obj_dict):
        """
        Generate a file
        **This one uses 3a for smaller obj_dict, and an approximation for larger ones
        :param file_path: Folder for storing things in
        :param obj_dict: A dict mapping file names to pandas dataframes containing chunks of those files
        """
        cutoff = 16
        file_count = len(obj_dict.keys())
        if file_count <= cutoff:
            # The number of files to arrange is small
            self.file_gen_3a(file_path, obj_dict)
        elif max(map(lambda x: x.shape[0], obj_dict.values())) > self.rg_size:
            # There exist files that are close to the row group size
            self.file_gen_3c(file_path, obj_dict)
        else:
            # The number of files to arrange is significantly larger than the number of row groups
            self.file_gen_3b(file_path, obj_dict)
