from pqd.split import PNode
from qd.qd_algorithms import index, table_gen
from qd.qd_query import Query
from numpy import argsort as np_argsort
from fastparquet import ParquetFile, write
from pyarrow import parquet as pq, Table
from pandas import concat
from json import dump
import os


def list_sig(l):
    k = list(l)
    k.sort()
    return tuple(k)


def factors(n):
    # Get all factors of a number n
    # Start by getting prime factorization
    primes = {} # one plus the number of times each prime appears in the prime factorization
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


class PFile:
    def __init__(self, file_list, size, is_overflow):
        """
        :param file_list: List of files
        :param size: Number of rows
        :param is_overflow: Whether this is an overflow file
        """
        self.file_list = file_list
        self.size = size
        self.made = False
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


class PQD:
    def __init__(self, root_path, table, workload, block_size, split_factor, row_group_size=1000000, dp_factor=100, verbose=False):
        """
        Initialize a PQD layout
        :param root_path: path to the root file, regardless of whether it exists
        :param a table object: used for indexing (necessary for resetting queries)
        :param workload: The workload upon which we will be creating this
        :param block_size: The block size (block_size <= # of rows per file <= 2 * block size)
        :param split_factor: How many ways to split the data
        :param row_group_size: Size of row groups that the data is made into
        :param dp_factor: Granularity of the dynamic programming by row count in file_gen_3
        :param verbose: whether to print stuff in initilization
        """

        # Save relevant stuff
        self.block_size = block_size
        split_path = root_path.split('/')
        self.name = '.'.join(split_path[-1].split('.')[:-1])
        self.path = '/'.join(split_path[:-1])
        self.split_factor = split_factor
        self.workload = workload

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
        all_objs = index(Query([], table), root_path, table, verbose=verbose)
        for obj in all_objs:
            self.index[obj] = []
            self.table_dict[obj] = table_gen(obj)
            self.table_q_dict[obj] = []

            # Don't bother dealing with files that are empty
            if self.table_dict[obj].size != 0:
                self.files_list.append(obj)

        # Index your workload on the tree
        self.indices = {}  # map of query ids to their files
        self.query_ids = {}  # maps query ids to queries
        id = 0
        for q in workload.queries:
            objs = index(q, root_path, table)
            self.indices[id] = objs
            self.query_ids[id] = q
            for obj in objs:
                if obj not in self.table_q_dict:
                    print("A contradiction has been found")
                    print(f"Query: {q}")
                    print(f"File: {obj}")
                self.table_q_dict[obj].append(id)
            id += 1

        # Assert that every leaf is queried
        for obj in all_objs:
            assert len(self.table_q_dict[obj]) > 0, f"The object {obj} is not queried"

        # Setup for all make_layout
        self.abstract_block_size = block_size * split_factor
        self.eff_size_dict = {}  # maps file names to their effective sizes (the size mod twice the abs blk size)
        for file in self.files_list:
            self.eff_size_dict[file] = self.table_dict[file].size % (2 * self.abstract_block_size)

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

    def make_layout_1(self, take_top=True):
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

    def make_layout_1a(self):
        self.make_layout_1(take_top=False)

    def make_files(self, folder_path, make_alg, verbose=False):
        """
        Generate files based on the current self.layout and make_alg
        :param folder_path: Folder for storing things in
        :param make_alg: which file_gen algorithm to use
        :param verbose: how much to print
        """
        # Make the folders, if they don't exist
        file_template = folder_path + "/" + "{}/" + self.name + "{}.parquet"
        folder_template = folder_path + "/" + "{}"
        for i in range(self.split_factor):
            folder = folder_template.format(i)
            if not os.path.exists(folder):
                os.makedirs(folder)

        # Begin the for loop.  Load objects into memory.
        file_num = 0
        for pfile in self.layout:
            # Initialize variables
            eff_dframes = {}  # Maps n to a dict mapping obj to the slice of the dframes[obj] going into f_path/n/pfile
            for i in range(self.split_factor):
                eff_dframes[i] = {}

            # Loop through all the objects in the pfile
            for obj in pfile.file_list:
                if verbose:
                    print(f"Loading dataframe for file {obj}...", end='\r')

                df = ParquetFile(obj).to_pandas()
                if verbose:
                    print(f"Loaded dataframe for file {obj} with {df.shape[0]} rows")
                df_index = 0  # Keep track of our index in the dataframe

                # Account for overflow
                for _ in range(self.table_dict[obj].size // (2 * self.abstract_block_size)):
                    for i in range(self.split_factor):
                        file_chunk = df[df_index:df_index + (2 * self.block_size)]
                        chunk_file_name = file_template.format(i, file_num)
                        if verbose:
                            n_rows = file_chunk.shape[0]
                            print(f"Making file {chunk_file_name} from file {obj} with {n_rows} rows and {-(-n_rows//self.rg_size)} row groups.")
                        make_alg(chunk_file_name, {obj: file_chunk})
                        df_index += 2 * self.block_size
                    file_num += 1

                # Add to effective dframes
                eff_size = self.eff_size_dict[obj]
                if eff_size == 0:
                    continue
                ind_size = eff_size // self.split_factor  # size of each chunk
                cutoff = eff_size % self.split_factor  # cutoff for when to stop adding 1 to size
                for i in range(self.split_factor):
                    split_size = ind_size + (1 if i < cutoff else 0)
                    eff_dframes[i][obj] = df[df_index: df_index + split_size]
                    df_index += split_size

            # Create each file
            for i in range(self.split_factor):
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
            file_num += 1

        # Save the index
        with open(folder_path + "/index.json", "w") as file:
            dump(self.index, file)

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
    def file_gen_2(self, file_path, obj_dict):
        """
        Generate a file
        **This one organizes row groups only according file size and which queries access what**
        :param file_path: Folder for storing things in
        :param obj_dict: A dict mapping file names to pandas dataframes containing chunks of those files
        """
        pass

    @remove_index
    def file_gen_3(self, file_path, obj_dict):
        """
        Generate a file
        **This one adds another column for ease of row group skipping**
        :param file_path: Folder for storing things in
        :param obj_dict: A dict mapping file names to pandas dataframes containing chunks of those files

        """
        pass



        # repeated knn clustering!!!

        changed = False

        while not changed:
            pass

        # TODO: Finish this?

    @remove_index
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

            # q_dict = dict([(key, set(self.table_q_dict[key])) for key in self.table_q_dict.keys()])
            q_dict = self.table_q_dict
            rg_size = self.rg_size

            class Ordering:
                def __init__(self, next, prev=None):
                    if prev is None:
                        self.ordering = []
                        self.remainder = set(llist_objs)
                        self.sig = ()
                        self.last = None
                        self.score = 0
                        self.size = 0
                        self.last_rg_queries = set()
                        return
                    self.last = next

                    # Initialize the ordering and signature
                    self.ordering = prev.ordering.copy()
                    self.ordering.append(self.last)
                    self.sig = (list_sig(self.ordering), self.last)

                    # Initialize the remainder
                    self.remainder = prev.remainder.copy()
                    self.remainder.remove(self.last)

                    # Initialize the size
                    self.size = prev.size + obj_dict[next].shape[0]

                    # Initialize the score
                    self.score = prev.score
                    next_qs = q_dict[self.last]
                    self.last_rg_queries = set()

                    # Account for score accrued through new queries in the current row group
                    if prev.size % rg_size != 0:
                        self.last_rg_queries = prev.last_rg_queries.copy()
                        for q in next_qs:
                            if q not in self.last_rg_queries:
                                self.last_rg_queries.add(q)
                                self.score += 1

                    # Account for score accrued through making new row groups
                    new_rgs = (self.size // rg_size) - (prev.size // rg_size)
                    if new_rgs > 0:
                        self.score += new_rgs * len(next_qs)
                        self.last_rg_queries = set(next_qs)

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

        ordered_chunks = list([obj_dict[obj] for obj in opt_order])
        print("Row group size: " + str(self.rg_size))
        pq.write_table(Table.from_pandas(concat(ordered_chunks).reset_index(drop=True)), file_path, row_group_size=self.rg_size)

    @remove_index
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

        # Write the file and finish
        order = map(lambda x: obj_dict[obj], ordering)
        for obj in ordering:
            if obj_dict[obj].shape[0] > 0:
                self.index[obj].append(file_path)
        print("Ordering:", ordering)
        pq.write_table(Table.from_pandas(concat(order).reset_index(drop=True)), file_path, row_group_size=self.rg_size)

    @remove_index
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

        # Make the file
        order = map(lambda x: obj_dict[obj], ordering)
        for obj in ordering:
            if obj_dict[obj].shape[0] > 0:
                self.index[obj].append(file_path)
        print("Ordering:", ordering)
        pq.write_table(Table.from_pandas(concat(order).reset_index(drop=True)), file_path, row_group_size=self.rg_size)

    @remove_index
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
