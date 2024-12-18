from pqd.split import PNode
from qd.qd_algorithms import index, table_gen
from qd.qd_query import Query
from numpy import argsort as np_argsort
from fastparquet import ParquetFile, write
from pyarrow import parquet as pq
from pandas import concat
from json import dump
import os


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
    def __init__(self, root_path, table, workload, block_size, split_factor):
        """
        Initialize a PQD layout
        :param root_path: path to the root file, regardless of whether it exists
        :param a table object: used for indexing (necessary for resetting queries)
        :param workload: The workload upon which we will be creating this
        :param block_size: The block size (block_size <= # of rows per file <= 2 * block size)
        :param split_factor: How many ways to split the data
        """

        # Save relevant stuff
        self.block_size = block_size
        split_path = root_path.split('/')
        self.name = '.'.join(split_path[-1].split('.')[:-1])
        self.path = '/'.join(split_path[:-1])
        self.split_factor = split_factor
        self.workload = workload
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
        all_objs = index(Query([], table), root_path, table)
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
                self.table_q_dict[obj].append(id)
            id += 1

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
                            print(f"Making file {chunk_file_name} from file {obj} with {file_chunk.shape[0]} rows")
                        make_alg(chunk_file_name, {obj: file_chunk})
                        df_index += 2 * self.block_size
                    file_num += 1

                # Add to effective dframes
                eff_size = self.eff_size_dict[obj]
                ind_size = eff_size // self.split_factor  # size of each chunk
                cutoff = eff_size % self.split_factor  # cutoff for when to stop adding 1 to size
                for i in range(self.split_factor):
                    split_size = ind_size + (1 if i < cutoff else 0)
                    eff_dframes[i][obj] = df[df_index: df_index + split_size]
                    df_index += split_size

            # Create each file
            for i in range(self.split_factor):
                if verbose:
                    p_temp = "{} rows from file {}"
                    p_str = ", ".join([p_temp.format(eff_dframes[i][obj3].shape[0], obj3) for obj3 in eff_dframes[i].keys()])
                    print(f"Making file {file_template.format(i, file_num)} with {p_str}")
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

        write(file_path, concat(obj_dict.values()).reset_index(drop=True))
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

    # @remove_index
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








