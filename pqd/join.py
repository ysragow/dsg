from join import PNode
from qd.qd_algorithms import index, table_gen
from numpy import argsort as np_argsort


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
        self.rep[next_obj] = (prev_obj, self.rep[next_obj][1])
        if prev_obj is None:
            self.first = next_obj
        self.rep[prev_obj] = (self.rep[next_obj][0], next_obj)
        if next_obj is None:
            self.last = prev_obj
        self.size -= 1
        del self.rep[obj]

    def pop(self):
        # Remove and return the top object
        output = self.first
        self.remove(self.first)
        return output


def argsort(obj_list, f):
    """
    Argsort a function and output a linked list
    :param obj_list: A list of hashable objects
    :param f: a function
    :return: The list of objects, sorted by the function
    """
    return list([obj_list[i] for i in np.argsort([f(obj) for obj in obj_list])])


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
        self.files_func = None  # Indicates which make_files function was used to make the files
        self.layout_func = None  # Indicates which make_layout function was used to make the files

        # Index your workload on the tree
        self.indices = {}  # map of query ids to their files
        self.table_dict = {}  # dict mapping file names to corresponding table objects
        self.table_q_dict = {}  # dict mapping file names to list of query ids in the workload which access it
        self.files_list = []
        self.query_ids = {}  # maps query ids to queries
        id = 0
        for q in workload.queries:
            files = index(q, root_path, table)
            self.indixes[id] = files
            self.query_ids[id] = q
            for file in files:
                if file not in table_dict:
                    self.table_dict[file] = table_gen(file)
                    self.table_q_dict[file] = []
                    self.files_list.append(file)  # todo: THIS MEANS FILES NOT ACCESSED BY THE WORKLOAD ARE LEFT OUT
                    # But that probably won't happen because all files generated must have at least one query?
                    # So it won't happen if we use the same workload...
                self.table_q_dict[file].append(id)
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

    def make_layout_1(self):
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

        # Make the linked list
        size_llist = LList(argsort(self.files_list, lambda x: self.eff_size_dict[x] * len(self.table_q_dict[x])))

        # Enter the while loop, which does not cease until everything is assigned
        while len(size_llist) > 0:
            # Initialize variables, including first obj
            obj = size_llist.pop()
            current_file = [obj]  # the files that will be in this set
            total_size = self.eff_size_dict[obj]
            queries_accessed = set(self.table_q_dict[obj])  # contains the ids of the queries that access this file

            # Initialize npq_dict: the dict mapping objs to a set of their queries not present in queries_accessed
            npq_dict = {}
            for obj_2 in size_llist:
                npq_dict[obj_2] = set()
                for q in self.table_q_dict[obj_2]:
                    if q not in queries_accessed:
                        npq_dict[obj_2].add(q)

            # If obj is too big, it needs to have some files to itself
            for i in range(self.table_dict[obj].size // (2 * self.abstract_block_size)):
                layout.append([obj])

            # Add more objects
            while total_size <= self.abstract_block_size:
                # Get the new object and remove it from size_llist
                active_files = list([obj_2 for obj_2 in size_llist])
                obj = argsort(active_files, lambda x: (len(npq_dict[x]) / len(self.table_q_dict[x])))[0]
                size_llist.remove(obj)

                # If obj is too big, it needs to have some files to itself
                for i in range(self.table_dict[obj].size // (2 * self.abstract_block_size)):
                    layout.append([obj])

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

                # If obj is too big, it needs to have some files to itself
                for i in range(self.table_dict[obj].size // (2 * self.abstract_block_size)):
                    layout.append([obj])

            # Finally, add the current file to the layout
            layout.append(current_file)

        # Assign the layout to self.layout
        self.layout = layout




    def make_files_1(self, folder_path):
        """
        Generate files based on the current self.layout
        **This one does not pay attention to row groups**
        :param folder_path: Folder for storing things in
        """
        pass

    def make_files_2(self, folder_path):
        """
        Generate files based on the current self.layout
        **This one organizes row groups only according to file size**
        :param folder_path: Folder for storing things in
        """
        pass

    def make_files_3(self, folder_path):
        """
        Generate files based on the current self.layout
        **This one adds another column for ease of row group skipping**
        :param folder_path: Folder for storing things in
        """
        pass








