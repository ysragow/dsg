from qd.qd_predicate import Predicate, Operator
from qd.qd_column import Column
from qd.qd_table import Table
from pyarrow.compute import field, scalar
from pyarrow import timestamp, scalar as pa_scalar
from numpy import datetime64
import json


class Categorical(Predicate):
    # A categorical predicate, using = or IN
    def __init__(self, op, column, categories):
        """
        :param op: the operator this predicate is based on
        :param column: the column this predicate breaks on
        :param categories: the set of categories included in this predicate
        """
        super().__init__(op, column, list(categories))
        self.values = categories
        self.str_right = str(categories)
        assert (not column.numerical), "This column cannot be used for a categorical predicate"
        assert op.symbol in ('IN', '!IN'), "Wrong type of predicate"

    def __contains__(self, item):
        return item[self.column.num] in self.values

    def intersect(self, preds):
        """
        :param preds: a dictionary mapping column names to predicates
        :return: whether that set intersects with the items in our predicate
        """
        output = True
        for pred in preds[self.column.name]:
            output &= self.op(self.values, pred.values)
        return output

    def flip(self):
        """
        :return: predicate: the inverse of this predicate
        """
        return Categorical(self.op.flip(), self.column, self.values)

    def to_expression(self):
        """
        :return: A parquet expression representing this predicate
        """
        if self.op.symbol == 'IN':
            return field(self.column.name).isin(list(self.values))
        else:
            output = None
            for value in list(self.values):
                new_exp = (field(self.column.name) != scalar(value))
                if output is None:
                    output = new_exp
                else:
                    output &= new_exp
            return output


class Numerical(Predicate):
    # A numerical predicate, using <, >, =>, <=, or =
    def __init__(self, op, column, num):
        """
        :param op: the operator this predicate is based upon
        :param column: the column this predicate breaks on
        :param num: the number that we measure against the column value
        """
        super().__init__(op, column, num)
        self.str_right = str(num)
        self.num = num
        assert column.numerical, "This column cannot be used for a numerical predicate"
        assert op.symbol != 'IN', "Wrong type of predicate"

    def __contains__(self, item):
        """
        :param item: an item to be tested against this parameter
        :return: whether this item is in this predicate
        """
        try:
            return self.op(item[self.column.num], self.num)
        except Exception as ex:
            print(self.column, self.column.num, self.num)
            raise ex

    def to_dnf(self):
        """
        :return: this predicate as a DNF expression
        """
        return self.column.name, self.op.symbol, self.num

    def intersect(self, preds):
        """
        :param preds: a dictionary mapping columns to predicates
        :return: whether the predicates intersect
        """
        output = True
        for pred in preds[self.column.name]:
            num = pred.num
            op = pred.op
            output &= self.op(num, self.num) or op(self.num, num) or ((self.op == op) & (self.num == num))
        return output

    def flip(self, parent_pred=None):
        """
        :param parent_pred: extraneous argument, here for inheritance reasons
        :return: predicate: the inverse of this predicate
        """
        return Numerical(self.op.flip(), self.column, self.num)

    def to_expression(self):
        """
        :return: A parquet expression representing this predicate
        """
        if self.column.ctype != 'DATE':
            return self.op(field(self.column.name), scalar(self.num))
        else:
            print(self.num)
            num_str = str(self.num)
            if len(num_str) == 10:
                num_str += ' 00:00:00'
            return self.op(field(self.column.name), pa_scalar(datetime64(num_str), type=timestamp('s')))


class CatComparative(Predicate):
    # A comparative predicate between two columns.  Only used in queries, never in nodes
    def __init__(self, op, col1, col2):
        """
        :param op: the operator this predicate is based upon
        :param col1: the column this predicate breaks on
        :param col2: the column this predicate breaks on
        """
        super().__init__(op, col1, col2)
        self.str_right = str(col2)
        self.comparative = True
        self.col2 = col2
        assert col1.numerical == col2.numerical, "These columns cannot be compared"
        assert (op.symbol == '='), "Categorical comparison requires equality statement"
        assert (not col1.numerical), "These columns are not categorical"

    def __contains__(self, item):
        """
        :param item: an item to be tested against this parameter
        :return: whether this item is in this predicate
        """
        return item[self.column.num] == item[self.col2.num]

    def intersect(self, preds):
        """
        :param preds: a dictionary mapping column names to predicates
        :return: whether the predicates intersect
        """
        output = True
        preds1 = preds[self.column.name]
        preds2 = {self.col2.name: preds[self.col2.name]}
        for pred in preds1:
            output &= pred.intersect(preds2)
        return output


class NumComparative(Predicate):
    # A comparative predicate between two columns.  Only used in queries, never in nodes
    def __init__(self, op, col1, col2):
        """
        :param op: the code for the operation this predicate is based upon.  must be 5 or less
        :param col1: the column this predicate breaks on
        :param col2: the column this predicate breaks on
        """
        super().__init__(op, col1, col2)
        self.str_right = str(col2)
        self.col2 = col2
        self.comparative = True
        assert col1.numerical == col2.numerical, "These columns cannot be compared"
        assert col1.numerical, "Wrong type of comparative predicate"
        assert op.symbol != "IN", "This operation cannot be used to compare columns"

    def __contains__(self, item):
        """
        :param item: an item to be tested against this parameter
        :return: whether this item is in this predicate
        """
        try:
            return self.op(item[self.column.num], item[self.col2.num])
        except Exception as ex:
            print(item)
            print(self.column, self.column.num, self.col2, self.col2.num)
            raise ex

    def intersect(self, preds):
        """
        :param preds: a dictionary mapping column names to predicates
        :return: whether the predicates intersect
        """
        output = True
        plist = list(preds[self.column.name] + preds[self.col2.name])
        ops = [self.op]
        if self.op.symbol == '=':
            # = can be defined as <= and >=
            ops = [Operator('<='), Operator('>=')]
        for i in range(len(ops)):
            p1 = plist[i]
            p2 = plist[-i - 1]
            # Either one is strictly greater than the other, or they are equal and all operations allow equality
            output &= (self.op(p1.num, p2.num) and (p1.num != p2.num)) or all(
                (self.op(p1.num, p2.num), p1.op(p1.num, p2.num), p2.op(p1.num, p2.num)))
        return output

    def to_dnf(self):
        return self.column.name, self.op.symbol, self.col2.name


    def flip(self, parent_pred=None):
        return NumComparative(self.op.flip(), self.column, self.col2)

    def to_expression(self):
        """
        :return: A parquet expression representing this predicate
        """
        return self.op(field(self.column.name), field(self.col2.name))


def intersect(preds, debug=False):
    block = BigColumnBlock()
    for pred in preds:
        if not block.add(pred, false_if_fail_test=True):
            return False
    return True
    # date_preds = []
    # num_preds = []
    # cat_preds = []
    # for pred in preds:
    #     if pred.column.numerical:
    #         if pred.column.ctype == 'DATE':
    #             date_preds.append(pred)
    #         else:
    #             num_preds.append(pred)
    #     else:
    #         cat_preds.append(pred)
    # if debug:
    #     print("Categoricals:", cat_intersect(cat_preds, debug))
    #     print("Dates:", num_intersect(date_preds, debug))
    #     print("Numbers:", num_intersect(num_preds, debug))
    # return cat_intersect(cat_preds) & num_intersect(date_preds) & num_intersect(num_preds)


def cat_intersect(preds, debug=False):
    """
    Checks whether a list of categorical predicates can all be satisfied
    :param preds: A list of numeric predicates
    :param debug: Whether to print anything causing a falsehood
    :return: Whether there exists a point where they intersect
    """
    # No cat comparatives, so no need for an index
    # Instead we map column names to sets of possible outcomes
    col_sets = {}
    col_not_sets = {}
    for pred in preds:
        if pred.op.symbol == '!IN':
            if pred.column.name in col_not_sets:
                for value in pred.values:
                    col_not_sets[pred.column.name].add(value)
            else:
                col_not_sets[pred.column.name] = pred.values.copy()
        elif pred.op.symbol == 'IN':
            if pred.column.name in col_sets:
                current_values = list(col_sets[pred.column.name])
                for value in current_values:
                    if value not in pred.values:
                        col_sets[pred.column.name].remove(value)
            else:
                col_sets[pred.column.name] = pred.values.copy()
        else:
            raise Exception("Invalid Predicate: " + str(pred))
    for column in col_sets.keys():
        if column in col_not_sets:
            for value in col_not_sets[column]:
                if value in col_sets[column]:
                    col_sets[column].remove(value)
        if len(col_sets[column]) == 0:
            if debug:
                print(f"Column {column} cannot take on any value")
            return False
    return True


class BigColumnBlock:
    """
    Handles the three categories: dates, integers, and categories
    """
    def __init__(self, debug=False, always_false=False):
        self.always_false = False
        self.date_block = ColumnBlock(debug=debug)
        self.num_block = ColumnBlock(debug=debug)
        self.cat_preds = []

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'Dates:\n' + str(self.date_block) + '\n\nNumbers:\n' + str(self.num_block) + '\n\nCategories:\n' + str(self.cat_preds)

    def test(self, pred, debug=False):
        """
        Test whether adding a pred makes the block unsatisfiable
        :param pred: a pred
        :return: Whether all the preds in this block, plus the new bred, can be satisfied
        """
        if self.always_false:
            return False
        if pred.column.numerical:
            if pred.column.ctype == 'DATE':
                return self.date_block.test(pred, debug=debug)
            else:
                return self.num_block.test(pred, debug=debug)
        else:
            return cat_intersect(self.cat_preds + [pred])

    def add(self, pred, debug=False, false_if_fail_test=False):
        if self.always_false:
            return False
        if not self.test(pred):
            if false_if_fail_test:
                return False
            raise ValueError("You can only add a predicate that does not contradict the bounds")
        if pred.column.numerical:
            if pred.column.ctype == 'DATE':
                self.date_block.add(pred, debug=debug)
            else:
                self.num_block.add(pred, debug=debug)
        else:
            self.cat_preds.append(pred)
        if false_if_fail_test:
            return True

    def fork(self, new_pred):
        """
        Add a new pred and return a copy
        :param new_pred: a predicate
        :return: A new BigColumnBlock object containing all of this block's predicates, and the new pred
        """
        output = BigColumnBlock()
        output.add(new_pred)
        for pred in (self.cat_preds + self.date_block.preds + self.num_block.preds):
            output.add(pred)
        return output


def find_cycle(graph, debug=False):
    '''
    Find a cycle in a graph
    :param graph: A dictionary mapping a node names to names of nodes they lead to
    :return: A list of nodes that form a cycle, otherwise None
    '''
    all_nodes = list(graph.keys())

    # Perform DFS
    never_found_nodes = set(all_nodes)
    once_found_nodes = set()
    useless_nodes = set()
    while len(never_found_nodes) > 0:
        node = list(never_found_nodes)[0]
        cycle = cycle_dfs(node, graph, once_found_nodes, never_found_nodes, useless_nodes, debug=debug)
        if cycle is not None:
            cycle.reverse()
            return cycle[:-1]
    return None


def kosaraju(graph, root_key=True):
    """
    Use Kosaraju's algorithm to get the strongly connected components of the graph
    :param graph: A graph
    :param root_key: Whether the roots are the keys of the output, or if every node maps to a root
    :return: the strongly connected components of the graph
    """
    # Initialize terms for the functions
    output = {}
    visited = set()
    assigned = set()
    inv_graph = {}
    all_nodes = list(graph.keys())
    for node in all_nodes:
        inv_graph[node] = []
    L = []

    def visit(u):
        if u not in visited:
            visited.add(u)
            for v in graph[u]:
                visit(v)
            L.append(u)

    if root_key:
        def assign(u, root):
            if u not in assigned:
                assigned.add(u)
                output[root].add(u)
                for v in inv_graph[u]:
                    assign(v, root)
    else:
        def assign(u, root):
            if u not in assigned:
                assigned.add(u)
                output[u] = root
                for v in inv_graph[u]:
                    assign(v, root)

    for node in all_nodes:
        visit(node)
        for out_node in graph[node]:
            inv_graph[out_node].append(node)

    L.reverse()

    for node in L:
        if node not in assigned:
            if root_key:
                output[node] = set()
            assign(node, node)

    return output



def cycle_dfs(node, graph, once_found_nodes, never_found_nodes, useless_nodes, found_nodes=None, debug=False):
    """
    Perform one step of DFS for finding cycles
    :param node: A node in the graph that we are currently operating on
    :param graph: A graph
    :param never_found_nodes: The set of nodes that has never been seen by BFS
    :param once_found_nodes: The set of known that has been seen by BFS
    :param useless_nodes: The set of nodes known to not lead to cycles
    :param found_nodes: The set of nodes that have been found in this depth specifically
    :return: Empty list if dead end
    """
    if found_nodes is None:
        found_nodes = set()
    if node in found_nodes:
        return [node]
    elif node in useless_nodes:
        return None
    else:
        found_nodes.add(node)
        once_found_nodes.add(node)
        if node in never_found_nodes:
            never_found_nodes.remove(node)
        for new_node in graph[node]:
            cycle = cycle_dfs(new_node, graph, once_found_nodes, never_found_nodes, useless_nodes, found_nodes=found_nodes.copy(), debug=debug)
            if cycle is not None:
                if (cycle[0] == cycle[-1]) and (len(cycle) > 1):
                    return cycle
                else:
                    cycle.append(node)
                    return cycle
    useless_nodes.add(node)
    return None


class ColumnBlock:
    """
    A set of ColumnNodes.  Used for long-term storage of bounds
    """
    def __init__(self, debug=False):
        self.col_index = {}
        self.preds = []
        self.comps = [] # List of tuples corresponding to node comparisons.  For example (a, b, True) means a >= b
        self.graph, self.edge_map = self.make_graph(include_edge_map=True)

    def __str__(self):
        return '\n'.join([str(node) for node in self.get_cnodes()])

    def get_cnodes(self):
        # Get all column nodes that have not been eliminated
        cols = []
        found_cols = set()
        for col in self.col_index.keys():
            real_col = self.col_index[col].name
            if real_col not in found_cols:
                found_cols.add(real_col)
                cols.append(real_col)
        return list([self.col_index[col] for col in cols])

    def make_graph(self, pred=None, ascending=True, include_edge_map=False):
        """
        Make a graph out of the ColumnNodes in this block
        :param pred: An extra pred to add in.  For testing preds.
        :param ascending: If true, then a -> b implies a <= b.  If false, then a -> b implies a >= b.
        :return: A dictionary mapping nodes to the nodes they have an edge to, and an edge map mapping nodes to dictionaries for whether the edge includes equality
        """
        # Make graph without pred
        graph = {}
        edge_map = {}
        for cnode in self.get_cnodes():
            node_dict = cnode.greater if ascending else cnode.smaller
            node_map = {} # Edge map for this node
            node_list = []
            for col in node_dict.keys():
                name = self.col_index[col].name
                if name not in node_map:
                    node_map[name] = node_dict[col]
                    node_list.append(name)
                else:
                    node_map[name] &= node_dict[col]
            graph[cnode.name] = node_list
            edge_map[cnode.name] = node_map

        # Add in pred
        if pred is not None:
            assert pred.comparative, "What is a non-comparative predicate doing in my graph???"
            assert pred.op.symbol in ('>=', '<=', '<', '>'), "No equalities here.  Not dealing with that"
            edge_e = pred.op.symbol in ('>=', '<=')
            name1 = self.col_index[pred.column.name].name
            name2 = self.col_index[pred.col2.name].name
            if (pred.op.symbol in ('<=', '<')) == ascending:
                if name2 not in edge_map[name1]:
                    graph[name1].append(name2)
                    edge_map[name1][name2] = edge_e
            else:
                if name1 not in edge_map[name2]:
                    graph[name2].append(name1)
                    edge_map[name2][name1] = edge_e

        if include_edge_map:
            return graph, edge_map
        else:
            return graph

    def test(self, pred, debug=False):
        """
        Test whether a pred contradicts the rest of the block
        :param pred: a new pred
        :param debug: debug
        :return: Whether the new pred contradicts the block
        """
        edge_e = pred.op.symbol in ('>=', '<=')
        if pred.comparative:
            # Deal with unseen nodes
            unseen_1 = pred.column.name not in self.col_index
            unseen_2 = pred.col2.name not in self.col_index
            if unseen_1 and unseen_2:
                if pred.op.symbol in ('>', '>='):
                    return pred.op(pred.column.max, pred.col2.min)
                else:
                    return pred.op(pred.column.min, pred.col2.max)
            if unseen_1 or unseen_2:
                if unseen_1:
                    unseen = pred.column
                    seen = pred.col2
                else:
                    unseen = pred.col2
                    seen = pred.column
                node = self.col_index[seen.name]
                if unseen_1 ^ (pred.op.symbol in ('>', '>=')):
                    return node.test_min(unseen.min, edge_e, debug=debug)
                else:
                    return node.test_max(unseen.max, edge_e, debug=debug)

            # Check for the case where the nodes are the same
            if self.col_index[pred.column.name].name == self.col_index[pred.col2.name].name:
                return edge_e

            # See if the graph (plus the pred) has a cycle containing an < or > edge
            graph, edge_map = self.make_graph(pred=pred, include_edge_map=True)
            conn_map = kosaraju(graph, root_key=False)
            if debug:
                print("Connection Map with pred:", conn_map)
                print("Graph with pred:", graph)
                print("Edge map with pred:", edge_map)
            for node in graph.keys():
                for out_node in graph[node]:
                    if not edge_map[node][out_node]:
                        if conn_map[node] == conn_map[out_node]:
                            if debug:
                                print(f"Column {node} is constrained to be strictly less than column {out_node}, which is impossible.")
                            return False

            # Assert the new bounds enacted by this comparative pred
            if pred.op.symbol in ('>', '>='):
                greater_node = self.col_index[pred.column.name]
                smaller_node = self.col_index[pred.col2.name]
            else:
                smaller_node = self.col_index[pred.column.name]
                greater_node = self.col_index[pred.col2.name]
            if not greater_node.test_min(smaller_node.min, edge_e & smaller_node.min_e, debug=debug):
                return False
            if not smaller_node.test_max(greater_node.max, edge_e & greater_node.max_e, debug=debug):
                return False
            return True
        else:
            if pred.column.name not in self.col_index:
                return True
            root_node = self.col_index[pred.column.name]
            if pred.op.symbol in ('>', '>='):
                return root_node.test_min(pred.value, edge_e, debug=debug)
            else:
                return root_node.test_max(pred.value, edge_e, debug=debug)

    def add(self, pred, debug=False):
        """
        Add a new pred to the block
        :param pred: A new pred
        :param debug: whether to be debugging
        """
        self.preds.append(pred)
        cname = pred.column.name
        if cname not in self.col_index:
            self.col_index[cname] = ColumnNode(pred.column, self.col_index, debug=debug)
        cnode = self.col_index[cname]
        if pred.comparative:
            cname2 = pred.col2.name
            if cname2 not in self.col_index:
                self.col_index[cname2] = ColumnNode(pred.col2, self.col_index, debug=debug)
            cnode2 = self.col_index[cname2]
            if cnode.name == cnode2.name:
                return
            tup = None
            if pred.op.symbol == '=':
                cnode.combine(cnode2, debug=debug)
            elif pred.op.symbol == '>=':
                cnode.add_smaller_node(cnode2, True)
                cnode2.add_greater_node(cnode, True)
            elif pred.op.symbol == '<=':
                cnode.add_greater_node(cnode2, True)
                cnode2.add_smaller_node(cnode, True)
            elif pred.op.symbol == '>':
                # if cnode.name == cnode2.name:
                #     # This is an error.  Remove the pred.
                #     raise ValueError(f"The pred {self.preds.pop(-1)} cannot be added")
                cnode.add_smaller_node(cnode2, False)
                cnode2.add_greater_node(cnode, False)
            elif pred.op.symbol == '<':
                # if cnode.name == cnode2.name:
                #     # This is an error.  Remove the pred.
                #     raise ValueError(f"The pred {self.preds.pop(-1)} cannot be added")
                cnode.add_greater_node(cnode2, False)
                cnode2.add_smaller_node(cnode, False)
            self.graph = self.make_graph()
            graph = self.graph
            scc_map = kosaraju(graph, root_key=True)
            for root in scc_map.keys():
                if len(scc_map[root]) > 1:
                    remaining_node = self.col_index[root]
                    for node in scc_map[root]:
                        if node not in remaining_node.col_set:
                            remaining_node.combine(self.col_index[node])
            # while True:
            #     # Keep finding cycles and combining until no more cycles exist
            #     cycle = find_cycle(self.make_graph())
            #     if cycle is None:
            #         return
            #     prev_node = cycle[-1]
            #     remaining_node = self.col_index[prev_node]
            #     for node in cycle:
            #         assert self.col_index[prev_node].greater[node], "This pred cannot have been added.  The block is broken."
            #         remaining_node.combine(self.col_index[node])
        else:
            if pred.op.symbol == '=':
                cnode.set_min_all(pred.value, True)
                cnode.set_max_all(pred.value, True)
            elif pred.op.symbol == '>=':
                cnode.set_min_all(pred.value, True)
            elif pred.op.symbol == '<=':
                cnode.set_max_all(pred.value, True)
            elif pred.op.symbol == '>':
                cnode.set_min_all(pred.value, False)
            elif pred.op.symbol == '<':
                cnode.set_max_all(pred.value, False)


class ColumnNode:
    """
    Turns a column into a column node
    """
    def __init__(self, c, col_index, debug=False):
        """
        Take a column and turn it into a column node
        :param c: A column
        :param col_index: A dictionary mapping all column names to their column nodes
        :param debug: Whether to print the mins and max of the column
        NOTE: col_index IS SHARED AMONG ALL COLUMN NODES
        """
        self.col_index = col_index

        # Linked list attributes
        self.left = None
        self.right = None
        self.debug = debug

        # Graph node attributes
        # self.greater: a dictionary mapping columns greater than
        # or equal to this one to a True or False to whether it is
        # [ (True) or ( (False)
        # self.smaller: a dictionary mapping columns smaller than
        # or equal to this one to a True or False to whether it is
        # [ (True) or ( (False)
        # constrained to be at least as small as this one
        # self.columns: the columns constrained to be equal contained
        # within this node
        # self.col_set: the names of the columns in self.columns
        self.type = c.ctype in ('DATE', 'INTEGER')
        self.greater = {}
        self.smaller = {}
        self.max = c.max
        self.max_e = True
        self.min = c.min
        self.min_e = True
        if debug:
            print(f"Max for column {c.name}: {c.max}")
            print(f"Min for column {c.name}: {c.min}")
        self.columns = [c.name]
        self.name = c.name
        self.col_set = {c.name}

    def test_min(self, n, e, debug=False):
        local_e = self.max_e & e
        if self.max < n:
            if debug:
                print(f"Column {self.name} has max {self.max}, so it cannot be greater than {n}.")
            return False
        elif (self.max == n) and (not local_e):
            if debug:
                print(f"Column {self.name} has max {self.max}, so it cannot be greater than {n}.")
            return False
        for col in self.greater.keys():
            if not self.col_index[col].test_min(n, e & self.greater[col], debug=debug):
                return False
        return True

    def test_max(self, n, e, debug=False):
        local_e = self.min_e & e
        if self.min > n:
            if debug:
                print(f"Column {self.name} has min {self.min}, so it cannot be less than {n}.")
            return False
        elif (self.min == n) and (not local_e):
            if debug:
                print(f"Column {self.name} has min {self.min}, so it cannot be less than {n}.")
            return False
        for col in self.smaller.keys():
            if not self.col_index[col].test_max(n, e & self.smaller[col], debug=debug):
                return False
        return True

    def __str__(self):
        min_sym = '>=' if self.min_e else '>'
        max_sim = '<=' if self.max_e else '<'
        gne = []
        ge = []
        for c in self.greater.keys():
            if self.greater[c]:
                ge.append(c)
            else:
                gne.append(c)
        lne = []
        le = []
        for c in self.smaller.keys():
            if self.smaller[c]:
                le.append(c)
            else:
                lne.append(c)

        return f'''{'{'}{self.columns}, {max_sim} {self.max}, {min_sym} {self.min}, < {gne}, > {lne}, <= {ge}, >= {le}'''

    def combine(self, other, debug=False):
        # Merge two ColumnNodes.  Return False if it raises a contradiction, else return True
        self.type |= other.type

        # Set mins and maxes
        self.set_min_all(other.min, other.min_e)
        self.set_max_all(other.max, other.max_e)

        # Remove from linked list
        if other.left is not None:
            other.left.right = other.right
        if other.right is not None:
            other.right.left = other.left

        # Combine the column lists and change the column index
        for c in other.columns:
            self.columns.append(c)
            self.col_set.add(c)
            self.col_index[c] = self

        # Combine the greater than lists
        for c, e in other.greater.items():
            self.add_greater_node(self.col_index[c], e)

        # Combine the smaller than lists
        for c, e in other.smaller.items():
            self.add_smaller_node(self.col_index[c], e)

        # Prune instances of pointing to yourself
        for c in self.columns:
            if c in self.smaller:
                del self.smaller[c]
            if c in self.greater:
                del self.greater[c]

        # Delete the other nodes' attributes
        del other.greater
        del other.smaller
        del other.columns
        del other.col_set
        return

        # all_columns = set(sum([self.columns, other.columns] + [list(d.keys()) for d in (
        #     self.greater,
        #     other.greater,
        #     self.smaller,
        #     other.smaller
        # )], []))
        # for c in all_columns:
        #     if c in self.columns:
        #         if c in other.smaller:
        #             if not other.smaller[c]:
        #                 return False
        #             del other.smaller[c]
        #             del self.greater[other.name]
        #         if c in other.greater:
        #             if not other.greater[c]:
        #                 return False
        #             del other.greater[c]
        #             del self.smaller[other.name]
        #     elif c in other.columns:
        #         self.columns.append(c)
        #         self.col_set.add(c)
        #         self.col_index[c] = self
        #     elif c in other.greater:
        #         if c in self.greater:
        #             # Only still [ if both are [
        #             self.greater[c] &= other.greater[c]
        #             del other.greater[c]
        #             self.col_index[c].smaller[self.name] = self.smaller[c]
        #             del self.col_index[c].smaller[other.name]
        #         else:
        #             self.greater[c] = other.greater[c]
        #     elif c in other.smaller:
        #         if c in self.smaller:
        #             # Only still [ if both are [
        #             self.smaller[c] &= other.smaller[c]
        #             del other.smaller[c]
        #             self.col_index[c].greater[self.name] = self.smaller[c]
        #             del self.col_index[c].greater[other.name]
        #         else:
        #             self.smaller[c] = other.smaller[c]
        #
        # self.col_index[other.name] = self
        # return True

    # def set_min(self, n, e):
    #     if n >= self.min:
    #         self.min = n
    #         self.min_e = e & (self.min_e | (n > self.min))
    #
    # def set_max(self, n, e):
    #     if n <= self.max:
    #         self.max = n
    #         self.max_e = e & (self.max_e | (n < self.max))
    def set_min_all(self, n, e):
        changed = False
        if n > self.min:
            self.min = n
            self.min_e = e
            changed = True
        elif (n == self.min) and (self.min_e != (self.min_e & e)):
            self.min_e = self.min_e & e
            changed = True
        if changed:
            for c in self.greater.keys():
                self.col_index[c].set_min_all(n, e & self.greater[c])

    def set_min(self, n, e):
        # old_min = self.min
        # old_e = self.min_e
        # if n >= self.min:
        #     old_min = n
        #     old_e = e & (self.min_e | (n > self.min))

        if n > self.min:
            self.min = n
            self.min_e = e
        elif n == self.min:
            self.min_e &= e

        # assert old_e == self.min_e, "They are different"
        # assert old_min == self.min, "They are different"

    def set_max_all(self, n, e):
        changed = False
        if n < self.max:
            self.max = n
            self.max_e = e
            changed = True
        elif (n == self.max) and (self.max_e != (self.max_e & e)):
            self.max_e = self.max_e & e
            changed = True
        if changed:
            for c in self.smaller.keys():
                self.col_index[c].set_max_all(n, e & self.smaller[c])

    def set_max(self, n, e):
        # if n <= self.max:
        #     self.max = n
        #     self.max_e = e & (self.max_e | (n < self.max))

        if n < self.max:
            self.max = n
            self.max_e = e
        elif n == self.max:
            self.max_e &= e
        #
        # assert new_e == self.max_e, f"They are different in case: [max: {old_max}, old_e: {old_e}, n: {n}, e: {e}"
        # assert new_max == self.max, f"They are different in case: [max: {old_max}, old_e: {old_e}, n: {n}, e: {e}"

    def add_greater(self, c, e):
        self.greater[c] = e & self.greater.get(c, True)

    def add_greater_node(self, other, e):
        if self.name != other.name:
            self.greater[other.name] = e & self.greater.get(other.name, True)

    def add_smaller(self, c, e):
        self.smaller[c] = e & self.smaller.get(c, True)

    def add_smaller_node(self, other, e):
        if self.name != other.name:
            self.smaller[other.name] = e & self.smaller.get(other.name, True)

    def set_left(self, other):
        # Only call this when len(self.smaller.keys()) = 0
        self.left = other

    def set_right(self, other):
        # Only call this when len(self.smaller.keys()) = 0
        self.right = other

    def remove(self):
        # Only call this when len(self.smaller.keys()) = 0
        for c in self.greater.keys():
            node = self.col_index[c]
            e = self.greater[c]
            if node.min == self.min:
                node.set_min(self.min, (self.min_e & node.min_e & e))
            elif node.min < self.min:
                node.set_min(self.min, (self.min_e & e))
            # Remove every associated node
            for col in self.columns:
                if col in node.smaller:
                    del node.smaller[col]


def num_intersect(preds, debug=False):
    """
    Checks whether a list of numeric predicates can all be satisfied
    :param preds: A list of numeric predicates
    :param debug: Whether to be in debug mode (additional print statements)
    :return: Whether there exists a point where they intersect
    """
    # This index keeps track of which column names point to which node objects
    # This is subject to change, so every ColumnNode needs to have access to this
    index = {}

    # Hopefully the code should be good enough that the order of adding nodes here doesn't matter
    for pred in preds:
        if pred.column.name not in index:
            index[pred.column.name] = ColumnNode(pred.column, index, debug)
        if pred.comparative:
            if pred.col2.name not in index:
                index[pred.col2.name] = ColumnNode(pred.col2, index, debug)
            if pred.op.symbol == '=':
                if not index[pred.column.name].combine(index[pred.col2.name], debug):
                    return False
            else:
                e = pred.op.symbol in ('<=', '>=')
                is_greater = pred.op.symbol in ('>', '>=')
                greater = pred.column.name if is_greater else pred.col2.name
                smaller = pred.col2.name if is_greater else pred.column.name
                index[greater].add_smaller(smaller, e)
                index[smaller].add_greater(greater, e)
        else:
            # if debug:
            #     print("Hi hello we got here right")
            if pred.op.symbol == '=':
                index[pred.column.name].set_min(pred.num, True)
                index[pred.column.name].set_max(pred.num, True)
            else:
                e = pred.op.symbol in ('<=', '>=')
                if pred.op.symbol in ('>', '>='):
                    index[pred.column.name].set_min(pred.num, e)
                else:
                    index[pred.column.name].set_max(pred.num, e)

    # Iterate through everything
    zeroes_list_top = None
    active_set = set()
    for col_name in index.keys():
        active_set.add(index[col_name].name)
    for cname in active_set:
        if len(index[cname].smaller.keys()) == 0:
            index[cname].left = zeroes_list_top
            if zeroes_list_top is not None:
                zeroes_list_top.right = index[cname]
            zeroes_list_top = index[cname]
    while len(active_set) > 0:
        # for c in active_set:
        #     print("Active set:")
        #     i = index[c]
        #     print(i.name, i.smaller)
        if zeroes_list_top is not None:
            changed_set = set(index[c].name for c in zeroes_list_top.greater.keys())
            zeroes_list_top.remove()
            active_set.remove(zeroes_list_top.name)
            zeroes_list_top = zeroes_list_top.left
            if zeroes_list_top is not None:
                zeroes_list_top.right = None
            for cname in changed_set:
                if len(index[cname].smaller.keys()) == 0:
                    index[cname].left = zeroes_list_top
                    if zeroes_list_top is not None:
                        zeroes_list_top.right = index[cname]
                    zeroes_list_top = index[cname]
        elif len(active_set) > 0:
            # There's a cycle.  Find it
            cname = None
            found_set = set()
            for cname in active_set:
                break
            found_list = [cname]
            cname = index[cname].name
            while cname not in found_set:
                found_set.add(cname)
                col = index[cname]
                cname = None
                for cname in col.smaller.keys():
                    break
                cname = index[cname].name
                found_list.append(cname)

            # We now have a path leading to a cycle.  Extract the cycle
            cycle_begun = False
            all_e = True
            prev = None
            root = None
            cycle_nodes = []
            for cname in found_list:
                if cycle_begun:
                    current = index[cname]
                    for cname2 in current.columns:
                        if cname2 in prev.smaller:
                            all_e &= prev.smaller[cname2]
                    prev = current
                    if prev.name != root.name:
                        cycle_nodes.append(prev)
                else:
                    if cname == found_list[-1]:
                        cycle_begun = True
                        prev = index[cname]
                        root = index[cname]
            if not all_e:
                if debug:
                    print("A cycle was found, and it did not consist entirely of less than or equals")
                return False
            for node in cycle_nodes:
                root.combine(node, debug)
                active_set.remove(node.name)

            # Look for zeroes again
            for cname in active_set:
                if len(index[cname].smaller.keys()) == 0:
                    if zeroes_list_top is not None:
                        zeroes_list_top.right = index[cname]
                    index[cname].left = zeroes_list_top
                    zeroes_list_top = index[cname]

    # It's over.  Breathe.  Check if things match.
    for col in index.keys():
        c = index[col]
        if debug:
            print(f"{c.min} {'<=' if c.min_e else '<'} {col} {'<=' if c.max_e else '<'} {c.max}")
        if c.min > c.max:
            if debug:
                print(f"Column {col} is constrained to have a max of {c.max} and a min of {c.min}")
            return False
        if (c.min == c.max) and not (c.min_e and c.max_e):
            if debug:
                print(f"Column {col} is constrained to have a max of {c.max} and a min of {c.min}")
            return False
    return True


def pred_gen(pred_string, table):
    """
    :param pred_string: a string of the form "column_name operator value"
    :param table: an instance of the table class
    :return: a predicate based on the string
    """
    # print(pred_string)
    col_name, op_name, value_name = pred_string.split(" ", 2)
    column = table.get_column(col_name)
    assert column is not None, "The column " + col_name + " does not exist in the table at " + table.path
    op = Operator(op_name)
    if table.get_column(value_name):
        # Instance of a comparative predicate
        column2 = table.get_column(value_name)
        if column.numerical:
            return NumComparative(op, column, column2)
        else:
            return CatComparative(op, column, column2)
    elif value_name.replace('.', '', 1).isdigit() or (
            value_name[1:].replace('.', '', 1).isdigit() and value_name[0] == '-'):
        # Instance of a numerical (non-date) predicate
        num = int(value_name) if value_name.isdigit() else float(value_name)
        assert column.numerical, "This is not a numerical column, so it cannot be compared with a number"
        return Numerical(op, column, num)
    elif (list([len(s) for s in value_name[:10].split('-')]) == [4, 2, 2]) and value_name[:10].replace('-', '').isdigit():
        num = datetime64(value_name)
        assert column.numerical, "This is not a numerical column, so it cannot be compared with a number"
        return Numerical(op, column, num)
    elif ((value_name[0] == '(') and (value_name[-1] == ')')) or ((value_name[0] == '{') and (value_name[-1] == '}')):
        # Instance of a categorical predicate
        values_list = value_name[1:-1].replace(', ', ',').split(',')
        values = set()
        for v in values_list:
            if v[0] == v[-1] == "'":
                values.add(v[1:-1])
            else:
                values.add(v)
        return Categorical(op, column, values)
    elif (op.symbol == '=') and (value_name[0] == "'") and (value_name[-1] == "'"):
        values = set()
        values.add(value_name[1:-1])
        return Categorical(Operator('IN'), column, values)
    else:
        raise Exception("Something's wrong")
