import sqlite3


class Column:
    """
    Represents a column in a table
    IMMUTABLE
    """
    def __init__(self, name, ctype):
        """
        :param name: name of the column;
        :param ctype: type of the column
        """
        self.name = name
        self.ctype = ctype
        self.numerical = ctype in ("REAL", "INTEGER")

    def __eq__(self, other):
        """
        :param other: a different column
        :return: whether they represent the same column
        """
        return self.name == other.name

    def __repr__(self):
        return "(" + self.name + " " + self.ctype + " " + str(self.numerical) + ")"


class Table:
    """
    Represents a table in a database
    IMMUTABLE
    """
    def __init__(self, tname, new=False, columns=None, dbname="test.db"):
        """
        :param tname: name of the table
        :param new: whether the table is new
        :param columns: dictionary of names of columns to their sqlite types, all strings
        :param dbname: name of the database (default test.db)
        """
        self.name = tname
        self.dbname = dbname
        con = sqlite3.connect(dbname)
        cur = con.cursor()
        self.columns = {}
        if new:
            create_query = "CREATE TABLE " + tname + "("
            assert columns, "If this is a new table, it must have columns listed"
            col_list = []
            for col in columns.keys():
                col_list.append(col + " " + columns[col])
                self.columns[col] = Column(col, columns[col])
            create_query += ", ".join(col_list)
            create_query += ") STRICT"
            cur.execute(create_query)
            con.commit()
            con.close()
        else:
            column_query = "PRAGMA table_info(" + tname + ")"
            output = cur.execute(column_query)
            for data in output.fetchall():
                self.columns[data[1]] = Column((data[1]), data[2])
            con.commit()
            con.close()

    def info(self):
        return str(self.columns)

    def get_column(self, column):
        return self.columns.get(column, None)



