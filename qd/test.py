from qd_data import Table

t = Table('q', True, {'id': 'INTEGER', 'num': 'REAL'})
print(t.info())
s = Table('q')
print(s.info())

