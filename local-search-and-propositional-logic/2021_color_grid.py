from clause import *

"""
For the color grid problem, the only code you have to do is in this file.

You should replace

# your code here

by a code generating a list of clauses modeling the grid color problem
for the input file.

You should build clauses using the Clause class defined in clause.py

Read the comment on top of clause.py to see how this works.
"""


def get_expression(size, points=None):
    expression = []
    if points is not None : 
        for point in points : 
            clause = Clause(size)
            clause.add_positive(point[0], point[1],point[2])
            expression.append(clause)
    
    for i in range(size): 
        for j in range(size): 
            clause = Clause(size)
            for k in range(size): 
                clause.add_positive(i,j,k)
                for alpha in range(size): 
                    if alpha != i : 
                        clause_col = Clause(size)
                        clause_col.add_negative(i,j,k)
                        clause_col.add_negative(alpha,j,k)
                        expression.append(clause_col)
                    if alpha != j : 
                        clause_row = Clause(size)
                        clause_row.add_negative(i,j,k)
                        clause_row.add_negative(i,alpha,k)
                        expression.append(clause_row)
                    if alpha != 0 and 0 <= i + alpha < size and 0 <= j + alpha < size:
                        clause_diff1 = Clause(size)
                        clause_diff1.add_negative(i, j, k)
                        clause_diff1.add_negative(i+alpha, j+alpha, k)
                        expression.append(clause_diff1)
                    if alpha != 0 and 0 <= i - alpha < size and 0 <= j - alpha < size:
                        clause_diff2 = Clause(size)
                        clause_diff2.add_negative(i, j, k)
                        clause_diff2.add_negative(i-alpha, j-alpha, k)
                        expression.append(clause_diff2)
                    if alpha != 0 and 0 <= i+alpha < size and 0 <= j-alpha < size:
                        clause_sum1 = Clause(size)
                        clause_sum1.add_negative(i, j, k)
                        clause_sum1.add_negative(i+alpha, j-alpha, k)
                        expression.append(clause_sum1)
                    if alpha != 0 and 0 <= i-alpha < size and 0 <= j+alpha < size:
                        clause_sum2 = Clause(size)
                        clause_sum2.add_negative(i, j, k)
                        clause_sum2.add_negative(i-alpha, j+alpha, k)
                        expression.append(clause_sum2)

            expression.append(clause)
    return expression


if __name__ == '__main__':
    expression = get_expression(3)
    for clause in expression:
        print(clause)
