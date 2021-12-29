import random

rows = 10
cols = 6

def getNeighbours(coord:(int,int) ):
    proposed = [ (coord[0]-1,coord[1]),
        (coord[0],coord[1]-1),
        (coord[0]+1,coord[1]),
        (coord[0],coord[1]+1)]
    r = [c for c in proposed if c[0]>=0 and c[1]>=0 and c[0]<rows and c[1] < cols ]
    return r

def startSolutions(matrix:[[int]], start:(int,int)):
    r = []
    neighbours = getNeighbours(start)
    for second in neighbours:
        lasts = getNeighbours(second)
        for last in lasts:
            if (last == start):
                continue;
            if (matrix[start[0]][start[1]] + matrix[second[0]][second[1]] == matrix[last[0]][last[1]]):
                r.append( [start, second, last])
    return r


def getSolutions(matrix:[[int]]):
    result = []
    for row in range(0,rows):
        for col in range(0,cols):
            s = startSolutions(matrix, (row,col))
            result.extend(s)
    return result

Matrix = [[random.randint(1, 9) for x in range(6)] for y in range(10)] 
Matrix[0][2] = Matrix[0][0]+Matrix[0][1]
Matrix[1][1] = Matrix[0][0]+Matrix[0][1]
print(Matrix)
print(startSolutions(Matrix, (0,0)))


print(getSolutions(Matrix))