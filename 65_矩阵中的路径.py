'''
从矩阵中能否找到包含某字符串的所有字符的路径，路径可以从矩阵中任意各自开启
每一步可以选择四个方向，但是不能重复选择各自
'''
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        for i in range(rows):
            for j in range(cols):
                if matrix[i*cols+j] == path[0] and self.find(list(matrix), rows, cols, path[1:],i,j):
                    return True
                return False
    def find(self, matrix, rows, cols, path, i,j):
        if not path:
            return True
        matrix[i*cols+j] = '0'  #使用的格子置零，防止重复使用
        if j+1 < cols and matrix[i*cols+j+1] == path[0]:
            return self.find(matrix, rows, cols. path[1:], i, j+1)
        elif j-1>=0 and matrix[i*cols +j-1] == path[0]:
            return self.find(matrix, rows, cols, path[1:], i, j-1)
        elif i+1<rows and matrix[(i+1)*cols+j] == path[0]:
            return self.find(matrix, rows, cols, path[1:], i+1, j)
        elif i-1>=0 and matrix[(i-1)*cols+j]==path[0]:
            return self.find(matrix, rows, cols, path[1:], i-1, j)
        else:
            return False
