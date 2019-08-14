'''
从外往里 螺旋形 顺时针打印矩阵
'''
class Solution:
    def printMatrix(self,matrix):
        return (
                matrix and list(matrix.pop(0)) +
                self.printMatrix(list(zip(*matrix))[::-1])
        )
#如果逆时针
def anti_clock_wise(self, matrix):
    if not matrix:
        return []
    clock_wise = list(zip(*(matrix[::-1])))
    a = list(clock_wise.pop(0))[::-1]
    b = self.anti_clock_wise(clock_wise)
    return a+b