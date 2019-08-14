'''
从右上角开始查找
如果当前数 > target，那就左移一位
如果当前数 < target，那就下移一位
'''
class Solution:
    def Find(self, target,array):
        if array == []:
            return False
        num_row = len(array)
        num_col = len(array[0])

        i = 0
        j = num_col -1

        while i<=num_row-1 and j>=0:
            if array[i][j]>target:
                j -= 1
            elif array[i][j]<target:
                i += 1
            else:
                return True
        return False