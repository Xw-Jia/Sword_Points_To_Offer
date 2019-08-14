'''
前面的数字大于后面的数字，称为一个逆序对，求总数
'''
class Solution:
    def InversePairs(self, data):
        count = 0
        copy = []
        for _ in data:
            copy.append(_)
        copy.sort()

        for i in range(len(copy)):
            count += data.index(copy[i])
            data.remove(copy[i])
        return count % 1000000007