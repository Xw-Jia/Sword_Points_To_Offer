'''
一个整型数组中除了两个数字以外，其他数字都出现了偶数次 找出这两个只出现一次的数字
最简单的hashmap，但是空间复杂度太高
考虑：把这两个元素分到两个组，由于两数不等，所以异或结果不为0，按异或结果二进制中1的所在位，
可以把他们分到两个子数组。子数组的异或结果就是这两个数
'''
class Solution:
    def FindNumsAppearOnce(self, array):
        if array == None:
            return []
        xor = 0
        for i in array:
            xor ^= i
        idxOf1 = self.getFirstIdx(xor)
        num1 = num2 = 0
        for j in range(len(array)):
            if self.IsBit(array[j], idxOf1):
                num1 ^= array[j]
            else:
                num2 ^= array[j]
        return [num1, num2]

    def getFirstIdx(self, num):
        idx = 0
        while num & 1 == 0 and idx <= 32:
            idx += 1
            num = num >> 1
            return idx

    def IsBit(self, num, indexBit):
        num = num >> indexBit
        return num & 1