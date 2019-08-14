'''
两数异或：相当于每一位相加不考虑进位
两数相与并左移一位：相当于求进位
python中负数会有问题
'''
class Solution:
    def Add(self, num1, num2):
        while num2 != 0:
            temp = num1 ^ num2
            num2 = (num1 & num2) << 1
            num1 = temp & 0xFFFFFFFF
            return num1 if num1 >> 31 == 0 else num1-4294967296