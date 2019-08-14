'''
思路： n 和 n-1 按位与，最右边的1会变成0 经过几次运算变成0，就是有几个1
'''
class Solution:
    def NemberOf1(self, n):
        count = 0
        for _ in range(32):
            count += (n&1==1)
            n >>= 1
        return count

    #或者
    def Nember_of_1(self, num):
        count = 0
        if num < 0:
            n = n&0xffffffff
            while n!=0:
                count += 1
                n = (n-1)&n
        return count
