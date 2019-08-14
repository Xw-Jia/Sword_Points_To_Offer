'''
给定double类型的浮点数base和int类型的整数exponent。求出base的exponent次方

利用指数右移一位，实现的是除2操作
再&1判断是否是奇数，是奇数就再×base
'''
class Solution:
    def Power(self, base, exponent):
        def PowerUnsign(base, exponent):
            if exponent==0:
                return 1
            if exponent==1:
                return  base
            ans = PowerUnsign(base, exponent>>1)
            ans *= ans
            if exponent & 1 == 1:
                ans *= base
            return ans

        if exponent < 0:
            return 1.0/PowerUnsign(base, abs(exponent))
        else:
            return PowerUnsign(base, abs(exponent))