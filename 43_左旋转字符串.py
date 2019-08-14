'''
就是循环左移
'''
class Solution:
    def LeftRotateString(self, s, n):
        if not s:
            return  ''
        n = n % len(s)
        return s[n:] + s[:n]