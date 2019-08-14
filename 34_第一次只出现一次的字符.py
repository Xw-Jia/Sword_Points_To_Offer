'''
字符串全部由字母组成，找到第一个只出现一次的字符的位置，没有返回-1
遍历两次，第一次用hash存放字符和出现的次数，第二次找到hash等于1的值
'''
class Solution:
    def FirstNotRepeatingChar(self, s):
        if s == None or len(s) <= 0:
            return -1
        alphabet = dict()
        lst = ''.join(s)
        for i in lst:
            if i not in alphabet.keys():
                alphabet[i] = 1
            alphabet[i] += 1
        for i in lst:
            if alphabet[i] == 1:
                return lst.index(i)