'''
找出字符流中第一个只出现一次的字符
不存在只出现一次的字符时，返回#

引入存储空间，一个dict存储当前字符和出现的次数，一个list存储当前出现的字符，
每次比较list的第一个字符再dict中对应的次数
'''
class Solution:
    def __init__(self):
        self.adict = {}
        self.alist = []
    def FirstAppearingOnce(self):
        while len(self.alist) > 0 and self.adict[self.alist[0]] == 2:
            self.alist.pop(0)
        if len(self.alist) == 0:
            return '#'
        else:
            return self.alist[0]
    def Insert(self, char):
        if char not in self.adict.keys():
            self.adict[char] = 1
            self.alist.append(char)
        elif self.adict[char]:
            self.adict[char] = 2