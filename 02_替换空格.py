'''
再开一个list，遇到空格就append['%20']，否则append本身
'''
class Solution:
    def replaceSpace(self,s):
        return ''.join(c if c!=' ' else '%20' for c in s)

    #也可以这样写replace
        #return s.replace(' ','%20')