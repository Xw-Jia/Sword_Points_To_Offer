'''
str --> int 当str不符合数字要求时，返回0
数值为0或者字符串不是合法数值返回0
'''
class Solution:
    def StrToInt(self, s):
        flag = False
        if not s or len(s) < 1:
            return 0
        num = []
        numdict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8:'8,'9':9}
        for i in s:
            if i in numdict.keys():
                num.append(numdict[i])
            elif i=='+' or i=='-':
                continue
            else:
                return 0
        ans = 0
        if len(num)==1 and num[0]==0:
            flag = True
            return 0
        for i in num:
            ans = ans*10 + i
        if s[0] == '-':
            ans=0-ans
            return  ans