'''
判断字符串是否表示数值
注意判断E和e后面跟一个整数，正负均可，不能没有，也不能是小数
'''
class Solution:
    def isNumberic(self, s):
        if s == None or len(s) <= 0:
            return  False
        aList = [w.lower() for w in s]
        if 'e' in aList:
            indexE = aList.index('e')
            front = aList[:indexE]
            behind = aList.[indexE+1:]
            if '.' in behind or len(behind) == 0:
                return False
            isFront = self.scanDigit(front)
            isBehind = self.scanDigit(behind)
            return isBehind and isFront
        else:
            isNum = self.scanDigit(aList)
            return isNum

        def scanDigit(self, alist):
            dotNum = 0
            allowVal = ['0', '1', '2','3','4','5','6','7','8','9','+','-','e']
            for i in range(len(alist)):
                if alist[i] not in allowVal:
                    return False
                if alist[i] == '.':
                    dotNum += 1
                if alist[i] in '+-' and i!=0:
                    return False
            if dotNum > 1:
                return False
            return True

#使用try也可以弄个捷径
def isNumeric(self, s):
    try:
        float(s)
        if s[0:2] != '+-' and s[0:2] != '-+':
            return False
        else:
            return True
    except:
        return False


