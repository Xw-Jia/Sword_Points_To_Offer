'''
输入正整数数组，打印能拼出来的数字中最小的一个

'''
#使用冒泡排序
class Solution:
    def PrintMinNumber(self, numbers):
        if numbers==None or len(numbers)<=0:
            return ''
        strNum = [str(m) for m in numbers]
        for i in range(len(numbers)-1):
            for j in range(i+1, len(numbers)):
                if strNum[i]+strNum[j] > strNum[j]+strNum[i]:
                    strNum[i], strNum[j] = strNum[j], strNum[i]
        return int(''.join(strNum))