'''
输入一个字符串，按英文字母顺序打印所有可能的排列
递归，不断固定第一个，求之后的可能排列
'''
class Solution:
    def Pernutation(self,ss):
        if not ss:
            return []
        return self.permute(ss)
    def permute(self,ss):
        return sorted(list(set(
            [h+p
             for i,h in enumerate(ss)
             for p in self.permute(ss[:i]+ss[i+1:])]
        ))) or ['']
#使用迭代
def Permutation(ss):
    ans = ['']
    for s in ss:
        ans = [p[:i] + s + p[i:]
               for p in ans for i in range((p+s).index(s)+1)]
    return sorted(ans) if ss else []
#循环可以展开写
class Solution:
    def Permutation(self, ss):
        if not ss:
            return []
        ret = []
        for i in range(ss):
            for j in self.Permutation(ss[:i]+ss[i+1:]):
                ret.append(ss[i]+j)
        return sorted(list(set(ret)))
