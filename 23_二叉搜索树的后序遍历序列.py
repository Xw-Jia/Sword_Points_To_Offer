'''
输入一个整数数组，判断是不是某个二叉搜索树的后序遍历
思路： 根据后序遍历特点，尾元素一定是root，小于尾元素的值是左子树，大于尾元素的值是右子树
且序列前半部分小于尾元素，后半部分大于尾元素，将序列分为左子树和右子树，递归
'''
class Solution:
    def VerifySquenceOfBST(self, seq):
        if seq == []:
            return False
        length = len(seq)
        root = seq[-1]
        for i in range(length):
            if seq[i] > root:
                break
        for j in range(i, length):
            if seq[j] < root:
                return False

        left = True
        if i>0:
            left = self.VerifySquenceOfBST(seq[:i])
        right = True
        if j<length-1:
            right = self.VerifySquenceOfBST(seq[i:length-1])

        return left and right