'''
第一行从左到右，第二行从右到左...
'''
class Solution:
    def Print(self, root):
        ans, level, order = [], root and [root], 1
        while level:
            ans.append([n.val for n in level][::order])
            order *= -1
            level = [kid for n in level for kid in (n.left, n.right) if kid]
        return ans
