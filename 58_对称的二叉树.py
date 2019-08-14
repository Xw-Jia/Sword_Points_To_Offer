'''
判断二叉树是不是对称的
'''
class Solution:
    def isSymmetrical(self, root):
        def symmetric(p1, p2):
            if p1 and p2:
                return(p1.val == p2.val and
                       symmetric(p1.left, p2.right) and
                       symmetric(p1.right, p2.left))
            else:
                return p1 is p2
        if not root:
            return True
        return symmetric(root.left, root.right)