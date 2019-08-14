'''
判断B是不是A的子结构（空树不是任何树的子结构）
思路：在A中查找B根节点一致的值，然后判断A中以该节点为根的子树，是不是和B有相同的结构
递归
'''
class Solution:
    def HasSubTree(self, pRoot1, pRoot2):
        result = False
        if pRoot1!=None and pRoot2!=None:
            if pRoot1.val == pRoot2.val:
                result = self.DoesTree1haveTree2(pRoot1, pRoot2)
            if not result:
                result = self.HasSubTree(pRoot1.left, pRoot2)
            if not result:
                result = self.HasSubTree(pRoot1.right, pRoot2)
        return  result

    def DoesTree1haveTree2(self, pRoot1, pRoot2):
        if pRoot2==None:
            return True
        if pRoot1==None:
            return False
        if pRoot1.val != pRoot2.val:
            return False
        return self.DoesTree1haveTree2(pRoot1.left, pRoot2.left) and self.DoesTree1haveTree2(pRoot1.right, pRoot2.right)

#写的太模糊了，递归看不懂
#重写一个
class Solution:
    def HasSubTree(self, s, t):
        def is_same(s, t):
            if s and t:     #根节点都不为空
                equal = (s.val==t.val)  #bool
                if not t.left and not t.right:  # t左右子树都为空，就只是一个点，判断equal即可，否则，判断root相等且左右相等
                    return equal
                else:
                    return (equal and is_same(s.left, t.left) and is_same(s.right, t.right))
            else:
                return s is t   #bool

        stack = s and [s]
        while stack:
            node = stack.pop()
            if node:
                res = is_same(node, t)  #判断s的节点子树与t是否相等
                if res:
                    return True
                stack.append(node.right)
                stack.append(node.left)
        return False

#再有，可以取巧，将Tree换成str
class Solution:
    def HasSub(self, pRoot1, pRoot2):
        def convert(p):
            if p:
                return str(p.val)+convert(p.left)+convert(p.right)
            else:
                return ''
        return convert(pRoot2) in convert(pRoot1) if pRoot2 else False
#这个可以有更简单的递归
class Solution:
    def HasSub(self, pRoot1, pRoot2):
        if pRoot1 and pRoot2:
            if pRoot1.val == pRoot2.val:
                return self.HasSub(pRoot1.left, pRoot2.left) and self.HasSub(pRoot1.right, pRoot2.right)
            else:
                return self.HasSub(pRoot1.left, pRoot2) or self.HasSub(pRoot1.right, pRoot2)
        if not pRoot1 or not pRoot2:
            return False
        return  True