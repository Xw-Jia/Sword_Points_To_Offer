'''
输入根节点和一个整数，打印出二叉树中节点值的和为输入整数的所有路径
思路： 用前序遍历访问二叉树，当访问道节点时，加入路径中，并累加节点值，直到访问到符合要求的节点或者访问到叶节点，
然后，递归访问该节点的父节点，函数退出时删除当前节点，并减去当前系欸但那的值，相当于出栈入栈过程
'''
#迭代
class Solution:
    def FindPath(self, root, total):
        stack = root and [(root, [root.val], total)]
        ans = []
        while stack:
            n, v, t = stack.pop()
            if not n.left and not n.right and n.val==t: #若左右子树都是空且root.val=total，则[root.val]是一个路径
                ans.append(v)
            if n.right:
                stack.append((n.right, v+[n.right.val], t-n.val))
            if n.left:
                stack.append((n.left, v+[n.left.val], t-n.val))
        return ans
#递归
#先找出所有路径，再过滤
class Solution:
    def FindPath(self, root, sum_val):
        paths = self.all_paths(root)
        return [path for path in paths if sum(path)==sum_val]
    def all_paths(self, root):
        if not root:
            return []
        return [
            [root.val]+path
            for kid in (root.left, root.right) if kid
            for path in self.all_paths(kid)
        ] or [[root.val]]

#递归
class Solution:
    def FindPath(self, root, sum):
        if not root:
            return []
        val, *kids = root.val, root.left, root.right
        if any(kids):
            return [
                [val]+path
                for kid in kids if kid
                for path in self.FindPath(kid,sum-val)
            ]       #这个语句太复杂了，改成更容易理解的下一种
        return [[val]] if val==sum else []

#递归
class Solution:
    def FindPath(self, root, sum):
        if not root: return []
        if root.left or root.right:
            a = self.FindPath(root.left, sum-root.val) + self.FindPath(root.right, sum-root.val)
            return [[root.val]+i for i in a]
        return [[root.val]] if sum==root.val else []
