'''
实现二叉树的序列化和反序列化
序列化：把二叉树按某种遍历方式某种格式保存为字符串
通过#表示空节点，！表示节点结束（value!）
反序列化：根据str，重建二叉树
'''
class Solution:
    def __init__(self):
        self.flag = -1
    def Serialize(self, root):
        if not root:
            return '#'
        return str(root.val)+','+self.Serialize(root.left)+self.Serialize(root.right)

    def Deserialize(self, s):
        self.flag += 1
        l = s.split(',')
        if self.flag >= len(s):
            return None
        root = None
        if l[self.flag] != '#':
            root = TreeNode(int(l[self.flag]))
            root.left = self.Deserialize(s)
            root.right = self.Deserialize(s)
        return root