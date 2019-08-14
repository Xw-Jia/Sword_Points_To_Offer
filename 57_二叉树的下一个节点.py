'''
找出中序遍历的下一个节点并返回
'''
class Solution:
    def GetNext(self, pNode):
        if pNode == None:
            return None
        #当前给定节点是root时，没有下一节点，先假定pNext=None
        pNext = None
        #如果输入节点有右子树，下一系欸但那是右子树的最左节点
        if pNode.right:
            pNode = pNode.right
            while pNode.left:
                pNode = pNode.left
            pNext = pNode
        else:
            #如果有父节点，且当前节点是父节点的左子节点，下一节点就是父节点
            if pNode.next and pNode.next.left == pNode:
                pNext = pNode.next
            #如果有father且是father的右，则向上遍历
            #当遍历道以当前节点为father的左子节点时，输入系欸但的下一系欸但那时当前节点的父节点
            elif pNode.next and pNode.next.right == pNode:
                pNode = pNode.next
                while pNode.next and pNode.next.right == pNode:
                    pNode = pNode.next
                    if pNode.next:
                        pNext = pNode.next
            return pNext
'''
写个窍门，把root开始的中序遍历保存，然后直接找pNode的下一个
'''
class Solution:
    def GetNext(self, pNode):
        dummy = pNode
        while dummy.next:
            dummy = dummy.next
        self.result = []
        self.midTraversal(dummy)
        return self.result[self.result.index(pNode)+1] if self.result.index(pNode) != len(self.result)-1 else None
    def midTraversal(self, root):
        if not root:
            return
        self.midTraversal(root.left)
        self.result.append(root)
        self.midTraversal(root.right)