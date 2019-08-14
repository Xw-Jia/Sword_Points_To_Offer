'''
输入复杂链表，节点有节点值和两个指针，一个指向下一节点，另一个指向任意一个节点
返回结果为复制后的复杂链表的head
思路：遍历两次，第一次复制到字典中，第二次关联
'''
class Solution:
    def Clone(self, head):
        cp = {None:None}
        m = n = head
        #复制
        while m:
            cp[m] = RandomListNode(m.label)
            m = m.next
        #关联
        while n:
            cp[n].next = cp[n.next]
            cp[n].random = cp[n.random]
            n = n.next
        return cp[head]