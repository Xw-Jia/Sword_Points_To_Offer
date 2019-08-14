'''
两个单调递增的链表 输出两个链表合成后的链表 合成后单调不减
两个指针指向两个链表的头节点，取小的放进合并后的，剩余部分再比较
'''

#使用递归
class Solution:
    def Merge(self, l1,l2):
        if not l1 or not l2:
            return l1 or l2     #l1或者l2有None则跳出递归
        if l1.val < l2.val:
            l1.next = self.Merge(l1.next, l2)
            return l1
        else:
            l2.next = self.Merge(l1, l2.next)
            return l2
#使用迭代
class Solution:
    def Merge(self, pHead1, pHead2):
        l = head = ListNode(0)  #新建一个虚拟节点
        while pHead1 and pHead2:
            if pHead1.val <= pHead2.val:
                l.next, pHead1 = pHead1, pHead1.next
            else:
                l.next, pHead2 = pHead2, pHead2.next
            l = l.next
        l.next = pHead1 or pHead2       #跳出while之后，多余的部分直接塞到后面尾部
        return head.next