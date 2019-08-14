'''
删除重复节点，返回链表头指针
思路：先找重复节点，头节点也可能重复，所以需要新建虚拟节点
遍历链表，同时需要把前一个节点与之后不重复的系欸但项链(last负责把前一节点和当前不重复节点相连)
'''
class Solution:
    def deletDuplication(self, pHead):
        if pHead is None or pHead.next is None:
            return pHead
        first = ListNode(-1)    #新建虚拟节点
        first.next = pHead
        last = first
        while pHead and pHead.next:
            if pHead.val == pHead.next.val:
                val = pHead.val
                while pHead and val == pHead.val:
                    pHead = pHead.next  #删除节点
                last.next = pHead
            else:
                last = pHead
                pHead =pHead.next
        return first.next
