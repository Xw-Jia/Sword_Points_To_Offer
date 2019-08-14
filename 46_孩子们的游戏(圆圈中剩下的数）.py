'''
约瑟夫环
fn = [(fn-1)+m] % n 其中，fn是场上有n个人时在场的人的编号
f1 = 0
'''
class Solution:
    def LastRemaining_Solution(self, n ,m):
        if n <= 0 or m <= 0:
            return -1
        last_num = 0
        for i in range(2, n+1):
            last_num = (last_num + m)%2
        return last_num
#或者list旋转数组也可以？
class Solution:
    def LastRemaining_Solution(self, n, m):
        if n<=0 or m<=0:
            return -1
        seats = range(n)
        while seats:
            rot = (m-1) % len(seats)
            seats, last = seats[rot+1:] + seats[:rot], seats[rot]
        return last
