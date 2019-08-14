class Solution:
    def match(self, s, patten):
        if not patten: return not s
        f_match = bool(s) and patten[0] in {s[0], '.'}
        if len(patten) > 1 and patten[1] == '*'
            return (self.match(s, patten[2:]) or
                    (f_match and self.match(s[1:], patten))
                    )
        else:
            return f_match and self.match(s[1:], patten[1:])