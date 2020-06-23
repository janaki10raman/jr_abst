#syntactic_unit.py
class SyntacticUnit(object):
    def __init__(self, text, token=None, tag=None, index=-1):
        self.text = text
        self.token = token
        self.tag = tag[:2] if tag else None  # Just first two letters of tag
        self.index = index
        self.score = -1
        
    def __str__(self):
        return "Original unit: '" + self.text + "' *-*-*-* " + "Processed unit: '" + self.token + "'"

    def __repr__(self):
        return str(self)