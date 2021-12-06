class Node(object):

    def __init__(self, character):
        self.character = character
        self.left_node = None
        self.middle_node = None
        self.right_node = None
        self.value = 0


class TST(object):

    def __init__(self):
        self.root_node = None

    def put(self, key, value):
        self.root_node = self.putItem(self.root_node, key, value, 0)

    def putItem(self, node, key, value, index):

        c = key[index]

        if node is None:
            node = Node(c)

        if c < node.character:
            node.left_node = self.putItem(node.left_node, key, value, index)
        elif c > node.character:
            node.right_node = self.putItem(node.right_node, key, value, index)
        elif index < len(key) - 1:
            node.middle_node = self.putItem(node.middle_node, key, value, index + 1)
        else:
            node.value = value

        return node

    def get(self, key):

        node = self.getItem(self.root_node, key, 0)

        if node is None:
            return -1

        return node.value

    def getItem(self, node, key, index):

        if node is None:
            return None

        c = key[index]

        if c < node.character:
            return self.getItem(node.left_node, key, index)
        elif c > node.character:
            return self.getItem(node.right_node, key, index)
        elif index < len(key) - 1:
            return self.getItem(node.middle_node, key, index + 1)
        else:
            return node


if __name__ == "__main__":

    tst = TST()

    tst.put("apple", 100)
    tst.put("orange", 200)
    tst.put("pear", 3)

    print(tst.get("orange"))
