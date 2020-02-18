


class TestClass:
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return str(self.value)

def test1():
    values = [TestClass(x) for x in range(10)]
    hold_values = values

    values = values[3:]
    values.extend([TestClass(x) for x in range(10, 14)])

    values[3].value = 33

    print(values)
    print(hold_values)


if __name__ == '__main__':
    test1()