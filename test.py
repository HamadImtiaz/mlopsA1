from app import add_numbers

def test_add_numbers():
    assert add_numbers(1, 2) == 3
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0
    assert add_numbers(10.5, 5.5) == 16.0

if __name__ == "__main__":
    test_add_numbers()
    print("All tests passed!")
