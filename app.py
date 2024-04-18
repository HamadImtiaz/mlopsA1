def add_numbers(num1, num2):
    return num1 + num2

if __name__ == "__main__":
    num1 = float(input("Enter the first number: "))
    num2 = float(input("Enter the second number: "))
    result = add_numbers(num1, num2)
    print("The sum is:", result)
