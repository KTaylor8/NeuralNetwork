x = float(input("First number "))
y = float(input("Second number "))
result = 0
op = input("Operator ")

if op == ("+"):
        result = x + y
        print(float(result))

elif op == ("-"):
        result = x-y
        print(float(result))

elif op == ("*"):
        result = x*y
        print(float(result))

elif op == ("/"):
        result = x/y
        print(float(result))

else:
        print("invalid operator")
