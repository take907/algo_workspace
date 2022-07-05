from unicodedata import decimal


a, b = input().split()
a = int(a)
b = decimal(b)

ans = a*b
print(int(ans))
