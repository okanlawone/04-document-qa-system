"""Practice: store a returned status so it can be displayed more than once."""


def return_status():
    return "Stated"


status = return_status()

print("First display:")
print(status)

print("Second display:")
print(status)
