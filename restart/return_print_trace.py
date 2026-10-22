"""A tiny demonstration of displayed text versus a function's return value."""


def return_status():
    return "Stated"


def print_status():
    print("Stated")


print("Calling the function that returns a value:")
print(return_status())

print("Calling the function that only prints:")
print(print_status())