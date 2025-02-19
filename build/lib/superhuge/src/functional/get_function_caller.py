import inspect


def get_caller():
    stack = inspect.stack()
    if len(stack) > 2:
        caller_frame = stack[2]  # Index 2 refers to the direct caller
        caller_name = caller_frame.function
        return caller_name
    return None
