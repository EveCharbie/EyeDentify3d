from enum import Enum


class ErrorType(Enum):
    """This enum defines the way errors are handled"""

    SKIP = "continues silently on errors"
    PRINT = "prints the error message to the console"
    FILE = "print the error message to a file"
    RAISE = "raises an exception on errors"

    def __call__(self, error_str: str) -> None:
        """
        Handle the error based on the enum type

        Parameters
        ----------
        error_str: The error message to handle
        """
        if self == ErrorType.FILE:
            with open("bad_data_files.txt", "a") as bad_data_file:
                bad_data_file.write(error_str + "\n")
        elif self == ErrorType.PRINT:
            print(error_str, end="")
        elif self == ErrorType.SKIP:
            pass
        elif self == ErrorType.RAISE:
            raise RuntimeError(error_str)
        else:
            raise ValueError(f"Unknown error type: {self}")
