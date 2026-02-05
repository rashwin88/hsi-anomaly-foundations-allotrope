"""
Standard Not Implemented
"""


class ImplementationIncompleteError(Exception):
    """
    An error that is raised when a method is not implemented.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
