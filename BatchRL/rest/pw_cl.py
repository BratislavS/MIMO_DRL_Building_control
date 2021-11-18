from typing import Tuple
from getpass import getpass


def get_pw() -> Tuple[str, str]:
    """Commandline login getter tool

    Returns:
        Tuple with username and password strings.
    """
    username = input("Please enter username: ")
    pw = getpass("Enter password: ")
    return username, pw
