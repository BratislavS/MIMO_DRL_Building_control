"""Setup module.

Sets up everything that is necessary for this project to
be run. Asks for various login info that will be needed.
If used, one can also specify dummy values as login info
and change it later (or never).

Tested on Windows only!
"""

import os

NEST_LOGIN_FILE = "rest_login.txt"
OPCUA_LOGIN_FILE = "opcua_login.txt"
EMAIL_LOGIN_FILE = "email_receiver_login.txt"
DEBUG_EMAIL_LOGIN_FILE = "email_receiver_debug_login.txt"
EMAIL_SEND_LOGIN_FILE = "notify_email_login.txt"
VENV_DIR = "venv"


def str2bool(v) -> bool:
    """Converts a string to a boolean.

    Raises:
        ValueError: If it cannot be converted.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', '1.0'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', '0.0'):
        return False
    else:
        raise ValueError(f"Boolean value expected, got {v}")


def get_login_and_write_to_file(file_name: str, name: str) -> None:
    """Asks the user for the login data.

    If the file already exists, asks if the user wants
    to overwrite it.

    Args:
        file_name: The file containing the login info.
        name: The name of whatever the login is for.
    """
    # Check if login already exists
    f_exists = os.path.isfile(file_name)
    create_file = not f_exists
    if f_exists:
        parse_error = True
        while parse_error:
            ans = input(f"Overwrite {name} login info? ")
            try:
                create_file = str2bool(ans)
                parse_error = False
            except ValueError:
                print("Your input was not understood!")

    # Ask for login and save to file
    if create_file:
        nest_user = input(f"Provide your {name} username: ")
        nest_pw = input("And password: ")

        with open(file_name, "w") as f:
            f.write(nest_user + "\n")
            f.write(nest_pw + "\n")


def main():
    print("Setting up everything...")

    # Check platform
    using_win = os.name == 'nt'
    if not using_win:
        print("May not work, only tested on Windows!")

    if not os.path.isdir(VENV_DIR):
        print("Setting up virtual environment...")
        cmd = "py" if using_win else "python3"
        act_path = os.path.join(VENV_DIR, "Scripts", "activate")
        req_path = os.path.join("BatchRL", "requirements.txt")
        os.system(f"{cmd} -m venv {VENV_DIR}")
        os.system(f"{act_path} & {cmd} -m pip install -r {req_path}")
        print("Venv setup done :)")
        print("")

    # Get NEST login data and store in file
    get_login_and_write_to_file(NEST_LOGIN_FILE, "NEST database")

    # Get NEST login data and store in file
    get_login_and_write_to_file(OPCUA_LOGIN_FILE, "Opcua client")

    # Get notification email login data and store in file
    get_login_and_write_to_file(EMAIL_LOGIN_FILE, "Notification receiver email")

    # Get notification email login data and store in file
    get_login_and_write_to_file(EMAIL_SEND_LOGIN_FILE, "Notification sender email")

    # Get debug notification email login data and store in file
    get_login_and_write_to_file(DEBUG_EMAIL_LOGIN_FILE, "Debug notification receiver email")

    print("Setup done!")


if __name__ == '__main__':
    main()
