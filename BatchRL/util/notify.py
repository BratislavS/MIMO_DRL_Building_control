"""Notification module.

Can be used to send mails from your gmail account.
You need to allow unsave apps access under your
account settings.

It is assumed that the `debug_email` and the normal email (`receiver_email`)
do have the same password, stored in the file `python_notifyer.txt`.
For obvious reasons, this file is not on Github, it should be placed
in the root folder of the Github repository.
"""
import os
import smtplib
import ssl
import sys
import time
import traceback
from pathlib import Path
from typing import List, Callable

from util.util import force_decorator_factory

curr_dir = Path(os.path.dirname(os.path.realpath(__file__)))

# Signal codes, valid on Windows at least
codes = [
    "Close Event (e.g. KeyboardInterrupt)",
    "Logoff (e.g. Ctrl + Pause / Break or Ctrl + Fn + B)",
    "Shutdown (e.g. X in Powershell, or disconnected in PyCharm),"
]


def set_exit_handler(func: Callable) -> None:
    """Catching kill events.

    Should work for windows and linux, only tested on windows.

    Not working in Powershell:
        If process is killed via task manager :(
        If PC is shutdown.
    Not working in PyCharm:
        If exited and process is terminated. (Works for exiting and disconnecting
            or killing PyCharm via task manager.)

    From: https://danielkaes.wordpress.com/2009/06/04/how-to-catch-kill-events-with-python/

    Args:
        func: The function to execute when handling the exit.
    """
    if os.name == "nt":
        try:
            import win32api
            win32api.SetConsoleCtrlHandler(func, True)
        except ImportError:
            version = ".".join(map(str, sys.version_info[:2]))
            raise Exception(f"pywin32 not installed for Python {version}")
    else:
        import signal
        signal.signal(signal.SIGTERM, func)


def test_kill_event() -> None:
    """Test for catching kill events.

    You have 30 seconds to kill the execution and see what
    happens, then check your mail. :)
    """
    with FailureNotifier("test", verbose=0, debug=True):
        print("Sleeping...")
        time.sleep(30.0)
        print("Done Sleeping, you were too late!")
        raise ValueError("Fuck")


class FailureNotifier:
    """Context manager for failure notifications.

    Sends a mail if an error happens while it is active including
    the stack trace if available.

    Uses :func:`util.notify.set_exit_handler` to catch all kinds
    of interrupts, but will not provide the stacktrace in those cases.
    """

    _sent_mail: bool = False

    def __init__(self, name: str, verbose: int = 1,
                 debug: bool = False, exit_fun: Callable = None):
        self.name = name
        self.verbose = verbose
        self.debug = debug
        self.exit_fun = exit_fun

    def __enter__(self):
        """Enter context, sets exit handler for uncaught interrupts."""

        if self.verbose:
            print("Entering FailureNotifier...")

        # Set the exit handler to the exit function since
        # e.g. if you press X on the powershell console, this
        # will not be caught by the context manager.
        def on_exit(sig, func=None):
            # Skip sig == 0 cases?
            if self.verbose:
                print("Exiting because of interrupt, sending notification...")
            sig_desc = codes[sig] if sig < len(codes) else "None"
            msg = f"Program was mysteriously killed by somebody or something. " \
                  f"Clues are: {sig_desc} (Code: {sig})"
            if os.name != "nt":
                msg += f"Func: {func}"
            self._on_exit(msg=msg)

            if self.verbose:
                print("Notification sent.")

        set_exit_handler(on_exit)
        return self

    def _on_exit(self, msg: str) -> None:
        """Called when an error happens."""
        if self.exit_fun is not None:
            self.exit_fun(None, None, None)
        sub = f"Error while executing '{self.name}'."
        if not self._sent_mail:
            send_mail(self.debug, subject=sub,
                      msg=msg)
            self._sent_mail = True

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        """Exits context, sends a mail if Exception happened."""
        if self.verbose:
            print("Exiting FailureNotifier..")

        p_msg = "Exited FailureNotifier "
        if exc_type is not None:
            # Unhandled exception happened, notify owner.
            msg = traceback.format_exc()
            self._on_exit(msg=msg)
            p_msg += "with unhandled Error."
        else:
            p_msg += "successfully."

        if self.verbose:
            print(p_msg)


def login_from_file(file_name: str) -> List[str]:
    """Loads login information from a file."""
    assert os.path.isfile(file_name), f"File: {file_name} not found!"
    with open(file_name, "r") as f:
        return [l.rstrip() for l in f if l.rstrip() != ""]


@force_decorator_factory()
def send_mail(debug: bool = True,
              subject: str = "Hello there!",
              msg: str = "General Kenobi",
              use_ssl: bool = True) -> None:
    """Sends a mail via python.

    Decorated with the force decorator since a connection timeout
    is likely to happen which will prevent the mail from being sent.

    Not tested for the case `use_ssl` = False.

    Args:
        debug: Whether to use debug mode, will send the mail to the
            debug address.
        subject: Subject of the mail.
        msg: Message of the mail.
        use_ssl: Whether to use SSL, use default.
    """
    # Define message
    message = f"Subject: {subject}\n\n{msg}\n\n" \
              f"This is an automatically generated message, do not reply!"

    # Choose port and smtp client
    port = 465 if use_ssl else 587  # Port for SSL
    smtp_server = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
    ssl.create_default_context()

    # Load password and select mail account
    receiver_email, receiver_pw = login_from_file("../email_receiver_login.txt")
    debug_email, debug_pw = login_from_file("../email_receiver_debug_login.txt")
    sender_email, password = login_from_file("../notify_email_login.txt")
    rec_mail = debug_email if debug else receiver_email

    # Send the mail
    with smtp_server("smtp.gmail.com", port) as server:
        server.login(sender_email, password)
        send_errs = server.sendmail(sender_email, rec_mail, message)
        if len(send_errs) > 0:
            print(f"Error(s) happened: {send_errs}")
