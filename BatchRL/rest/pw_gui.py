"""Login GUI to get login data from a user.

The main function is `get_pw`, it uses the other classes
to do all the stuff.
"""

from typing import Tuple

import wx


class LoginHolder:
    """Class that holds the login information.
    """

    login_data: Tuple[str, str] = None  #: The login information

    def __init__(self):
        pass


class LoginDialog(wx.Dialog):
    """Class defining the login dialog GUI

    Adapted from: https://dzone.com/articles/wxpython-how-create-login
    """

    def __init__(self, lh: LoginHolder):
        """
        Constructs the login dialog to ask for
        username and password.

        Args:
            lh: LoginHolder, class where the login data will be saved.
        """

        wx.Dialog.__init__(self, None, title="Login")
        self.lh = lh

        # User info        
        user_sizer = wx.BoxSizer(wx.HORIZONTAL)
        user_lbl = wx.StaticText(self, label="Username:")
        user_sizer.Add(user_lbl, 0, wx.ALL | wx.CENTER, 5)
        self.user = wx.TextCtrl(self)
        user_sizer.Add(self.user, 0, wx.ALL, 5)

        # Password info
        p_sizer = wx.BoxSizer(wx.HORIZONTAL)
        p_lbl = wx.StaticText(self, label="Password:")
        p_sizer.Add(p_lbl, 0, wx.ALL | wx.CENTER, 5)
        self.password = wx.TextCtrl(self, style=wx.TE_PASSWORD | wx.TE_PROCESS_ENTER)
        p_sizer.Add(self.password, 0, wx.ALL, 5)

        # Add text fields
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(user_sizer, 0, wx.ALL, 5)
        main_sizer.Add(p_sizer, 0, wx.ALL, 5)

        # Login button
        btn = wx.Button(self, label="Login")
        btn.Bind(wx.EVT_BUTTON, self.on_login)
        main_sizer.Add(btn, 0, wx.ALL | wx.CENTER, 5)

        self.SetSizer(main_sizer)

    def on_login(self, event) -> None:
        """
        Saves login data to login holder and destroys dialog.
        Executed when the login button is clicked.

        Args:
            event: Not used
        Returns:
            None
        """

        # Save login data and destroy
        user_password = self.password.GetValue()
        username = self.user.GetValue()
        self.lh.login_data = (username, user_password)
        self.Destroy()


class MainFrame(wx.Frame):
    """
    Frame calling the login dialog only.
    """

    def __init__(self, lh: LoginHolder):
        """
        Constructor of single frame which calls
        the LoginDialog when initialized.

        :param lh: LoginHolder, class where the login data will be saved.
        """
        self.lh = lh
        wx.Frame.__init__(self, None, title="Main App")
        wx.Panel(self)

        # Ask user to login
        dlg = LoginDialog(self.lh)
        dlg.ShowModal()
        self.Destroy()


def get_pw() -> Tuple[str, str]:
    """The main function to get the login info.

    Opens GUI and retrieves the login information
    and returns it.

    Returns:
        Tuple containing username and password.
    """
    lh = LoginHolder()
    app = wx.App(False)
    frame = MainFrame(lh)

    # Run GUI
    frame.Show(False)
    app.MainLoop()
    if frame:
        frame.Destroy()

    return lh.login_data
