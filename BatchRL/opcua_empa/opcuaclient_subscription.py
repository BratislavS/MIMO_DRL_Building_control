"""Opcua client wrapper module.

Handles a few common exceptions and hides implementation details.
See the function `example_usage` for an example how to use the `OpcuaClient`.

Original implementation by Ralf Knechtle, modified by me.
Main changes: Added context manager functionality, more docstrings,
more specific error handling, type hints and an example.

.. moduleauthor:: Christian Baumann and Ralf Knechtle
"""
import datetime
import logging
import socket
import time
import warnings
from asyncio.base_futures import CancelledError
from typing import List, Optional

import pandas as pd
from opcua import Client, Subscription
from opcua.ua import UaStatusCodeError, DataValue, Variant

# Set pandas printing options, useful e.g. if you want to print
# dataframes with long strings in them, as they are needed in the client.
pd.options.display.width = 1000
pd.options.display.max_columns = 500
pd.options.display.max_colwidth = 200

# Initialize and configure logger
logging.basicConfig(format='%(asctime)s - OPC UA %(message)s', level=logging.WARNING)
logger = logging.getLogger('opc ua client')


MAX_TEMP: int = 28  #: Them maximum temperature to set.
MIN_TEMP: int = 10  #: Them minimum temperature to set.


def example_usage() -> None:
    """Example usage of the :class:`OpcuaClient` class defined below.

    You will need to set `user` and `password` to your
    personal credentials.

    Sets the room temperature setpoint of room 475 at DFAB to 10 degrees C
    for a short time.

    DO NOT RUN IF AN EXPERIMENT IS CURRENTLY RUNNING!
    """

    # The initial values to write
    write_vals = [
        28,  # Temperature setpoint
        True,  # Research request
        True,  # Watchdog
    ]

    # Define nodes to read and write
    write_nodes = [
        'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strWrite_L.strSensoren.strR2.strB872.rValue1',
        'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strWrite_L.strSensoren.strR2.strB872.bReqResearch',
        'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strWrite_L.strSensoren.strR2.strB872.bWdResearch',
    ]
    read_nodes = [
        'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR2.strB872.bAckResearch',
        'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR2.strB872.rValue1',
        'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR2.strB872.rValue2',
        'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR2.strB872.bValue1',
        'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ2.strY703.bValue1',
        'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ2.strY704.bValue1',
        'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ2.strY705.bValue1',
    ]

    # Define dataframes
    df_write = pd.DataFrame({'node': write_nodes, 'value': write_vals})
    df_read = pd.DataFrame({'node': read_nodes})

    # Use the opcua client as a context manager, it connects and disconnects
    # automatically.
    with OpcuaClient(user='user', password='password') as opcua_client:
        # Subscribe to read nodes and wait a bit before reading
        opcua_client.subscribe(df_read, sleep_after=1.0)

        for k in range(60):
            # Read values
            read_vals = opcua_client.read_values()

            # Do something with the read values
            print(read_vals)

            # Write values and wait
            df_write['value'][0] = 10  # Set temperature setpoint
            df_write['value'][2] = toggle()  # Toggle for watchdog
            opcua_client.publish(df_write, log_time=True, sleep_after=1.0)


def toggle() -> bool:
    """Toggles every 5 seconds.

    The watchdog has to toggle every 5 seconds
    otherwise the connection will be refused.
    """
    return datetime.datetime.now().second % 10 < 5


class _SubHandler(object):
    """Subscription Handler.

    To receive events from server for a subscription
    data_change and event methods are called directly from receiving thread.
    Do not do expensive, slow or network operation there. Create another
    thread if you need to do such a thing. You have to define here
    what to do with the received date.
    """

    def __init__(self):
        self.df_Read = pd.DataFrame(data={'node': [], 'value': []}).astype('object')
        self.json_Read = self.df_Read.to_json()

    def datachange_notification(self, node, val, _):
        try:
            df_new = pd.DataFrame(data={'node': [], 'value': []}).astype('object')
            df_new.at[0, 'node'] = str(node)
            df_new.at[0, 'value'] = str(val)
            self.df_Read = self.df_Read.merge(df_new, on=list(self.df_Read), how='outer')
            self.df_Read.drop_duplicates(subset=['node'], inplace=True, keep='last')
            self.json_Read = self.df_Read.to_json()
            logger.info('read %s %s' % (node, val))
        except Exception as e:
            logger.error(e)

    @staticmethod
    def event_notification(event):
        logger.info("Python: New event", event)


class OpcuaClient(object):
    """Wrapper class for Opcua Client.

    Can be used as a context manager, then it will connect
    and disconnect automatically. Especially it will also disconnect
    in case of e.g. KeyboardInterrupts.
    """

    # Public member variables
    client: Client  #: The original opcua.Client

    df_read: pd.DataFrame = None  #: Read data frame.
    df_write: pd.DataFrame = None  #: Write data frame.

    # Private member variables
    _node_objects: List
    _data_types: List
    _ua_values: List

    _nodelist_read: List

    _connected: bool = False  #: Bool specifying successful connection.
    _force_connect: bool
    _type_defs = None
    _sub: Subscription = None  #: The subscription object.
    _sub_init: bool = False  #: Whether a subscription was initialized.
    _pub_init: bool = False  #: Whether publishing was initialized.

    def __init__(self, url: str = 'opc.tcp://ehub.nestcollaboration.ch:49320',
                 application_uri: str = 'Researchclient',
                 product_uri: str = 'Researchclient',
                 user: str = 'username',
                 password: str = 'password',
                 force_connect: bool = True):
        """Initialize the opcua client."""

        # Setup client
        c = Client(url=url, timeout=4)
        c.set_user(user)
        c.set_password(password)
        host_name = socket.gethostname()
        c.application_uri = f"{application_uri}:{host_name}:{user}"
        c.product_uri = f"{product_uri}:{host_name}:{user}"

        # Store in class
        self._force_connect = force_connect
        self.client = c
        self.handler = _SubHandler()

    def __enter__(self):
        """Enter method for use as context manager."""
        suc_connect = self.forced_connect() if self._force_connect else self.connect()
        if suc_connect:
            return self
        self.disconnect()
        raise UaStatusCodeError("Connection failed!")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit method for use as context manager."""
        self.disconnect()

    def forced_connect(self) -> bool:
        """Forces a connection by trying until it succeeds."""
        while not self._connected:
            suc_con = self.connect()
            if not suc_con:
                time.sleep(0.5)
                if self._connected:
                    self._connected = False
        return True

    def _connect_and_renew_sub(self, silent: bool = True):
        if self.connect(silent=silent):
            self._sub = self.client.create_subscription(period=0, handler=self.handler)
            self._sub.subscribe_data_change(self._nodelist_read)

    def connect(self, silent: bool = False) -> bool:
        """Connect the client to the server.

        If connection fails, prints a possible solution.

        Returns:
            True if connection successful, else False.
        """
        msg, ret_val = None, True
        try:
            self.client.connect()
            self._connected = True
            self._type_defs = self.client.load_type_definitions()
            msg = "Connection to server established."
        except UaStatusCodeError as e:
            msg = f"Exception: {e} happened while connecting, " \
                  f"try waiting a bit and rerun it!"
            if self._type_defs is None and self._connected:
                msg += " Type definition loading failed!"
            ret_val = False
        except socket.gaierror:
            msg = f"Gaierror happened while connecting, "\
                  f"check your internet connection!"
            ret_val = False
        except Exception as e:
            msg = f"Unexpected exception of type {type(e)} happened: {e}"
            ret_val = False
        if not silent:
            logging.warning(msg)
        return ret_val

    def read_values(self) -> Optional[pd.DataFrame]:
        """Returns the read values in the dataframe.

        If something fails, returns None.
        """
        if not self._connected:
            logging.warning("Not connected for reading, connecting...")
            self._connect_and_renew_sub()
            return

        if not self._sub_init:
            logging.warning("You need to subscribe first!")
            return

        try:
            self.handler.df_Read.set_index('node', drop=True)
            return self.handler.df_Read.copy()
        except ValueError as e:
            logging.warning(f"Exception: {e} while reading values")

    def disconnect(self) -> None:
        """Disconnect the client.

        Deletes the subscription first to avoid error.
        If client hasn't been connected, does nothing.
        """
        # If it wasn't connected, do nothing
        if not self._connected:
            try:
                self.client.disconnect()
            except Exception as e:
                logging.warning(f"Exception {e} while disconnecting..")
            return
        try:
            # Need to delete the subscription first before disconnecting
            self._connected = False
            if self._sub is not None:
                self._sub.delete()
            self.client.disconnect()
            logging.warning("Server disconnected.")
        except UaStatusCodeError as e:
            logging.warning(f"Server disconnected with error: {e}")
        except Exception as e:
            logging.warning(f"Unexpected exception of type {type(e)}: "
                            f"{e} happened while disconnecting...")

    def subscribe(self, df_read: pd.DataFrame, sleep_after: float = None) -> None:
        """Subscribe all values you want to read.

        If it fails, a warning is printed and some values might
        not be read correctly.

        If `sleep_after` is None, there is no sleeping after subscribing.

        Args:
            df_read: The dataframe with the read nodes.
            sleep_after: Number of seconds to wait after subscribing.
        """
        # Check if already subscribed
        if self._sub_init:
            logging.warning("You already subscribed!")

        self.df_read = df_read
        nodelist_read = [self.client.get_node(row['node'])
                         for i, row in self.df_read.iterrows()]
        self._nodelist_read = nodelist_read

        # Try subscribing to the nodes in the list.
        try:
            # Create subscription and subscribe to read nodes
            self._sub = self.client.create_subscription(period=0, handler=self.handler)
            sub_res = self._sub.subscribe_data_change(nodelist_read)
            self._sub_init = True

            # Check if subscription was successful
            for ct, s in enumerate(sub_res):
                if not type(s) is int:
                    warnings.warn(f"Node: {nodelist_read[ct]} not found!")
            logging.warning("Subscription requested.")
        except Exception as e:
            # TODO: Remove or catch more specific error!
            logging.warning(f"Exception: {e} happened while subscribing!")
            raise e

        # Sleep
        if sleep_after is not None:
            time.sleep(sleep_after)

    def publish(self, df_write: pd.DataFrame,
                log_time: bool = False,
                sleep_after: float = None) -> None:
        """Publish (write) values to server.

        Initializes publishing if called for first time. If the actual
        publishing fails, a warning message is printed.
        If `sleep_after` is None, there is no sleeping after publishing.

        Args:
            df_write: The dataframe with the write nodes and values.
            log_time: Whether to log the time it took to publish.
            sleep_after: Time in seconds to sleep after publishing.

        Raises:
            UaStatusCodeError: If initialization of publishing fails.
        """
        if not self._connected:
            if log_time:
                logging.warning("Not connected for publishing, connecting...")
            self._connect_and_renew_sub()

            # Sleep
            if sleep_after is not None:
                time.sleep(sleep_after)
            return

        # Remember current time
        t0 = datetime.datetime.now()
        self.df_write = df_write

        # Initialize publishing
        if not self._pub_init:
            self._node_objects = [self.client.get_node(node)
                                  for node in self.df_write['node'].tolist()]
            try:
                self._data_types = [nodeObject.get_data_type_as_variant_type()
                                    for nodeObject in self._node_objects]
                self._pub_init = True
                logging.warning("Publishing initialized.")
            except UaStatusCodeError as e:
                logging.warning(f"UaStatusCodeError while initializing publishing!: {e}")
                raise e

        # Publish values, failures to publish will not raise an exception.
        try:
            self._ua_values = [DataValue(Variant(value, d_t)) for
                               value, d_t in zip(self.df_write['value'].tolist(), self._data_types)]
            self.client.set_values(nodes=self._node_objects, values=self._ua_values)
            for n, val in zip(self._node_objects, self._ua_values):
                logger.info('write %s %s' % (n, val))
        except UaStatusCodeError as e:
            logging.warning(f"UaStatusCodeError: {e} happened while publishing!")
        except (ConnectionError, AttributeError, CancelledError, TimeoutError):
            # Try reconnecting including renewing the subscription.
            logging.warning(f"Lost connection, trying to reconnect...")
            self._connected = False
            self._connect_and_renew_sub()
        except Exception as e:
            logging.warning(f"Unexpected exception of type {type(e)} happened: {e}")

        # Log the time used for publishing
        if log_time:
            dt = datetime.datetime.now() - t0
            logging.warning(f"Publishing took: {dt}")

        # Sleep
        if sleep_after is not None:
            time.sleep(sleep_after)
