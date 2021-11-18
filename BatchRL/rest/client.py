"""REST client for data retrieval from NEST database.

This module can be used to download and save data
from the NEST database. Returns the values and the
timesteps in numpy format for each defined data series.
The following example shows how to use this module.

If you set `pw_from_cl` of :class:`rest.client._Client` to
True, then the login data can be input via command line.
Otherwise it is read from the file `rest_login.txt` located
in the root directory of the repository.

Example usage::

    # Define data.
    test_data = DataStruct(
        id_list=[421100171, 421100172],
        name="Test",
        start_date='2019-08-08',
        end_date='2019-08-09'
    )

    # Get data from SQL database
    data, metadata = test_data.get_data()

    # Get data corresponding to first ID (421100171)
    values, timestamps = data[0]

    # Do something with the data
    # Add your code here...
    print(values, timestamps)

.. moduleauthor:: Christian Baumann and Ralf Knechtle
"""
import os
import shutil
import time
from ast import literal_eval
from pathlib import Path
from typing import Tuple, List, Optional, Union, Any

import numpy as np
import pandas as pd
import requests

from util.notify import login_from_file
from util.util import DEFAULT_END_DATE

USE_CL: bool = True  #: Whether to use the command line for the login.
if not USE_CL:
    from .pw_gui import get_pw
else:
    from .pw_cl import get_pw

# Find login file
curr_dir = Path(os.path.dirname(os.path.realpath(__file__)))
login_path = os.path.join(curr_dir.parent.parent, "rest_login.txt")

#: Where to put the local copy of the data.
save_dir: str = '../Data/'

#: Data type for data returned by reading from database
NestDataT = Tuple[List[Tuple[np.ndarray, np.ndarray]], List[str]]


def check_date_str(ds: str, err: Exception = None) -> None:
    """Raises an error if `ds` is not a valid date string."""
    invalid = len(ds) != 10 or ds[4] != "-" or ds[7] != "-"
    if invalid:
        if err is None:
            err = ValueError(f"Invalid date string: {ds}")
        raise err


def _get_data_folder(name: str, start_date: str, end_date: str) -> str:
    """
    Defines the naming of the data directory given
    the name and the dates.

    Args:
        name: Name of data.
        start_date: Start of data collection.
        end_date: End of data collection.

    Returns:
        Full path of data folder.
    """
    full_name = f"{start_date}__{end_date}__{name}"
    data_dir = os.path.join(save_dir, full_name)
    return data_dir


class _Client(object):
    """Client for data retrieval.

    Reads from local disk if it already exists or else
    from SQL data base of NEST. Once loaded from the
    server, it can be stored to and reloaded from
    the local disk.
    """
    pw_from_cl: bool = False
    name: str = None

    np_data: List[Tuple[np.ndarray, np.ndarray]] = None
    meta_data: List[str] = None

    _auth: Any = None

    _DOMAIN: str = 'nest.local'
    _URL: str = 'https://visualizer.nestcollaboration.ch/Backend/api/v1/datapoints/'

    def __init__(self,
                 name,
                 start_date: str = '2019-01-01',
                 end_date: str = DEFAULT_END_DATE,
                 verbose: int = 0):
        """Initialize parameters and empty data containers.

        Args:
            name: Name of the data.
            start_date: Starting date in string format.
            end_date: End date in string format.
        """
        self.save_dir = save_dir
        self.start_date = start_date
        self.end_date = end_date
        self.name = name
        self.verbose = verbose

    def read(self, df_data: List[str]) -> Optional[NestDataT]:
        """Reads data defined by the list of column IDs `df_data`.

        Reads only the data that was collected between
        `self.start_date` and `self.end_date`. Returns None if
        some error is raised by the client.

        Args:
            df_data: List of IDs in string format.

        Returns:
            (List[(Values, Dates)], List[Metadata])
        """

        self.np_data = []
        self.meta_data = []

        # https://github.com/brandond/requests-negotiate-sspi/blob/master/README.md
        from requests_negotiate_sspi import HttpNegotiateAuth

        # Check Login
        username, pw = get_pw() if self.pw_from_cl else login_from_file(login_path)
        self._auth = HttpNegotiateAuth(domain=self._DOMAIN,
                                       username=username,
                                       password=pw)
        s = requests.Session()
        try:
            # This fails if the username exists but the password
            # is wrong, but not if the username does not exist?!!
            r = s.get(url=self._URL, auth=self._auth)
        except TypeError:
            print("Login failed, invalid password!")
            return None
        except Exception as e:
            print(e)
            print("A problem occurred!")
            return None

        # Check if login valid.
        if r.status_code != requests.codes.ok:
            print("Login failed, invalid username!")
            return None
        print(time.ctime() + ' REST client login successful.')

        # Iterate over column IDs
        for ct, column in enumerate(df_data):
            url = self._URL + column
            meta_data = s.get(url=url).json()
            self.meta_data += [meta_data]
            url += f"/timeline?startDate={self.start_date}&endDate={self.end_date}"
            df = pd.DataFrame(data=s.get(url=url).json())

            # Convert to Numpy
            values = df.loc[:, "value"].to_numpy()
            ts = pd.to_datetime(df.loc[:, "timestamp"])
            ts = ts.to_numpy(dtype=np.datetime64)
            self.np_data += [(values, ts)]
            print(f"Added column {ct + 1} with ID {column}.")

        print(time.ctime() + ' REST client data acquired')
        return self.np_data, self.meta_data

    def read_offline(self) -> NestDataT:
        """Read numpy and text data that has already been created.

        Returns:
             values, dates and metadata.
        """

        # Get folder name
        data_dir = _get_data_folder(self.name, self.start_date, self.end_date)

        # Count files
        ct = 0
        for f in os.listdir(data_dir):
            if f[:5] == "dates":
                ct += 1

        # Loop over files in directory and insert data into lists
        val_list: List[Optional[np.ndarray]] = [None] * ct
        ts_list: List[Optional[np.ndarray]] = [None] * ct
        meta_list: List[str] = [""] * ct
        for k in range(ct):
            val_list[k] = np.load(os.path.join(data_dir, f"values_{k}.npy"))
            ts_list[k] = np.load(os.path.join(data_dir, f"dates_{k}.npy"))
            with open(os.path.join(data_dir, f"meta_{k}.txt"), 'r') as data:
                contents = data.read()
                meta_list[k] = literal_eval(contents)

        # Transform to list of pairs and return
        list_of_tuples = list(zip(val_list, ts_list))
        return list_of_tuples, meta_list

    def write_np(self, overwrite: bool = False) -> None:
        """
        Writes the read data in numpy format
        to files.

        Args:
            overwrite: Whether to overwrite existing data with same name.
        """
        name = self.name
        print("Writing Data to local disk.")

        # Create directory
        if self.start_date is None:
            raise ValueError("Read data first!!")
        data_dir = _get_data_folder(name, self.start_date, self.end_date)
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        elif overwrite:
            raise NotImplementedError("Not implemented, remove manually and try again.")
        else:
            print("Directory already exists.")
            return

        # Loop over data and save column-wise
        assert len(self.np_data) != 0, f"Nothing to save!"
        for ct, data_tup in enumerate(self.np_data):
            v, t = data_tup
            v_name = os.path.join(data_dir, 'values_' + str(ct) + '.npy')
            np.save(v_name, v)
            d_name = os.path.join(data_dir, 'dates_' + str(ct) + '.npy')
            np.save(d_name, t)
            meta_name = os.path.join(data_dir, 'meta_' + str(ct) + '.txt')
            with open(meta_name, 'w') as data:
                data.write(str(self.meta_data[ct]))


class DataStruct:
    """Main Class for data retrieval from NEST database.

    The data is defined when initializing the class
    by a list of IDs, a name and a date range.
    The method `get_data` then retrieves the data when needed.
    Once read, the data is cached in `save_dir` for faster
    access if read again.
    """

    def __init__(self,
                 id_list: Union[List[int], List[str]],
                 name: str,
                 start_date: str = '2019-01-01',
                 end_date: str = DEFAULT_END_DATE):
        """Initialize DataStruct.

        Args:
            id_list: IDs of the data series, can be strings or ints.
            name: Name of the collection of data series.
            start_date: Begin of time interval.
            end_date: End of time interval.
        """
        # Initialize values
        self._name = name
        self._start_date = start_date
        self._end_date = end_date
        self.REST = _Client(self._name, self._start_date, self._end_date)

        # Convert elements of id_list to strings.
        self.data_ids = [str(e) for e in id_list]

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        self._name = new_name
        self.REST.name = new_name

    @property
    def end_date(self) -> str:
        return self._end_date

    @end_date.setter
    def end_date(self, new_end_date: str) -> None:
        self._end_date = new_end_date
        self.REST.end_date = new_end_date

    @property
    def start_date(self) -> str:
        return self._start_date

    @start_date.setter
    def start_date(self, new_start_date: str) -> None:
        self._start_date = new_start_date
        self.REST.start_date = new_start_date

    def copy(self) -> 'DataStruct':
        """Returns a (deep) copy of self."""
        return DataStruct(self.data_ids.copy(), self.name,
                          self.start_date, self.end_date)

    def _slice(self, start_ind: int, end_ind: int) -> 'DataStruct':
        """Helper function for `__getitem__`."""
        n = len(self.data_ids)

        # Raises an error if out of range...
        assert -n <= start_ind <= n and -n <= end_ind <= n, \
            f"Indices: {start_ind} or {end_ind} out of range (n = {n})!"

        # Handles negative indices
        if start_ind < 0:
            start_ind += n
        if end_ind < 0:
            end_ind += n

        new_ids = self.data_ids[start_ind:end_ind].copy()
        new_name = f"{self.name}[{start_ind}:{end_ind}]"
        return DataStruct(new_ids, new_name,
                          self.start_date, self.end_date)

    def __getitem__(self, key) -> 'DataStruct':
        """Allows for slicing w.r.t. the ids."""
        if isinstance(key, slice):
            if key.step is not None and key.step != 1:
                raise NotImplementedError("Only implemented for contiguous ranges!")
            return self._slice(key.start, key.stop)
        return self._slice(key, key + 1)

    def __add__(self, other: 'DataStruct') -> 'DataStruct':
        """Allows the usage of the + operator."""
        new_ids = self.data_ids + other.data_ids
        new_name = f"{self.name}_{other.name}"
        assert self.start_date == other.start_date, f"Incompatible start date!"
        assert self.end_date == other.end_date, f"Incompatible start date!"
        return DataStruct(new_ids, new_name,
                          self.start_date, self.end_date)

    def set_end(self, end_str: str = None) -> None:
        """Set the end date to given string.

        If input is None, the newest already loaded data is chosen.

        Args:
            end_str: The string specifying the end date.
        """
        if end_str is not None:
            check_date_str(end_str)
            self.end_date = end_str
        else:
            # Find already loaded data
            name_len = len(self.name)
            init_str = "0000-00-00"
            curr_str = init_str
            assert name_len > 0, f"What the fuck??"
            for f in os.listdir(save_dir):
                if len(f) > max(24, name_len):
                    if f[-name_len:] == self.name:
                        # Found match
                        end_date_str = f[12:22]
                        if end_date_str > curr_str:
                            curr_str = end_date_str
            if curr_str != init_str:
                check_date_str(curr_str)
                self.end_date = curr_str
            else:
                print("No existing data found!")
                self.end_date = DEFAULT_END_DATE

        # Set attribute in rest client
        self.REST.end_date = self.end_date

    def get_data_folder(self) -> str:
        """Returns path to data.

        Returns the path of the directory where
        the data has been / will be stored.

        Returns:
            Full path to directory of data.
        """
        return _get_data_folder(self.name, self.start_date, self.end_date)

    def get_data(self, verbose: int = 0) -> Optional[Tuple[List, List]]:
        """Get the data associated with the DataStruct

        If the data is not found locally it is
        retrieved from the SQL database and saved locally, otherwise
        the local data is read and returned.

        Returns:
            Tuple with two lists, the first one contains the values and
            the datetimes for each series and the second one contains
            the metadata dict of each series.
        """
        data_folder = self.get_data_folder()
        if not os.path.isdir(data_folder):
            # Read from SQL database and write for later use
            if verbose:
                print("Getting data from NEST database.")
            ret_val, meta_data = self.REST.read(self.data_ids)
            if ret_val is None:
                return None
            self.REST.write_np()
        else:
            # Read locally
            if verbose:
                print("Reading data locally.")
            ret_val, meta_data = self.REST.read_offline()

        # Check data and return
        assert len(ret_val) == len(meta_data) == len(self.data_ids), \
            f"Something fucked up horribly!"
        return ret_val, meta_data


def example():
    """Example usage of REST client.

    Shows you how to use the `DataStruct` class
    to define the data and retrieve it.

    Returns:
        None
    """
    # Example data.
    test_data = DataStruct(
        id_list=[421100171, 421100172],
        name="Test",
        start_date='2019-08-08',
        end_date='2019-08-09'
    )

    # Get data from SQL 
    data, metadata = test_data.get_data()

    # Get data corresponding to first ID (421100171)
    values, timestamps = data[0]

    # Do something with the data
    # Add your code here...
    print(values, timestamps)


# Test DataStruct
TestData = DataStruct(id_list=[421100171, 421100172],
                      name="Test",
                      start_date='2019-08-08',
                      end_date='2019-08-09')


def test_rest_client() -> None:
    """Tests the REST client by requesting test data,
    saving it locally, reading it locally and deleting
    it again."""

    # Load using REST api and locally
    t_dat = TestData
    t_dat.get_data()
    t_dat.get_data()

    # Remove data again
    fol = TestData.get_data_folder()
    shutil.rmtree(fol)
