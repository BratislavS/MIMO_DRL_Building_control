"""This is run before the documentation is built."""

import os
from shutil import copyfile


DOC_DIR = "../MasterThesis/DocFiles"


def main():
    """Copies the files in `DOC_DIR` to the documentation folder.

    This function might be a bit fragile, but it does its job.
    """
    print("Setting up documentation...")
    add_str = []
    for f in os.listdir(DOC_DIR):
        full_path = os.path.join(DOC_DIR, f)
        a, _ = f.split(".")
        add_str += [f"   {a}\n"]
        dest_path = f"../Docs/{f}"
        copyfile(full_path, dest_path)

    s_ind = ":caption: Contents:"
    with open("../Docs/index.rst", "r") as f:
        data = f.read()
        d1, d2 = data.split(s_ind)

    with open("../Docs/index.rst", "w") as f:
        f.write(d1 + s_ind + "\n\n" + "".join(add_str) + "   " + d2.lstrip())


if __name__ == '__main__':
    main()
