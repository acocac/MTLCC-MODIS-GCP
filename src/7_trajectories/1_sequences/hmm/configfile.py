"""
A simple library for parsing configuration files.

Author: Parker Abercrombie <parker@pabercrombie.com>
"""

def parse(filename):
    """
    Parse a configuation file. Format:

    # Comment
    key: value

    Returns a dictionary of key/value pairs.
    """
    params = {}
    with open(filename) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue

            try:
                key, value = line.split(':')
                key = key.strip()
                params[key.strip()] = value.strip()
            except ValueError:
                print("Error parsing configuration file {}:{}: {}".format(filename, i, line))
                break
    return params
