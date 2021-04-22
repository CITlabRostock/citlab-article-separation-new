import argparse
import logging
import sys
from collections import OrderedDict


# Lightweight argparse wrapper


class LineArgumentParser(argparse.ArgumentParser):
    """
    Object for parsing command line strings into Python objects. Inherits from `argparse.ArgumentParser`.

    Overrides the `convert_arg_line_to_args` method, such that each line in a file can contain the argument
    and its values instead of each line only containing a single entry. Argument and values can be seperated
    by spaces or with a ' = '. Anything commented with '#' is ignored.
    """

    def convert_arg_line_to_args(self, arg_line):
        args = arg_line.split()
        for i, arg in enumerate(args):
            # Cut off anything that is commented
            if arg == "#":
                return args[:i]
            # Remove equals sign from args
            if arg == "=":
                args.remove("=")
        return args


# Global object that can be used to access the parser.
usage_string = """%(prog)s [OPTIONS] [CONFIG]
You can add specific options via '--OPTION VALUE'
You can reference config files via '@path/to/config'"""
global_parser = LineArgumentParser(usage=usage_string, fromfile_prefix_chars="@")


class NamespaceWithOrder(argparse.Namespace):
    """
    Object for storing attributes. Inherits from `argparse.Namespace`.

    Implements the `__init__` and `__setattr__` methods, such that any call to `__setattr__`
    saves the (`attr`, `value`) pair in an extra argument `order`, which is an ordered list.
    This is useful if one wants to preserve the information in which order (`argument`, `value`)
    pairs were added to the namespace. This includes default values of arguments, aswell as arguments
    fed via the command line, although it will include duplicates if arguments get fed more than once.
    """

    def __init__(self, **kwargs):
        self.__dict__['order'] = []
        super(NamespaceWithOrder, self).__init__(**kwargs)

    def __setattr__(self, attr, value):
        self.__dict__['order'].append((attr, value))
        super(NamespaceWithOrder, self).__setattr__(attr, value)


class FlagValues(object):
    """Global container and accessor for flags and their values."""

    def __init__(self):
        self.__dict__['__flags'] = {}
        self.__dict__['__parsed'] = False
        self.__dict__['namespace'] = NamespaceWithOrder()

    def parse_flags(self, args=None):
        result, unparsed = global_parser.parse_known_args(args=args, namespace=self.__dict__['namespace'])
        for flag_name, val in vars(result).items():
            self.__dict__['__flags'][flag_name] = val
        self.__dict__['__parsed'] = True
        return unparsed

    def hasKey(self, name_string):
        return name_string in self.__dict__['__flags']

    def __getattr__(self, name):
        """Retrieves the 'value' attribute of the flag --name."""
        if not self.__dict__['__parsed']:
            self.parse_flags()
        if name not in self.__dict__['__flags']:
            raise AttributeError(name)
        return self.__dict__['__flags'][name]

    def __setattr__(self, name, value):
        """Sets the 'value' attribute of the flag --name."""
        if not self.__dict__['__parsed']:
            self.parse_flags()
        self.__dict__['__flags'][name] = value


# Global object that can be used to access the flags.
FLAGS = FlagValues()


def _define_helper(flag_name, default_value, docstring, flagtype, metavar):
    """Registers 'flag_name' with 'default_value', 'docstring' and 'metavar'."""
    global_parser.add_argument('--' + flag_name,
                               default=default_value,
                               help=docstring,
                               type=flagtype,
                               metavar=metavar)


def define_string(flag_name, default_value, docstring, metavar="STR"):
    """
    Defines a flag of type 'string'.

    Args:
        flag_name: `str`, the name of the flag.
        default_value: `str`, the default value the flag should take.
        docstring: `str`, a helpful message explaining the use of the flag.
        metavar: `str`, a name for the argument in usage messages.
    """
    _define_helper(flag_name, default_value, docstring, str, metavar)


def define_integer(flag_name, default_value, docstring, metavar="INT"):
    """
    Defines a flag of type 'int'.

    Args:
        flag_name: `str`, the name of the flag.
        default_value: `int`, the default value the flag should take.
        docstring: `str`, a helpful message explaining the use of the flag.
        metavar: `str`, a name for the argument in usage messages.
    """
    _define_helper(flag_name, default_value, docstring, int, metavar)


def define_float(flag_name, default_value, docstring, metavar="FLOAT"):
    """
    Defines a flag of type 'float'.

    Args:
        flag_name: `str`, the name of the flag.
        default_value: `float`, the default value the flag should take.
        docstring: `str`, a helpful message explaining the use of the flag.
        metavar: `str`, a name for the argument in usage messages.
    """
    _define_helper(flag_name, default_value, docstring, float, metavar)


def define_boolean(flag_name, default_value, docstring, metavar="BOOL"):
    """
    Defines a flag of type 'boolean'.

    Args:
        flag_name: `str`, the name of the flag.
        default_value: `bool`, the default value the flag should take.
        docstring: `str`, a helpful message explaining the use of the flag.
        metavar: `str`, a name for the argument in usage messages.
    """

    # Register a custom function for 'bool' so --flag=True works.
    def str2bool(v):
        return v.lower() in ('true', 't', '1')

    global_parser.add_argument('--' + flag_name,
                               nargs='?',
                               const=True,
                               help=docstring,
                               default=default_value,
                               type=str2bool,
                               metavar=metavar)


# The internal google library defines the following alias, so we match
# the API for consistency.
DEFINE_bool = define_boolean  # pylint: disable=invalid-name


def define_list(flag_name, flag_type, metavar, docstring, default_value=None):
    """
    Defines a flag as a list of multiple entries.

    Args:
        flag_name: `str`, the name of the flag.
        flag_type: the data type to which the list elements should be converted.
        metavar: `str`, a name for the argument in usage messages.
        docstring: `str`, a helpful message explaining the use of the flag.
        default_value: `flag_type`, the default value the flag should take.
    """
    global_parser.add_argument('--' + flag_name,
                               type=flag_type,
                               default=default_value,
                               nargs='*',
                               metavar=metavar,
                               help=docstring)


def define_choices(flag_name, choices, default_value, flag_type, metavar, docstring):
    """
    Defines a flag with predefined choices.

    Args:
        flag_name: `str`, the name of the flag.
        choices: `container`, contains the allowed values for the argument.
        default_value: entry of `choices`, the default value the flag should take.
        flag_type: the data type to which the flag should be converted.
        metavar: `str`, a name for the argument in usage messages.
        docstring: `str`, a helpful message explaining the use of the flag.
    """
    global_parser.add_argument('--' + flag_name,
                               type=flag_type,
                               default=default_value,
                               choices=choices,
                               metavar=metavar,
                               help=docstring)


def define_dict(flag_name, default_value, docstring):
    """
    Defines a flag as dictionary of key-value pairs.

    Args:
        flag_name: `str`, the name of the flag.
        default_value: `dict` of key=value pairs, the default value the flag should take.
        docstring: `str`, a helpful message explaining the use of the flag.
    """
    global_parser.add_argument('--' + flag_name,
                               action=StoreDictKeyPair,
                               default=default_value,
                               nargs="*",
                               metavar="KEY=VAL",
                               help=docstring)


class StoreDictKeyPair(argparse.Action):
    def _is_number(self, string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    def _is_list(self, string):
        s = string.strip()
        try:
            return s[0] == '[' and s[-1] == ']'
        except IndexError:
            return False

    def _str_is_true(self, v):
        return v.lower() in ('true', 't')

    def _str_is_false(self, v):
        return v.lower() in ('false', 'f')

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if not getattr(namespace, self.dest):
            setattr(namespace, self.dest, {})
        for kv in values:
            if len(kv.split('=')) == 2:
                key, val = kv.split("=")
                # convert to type
                if self._str_is_true(val):
                    val = True
                elif self._str_is_false(val):
                    val = False
                elif self._is_number(val):
                    f = float(val)
                    i = int(f)
                    val = i if i == f else f
                elif self._is_list(val):
                    list_vals = val.strip()[1:-1].split(",")  # remove parantheses and split into elements
                    list_vals = [element.strip() for element in list_vals]  # remove leading/trailing spaces of elements
                    val = []
                    for element in list_vals:
                        # convert elements to type
                        if self._str_is_true(element):
                            element = True
                        elif self._str_is_false(element):
                            element = False
                        elif self._is_number(element):
                            f = float(element)
                            i = int(f)
                            element = i if i == f else f
                        elif element == '':
                            continue
                        val.append(element)
                # update the dict
                getattr(namespace, self.dest).update({key: val})


def print_flags():
    """Prints all registered flags in order."""
    order_final = OrderedDict()
    for key, value in FLAGS.order:
        order_final[key] = value

    print("FLAGS:")
    if int(sys.version[0]) < 3:
        flag_list = order_final.items()
    else:
        flag_list = iter(order_final.items())
    for key, value in flag_list:
        print("  {} = {}".format(key, value))


def update_params(class_params, flag_params, name="", print_params=False):
    """update a dictionary holding various parameters using a 'flag' containing a dictionary

    :param class_params: dictionary whose entries will get updated
    :param flag_params: dictionary used for the update (should only contain keys already present in `class_params`)
    :param name: string that will be printed if `print_params`=True
    :param print_params: boolean that decides if the available and updated params should be printed
    :return: updated dictionary `class_params`
    """
    if print_params:
        print("---{}---".format(name))
        print("available {}_params:".format(name))
        for i, j in enumerate(class_params):
            print("  {}: {}".format(j, class_params[j]))

        print("passed FLAGS.{}_params:".format(name))
        for i, j in enumerate(flag_params):
            print("  {}: {}".format(j, flag_params[j]))

    for i in flag_params:
        if i not in class_params:
            logging.critical("Given {0}_params-key '{1}' is not used by {0}-class!".format(name, i))

    class_params.update(flag_params)

    if print_params:
        print("updated {}_params:".format(name))
        for i, j in enumerate(class_params):
            print("  {}: {}".format(j, class_params[j]))

    return class_params
