from collections import defaultdict, Hashable


def filter_by_attribute(lst, attr):
    """ Given a list of elements/objects `lst`, convert to a list of lists according to a specific attribute `attr`.

    :param lst: A list holding elements/objects.
    :param attr: An attribute of the objects in the list to filter.
    :return: A list of lists holding the objects with the same specific attribute.
    """
    if not check_type(lst, type(lst[0])):
        raise ValueError('All items in the list should have the same type.')

    d = defaultdict(list)
    for el in lst:
        if not hasattr(el, attr):
            raise ValueError("At least one item in the list doesn't have the requested attribute.")
        key = getattr(el, attr)
        if not isinstance(key, Hashable):
            if len(key) == 1:
                key = next(iter(key))
            elif len(key) == 0:
                key = "blank"
            else:
                raise TypeError('Key must be hashable, but got {} of type {}.'.format(key, type(key)))
        d[key].append(el)

    return dict(d)


def check_type(lst, _type):
    return all([isinstance(el, _type) for el in lst])


class MyClass:
    def __init__(self, idx):
        self.idx = idx


if __name__ == '__main__':
    o1 = MyClass(2)
    o2 = MyClass(2)
    o3 = MyClass(3)

    l = [o1, o2, o3]

    _dict = filter_by_attribute(l, "idx")
    print(_dict)
    list_of_list = list(_dict.values())
    print(list_of_list)
    for _l in list_of_list:
        for o in _l:
            print(o.idx)
