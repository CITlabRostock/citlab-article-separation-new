class PageXmlException(Exception):
    pass


def format_custom_attr(ddic):
    """
    Format a dictionary of dictionaries in string format in the "custom attribute" syntax
    e.g. custom="readingOrder {index:1;} structure {type:heading;}"
    """
    s = ""
    for k1, d2 in ddic.items():
        if s:
            s += " "
        s += "%s" % k1
        s2 = ""
        for k2, v2 in d2.items():
            if s2:
                s2 += " "
            s2 += "%s:%s;" % (k2, v2)
        s += " {%s}" % s2
    return s

