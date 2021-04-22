import jpype


def get_java_util_object():
    return jpype.JPackage("citlab_python_util.external.java").Util()


def test():
    jpype.startJVM(jpype.getDefaultJVMPath())
    java_util_obj = get_java_util_object()
    java_util_obj.testPrint()
    jpype.shutdownJVM()


if __name__ == '__main__':
    jpype.startJVM(jpype.getDefaultJVMPath())
    java_util_obj = get_java_util_object()
    java_util_obj.testPrint()
    jpype.shutdownJVM()
