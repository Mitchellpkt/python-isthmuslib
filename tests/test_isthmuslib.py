from isthmuslib.cli import main
import isthmuslib as isli

def test_main():
    assert main([]) == 0

def test_version():
    assert isli.version() == '0.0.6'

def test_handystrings():
    assert isli.handystrings("parent directory") == "sys.path.append(os.path.abspath('../'))"