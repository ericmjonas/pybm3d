from nose.tools import * 


def test_import():
    
    import pybm3d
    assert_equal(pybm3d.bm3d.hello(), "hello world")
