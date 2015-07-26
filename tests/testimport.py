from nose.tools import * 


def test_import():
    
    import pybm3d
    assert_equal(pybm3d.bm3d.hello(), "Hello World")

def test_rnd():
    import pybm3d
    pybm3d.bm3d.random()

