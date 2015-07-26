
cdef extern from "../bm3d_src/mt19937ar.h":
    double mt_genrand_res53()


def hello():
    return "Hello World"

def random():
    return mt_genrand_res53()
