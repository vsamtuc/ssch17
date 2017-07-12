from math import sqrt
import numpy as np


wcup_native_type = np.dtype([
    ('timestamp','>u4'),
    ('clientID','>u4'),
    ('objectID','>u4'),
    ('size','>u4'),
    ('method','u1'),
    ('status','u1'),
    ('type','u1'),
    ('server','u1')
    ])

def read_wcup_native(fname):
    """ 
    Read a file of native WordCup data and return an array. 
    """
    return np.fromfile(fname, wcup_native_type, -1, '')


class hash_family:
        def __init__(self, depth):
                self.depth = depth
                self.F = np.random.randint(0, 1<<63-1, size=(6,depth), dtype=np.int64)

        @staticmethod
        def hash31(a,b,x):
                r = a*x+b
                return ((r>>31)^r) & 2147483647

        def hash(self, x):
                F = self.F
                return self.hash31(F[0], F[1], x)

        def fourwise(self, x):
                F = self.F
                return 2*(((
                        self.hash31(
                                self.hash31(
                                        self.hash31(x,F[2],F[3]),
                                        x,F[4]),
                                x,F[5])

                        ) & 32768)>>15)-1


        _cache = {}

        @staticmethod
        def get_cached(d):
                hf = hash_family._cache.get(d)
                if hf is None:
                        hf = hash_family(d)
                        hash_family._cache[d] = hf
                return hf


class projection:
        def __init__(self, width, depth=None, hf=None):
                if hf is None:
                        assert depth is not None
                        self.depth = depth
                        self.hf = hash_family.get_cached(depth)
                else:
                        self.hf = hf
                        self.depth = hf.depth
                self.width = width

        def __eq__(self, other):
                return self.hf is other.hf and self.width==other.width
        def __ne__(self, other):
                return not (self.__eq__(other))
        def __hash__(self):
                return hash(self.hf)^self.width

        def epsilon(self):
                return 4./sqrt(self.width)
        def prob_failure(self):
                return 0.5**(self.depth/2)



class sketch:
        def __init__(self, proj):
                self.proj = proj
                self.vec = np.zeros((proj.depth, proj.width))
                self.pos = None
                self.delta = None

        def update(self, key, freq = 1):
                self.pos = self.proj.hf.hash(key) % self.proj.width
                self.delta = self.proj.hf.fourwise(key)*freq
                self.vec[range(self.proj.depth), self.pos] += self.delta

def row_dot(vec1, vec2):
        return np.einsum('ij,ij->i',vec1, vec2)

def sk_inner(sk1, sk2):
        return np.median(row_dot(sk1.vec, sk2.vec))

#
# Testing code
#

from collections import Counter as sparse

def make_stream(nkeys, length):
        return np.random.randint(nkeys, size=length)

def make_sparse(S):
        return sparse(S)

def sparse_inner(s1, s2):
        return sum(s1[k]*s2[k] for k in s1)

def make_sketch(proj,sp):
        sk = sketch(proj)
        for x in sp:
                sk.update(x, sp[x])
        return sk

def test_sk_inner():
        proj = projection(500,11)
        sk1 = sketch(proj)
        sk2 = sketch(proj)
        sk1.update(243521,1)
        sk2.update(243521,1)
        assert sk_inner(sk1,sk2)==1


def test_sk_est():
        proj = projection(1500,7)
        print("sketch accuracy = ",proj.epsilon())

        S1 = make_stream(10000, 10000)
        S2 = make_stream(1000, 10000)
        #S2=S1

        sp1 = make_sparse(S1)
        sp2 = make_sparse(S2)
        sk1 = make_sketch(proj, sp1)
        sk2 = make_sketch(proj, sp2)
        
        exc = sparse_inner(sp1,sp2)
        nsp1 = sqrt(sparse_inner(sp1,sp1))
        nsp2 = sqrt(sparse_inner(sp2,sp2))

        cossim = exc/(nsp1*nsp2)
        print("similarity=", cossim)

        est = sk_inner(sk1, sk2)
        err= abs((exc-est)/exc)

        print("error=",err," exc=",exc," est=",est)
        #return v1, sk1
        assert err < proj.epsilon(), "bad accuracy %f"%err


test_sk_est()
