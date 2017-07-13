from math import sqrt
from collections import Counter as frequency_vector
import numpy as np

# This is the record type for reading WorldCup datasets from disk
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
    Read a file of native WordCup data and return an array of records
    """
    return np.fromfile(fname, wcup_native_type, -1, '')


def make_random_data(nkeys, length):
    """
    Create a random data vector with `nkeys` distinct keys and
    `length` elements.
    """
    return np.random.randint(nkeys, size=length)


def make_gaussian_data(sigma, length):
    """
    Create a gaussian random data vector, with `sigma` standard deviation and `length` elements.
    
    The generated values will all be nonnegative integers.
    """
    v = np.random.normal(0, sigma, length)
    v -= np.min(v)
    return v.astype(np.int)


class hash_family:
    """
    A family of hash functions.

    Instances of this class are used to compute hashes for 
    integer-valued keys. 

    The depth of a hash family is the number of hash functions
    encapsulated in the family.
    """

    def __init__(self, depth):
        """
        Create a hash family of the given depth
        """ 
        self.depth = depth
        self.F = np.random.randint(0, 1<<63-1, size=(6,depth), dtype=np.int64)


    @staticmethod
    def hash31(a,b,x):
        """
        A convenience static function
        """     
        r = a*x+b
        return ((r>>31)^r) & 2147483647

    def hash(self, x):
        """
        Return a vector of hash values for key x.
        """
        F = self.F
        return self.hash31(F[0], F[1], x)

    def fourwise(self, x):
        """
        Return a vector of +1/-1 for key x.

        The values returned are 4-wise independent over keys.
        Thus, they are suitable for use in AMS sketches.
        """
        F = self.F
        return 2*(((
                self.hash31(
                        self.hash31(
                                self.hash31(x,F[2],F[3]),
                                x,F[4]),
                        x,F[5])

                ) & 32768)>>15)-1


    # This private member is used to map depth to hash_family
    # instances. It is used to implement static method get_cached().
    _cache = {}

    @staticmethod
    def get_cached(depth):
        """
        A static method returning the same hash family given depth.
        """    
        hf = hash_family._cache.get(depth)
        if hf is None:
                hf = hash_family(depth)
                hash_family._cache[depth] = hf
        return hf


class projection:
    """
    A projection combines a hash family of given depth, and a width.

    Together, a hash family and a width determine the dimensionality
    reduction performed by a sketch. Compatible sketches must have 
    equal projections.
    """

    def __init__(self, width, depth=None, hf=None):
        """
        Create a new projection.

        Either the depth or the hash family must be provided, or an
        expection is raised.
        """
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

    def hash(self, x):
        """
        Return the hashed cells for a key.

        This method returns the vector of cells (of dimension equal
        to the depth), into which key `x` hashes.
        """
        return self.hf.hash(x) % self.width

    def epsilon(self):
        """
        Return the error guarantee for AMS sketches, for 
        this projection.
        """
        return 4./sqrt(self.width)

    def prob_failure(self):
        """
        Return the probability of failure for AMS sketches.
        """
        return 0.5**(self.depth/2)


class basic_sketch:
    """
    This is a base class for Fast-AGMS and Count-min sketches.
    """

    def __init__(self, proj):
        """
        Construct a new Fast-AGMS sketch with the given projection `proj`
        """
        self.proj = proj
        self.vec = np.zeros((proj.depth, proj.width))




class ams_sketch(basic_sketch):
    """
    Fast-AGMS sketch.
    """

    def update(self, key, freq = 1):
        """
        Add a new key frequency to the sketch.
        """

        pos = self.proj.hash(key)
        delta = self.proj.hf.fourwise(key)*freq
        self.vec[range(self.proj.depth), pos] += delta

    def __add__(self, other):
        """
        Return the sum of two FastAGMS sketches.

        The sketches must have equal projections, or an 
        exception is raised.
        """
        if not isinstance(other, ams_sketch):
            return NotImplemented
        if self.proj!=other.proj:
            raise ValueError("Sketches do not have equal projection")
        sk = ams_sketch(self.proj)
        sk.vec = self.vec+other.vec
        return sk

    def __matmul__(self, other):
        """
        Return the estimated inner product of two AGMS sketches.
        
        This method implements the @ operator, e.g., sk1 @ sk2
        The implementation forwards the result of function
        `ams_sketch_inner(self, other)`.
        """
        if not isinstance(other, ams_sketch):
            return NotImplemented
        return ams_sketch_inner(self, other)


def ams_sketch_inner(sk1, sk2):
    """
    Return the estimated inner product of two AGMS sketches.

    The estimate is the median of the depth-size vector of
    the inner products of corresponding columns., i.e.,
        median { sum_j sk1[i,j]*sk2[i,j]  | i=1,...,depth }
    """
    if sk1.proj!=sk2.proj:
        raise ValueError("Sketches do not have equal projection")
    return np.median(np.einsum('ij,ij->i',sk1.vec, sk2.vec))



class count_min_sketch(basic_sketch):
    """
    Count-Min sketch
    """

    def update(self, key, freq = 1):
        """
        Add a new key frequency to the sketch.
        """
        pos = self.proj.hash(key)
        self.vec[range(self.proj.depth), pos] += freq

    def __matmul__(self, other):
        """
        Return the estimated inner product of two Count-min sketches.
        
        This method implements the @ operator, e.g., sk1 @ sk2
        The implementation forwards the result of function
        `count_min_sketch_inner(self, other)`.
        """
        if not isinstance(other, count_min_sketch):
            return NotImplemented
        return count_min_sketch_inner(self, other)
        
        
def count_min_sketch_inner(sk1, sk2):
    """
    Return the estimated inner product of two Count-min sketches.

    The estimate is the minimum of the depth-size vector of
    the inner products of corresponding columns., i.e.,
        min { sum_j sk1[i,j]*sk2[i,j]  | i=1,...,depth }
    """
    if sk1.proj!=sk2.proj:
        raise ValueError("Sketches do not have equal projection")
    return np.min(np.einsum('ij,ij->i',sk1.vec, sk2.vec))

        


def frequency_vector_inner(s1, s2):
    """
    Return the inner product of two frequency vectors
    """
    return sum(s1[k]*s2[k] for k in s1)


def make_ams_sketch(proj,sp):
    """
    Construct a FastAGMS sketch from a frequency vector
    """
    sk = ams_sketch(proj)
    for x in sp:
        sk.update(x, sp[x])
    return sk

def make_count_min_sketch(proj,sp):
    """
    Construct a FastAGMS sketch from a frequency vector
    """
    sk = count_min_sketch(proj)
    for x in sp:
        sk.update(x, sp[x])
    return sk



#
# Testing code
#



def test_ams_sketch_inner():
    proj = projection(500,11)
    sk1 = ams_sketch(proj)
    sk2 = ams_sketch(proj)
    sk1.update(243521,1)
    sk2.update(243521,1)
    assert ams_sketch_inner(sk1,sk2)==1


def test_sk_est():
    proj = projection(1500,7)
    print("ams_sketch accuracy = ",proj.epsilon())

    S1 = make_random_data(10000, 10000)
    S2 = make_random_data(1000, 10000)

    sp1 = frequency_vector(S1)
    sp2 = frequency_vector(S2)
    sk1 = make_ams_sketch(proj, sp1)
    sk2 = make_ams_sketch(proj, sp2)
    
    exc = frequency_vector_inner(sp1,sp2)
    nsp1 = sqrt(frequency_vector_inner(sp1,sp1))
    nsp2 = sqrt(frequency_vector_inner(sp2,sp2))

    cossim = exc/(nsp1*nsp2)
    print("similarity=", cossim)

    est = ams_sketch_inner(sk1, sk2)
    err= abs((exc-est)/exc)

    print("error=",err," exact=",exc," estimate=",est)
    assert err < proj.epsilon(), "bad accuracy %f"%err

if __name__=='__main__':
    test_sk_est()
