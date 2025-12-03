from pathlib import Path
import urllib.request
import numpy as np
import time

class RandomNumberGenerator:
    """
    A class which uses random.org to generate random numbers

    Parameters
    ----------
    file_length: int = 1000
        Length of the file to download from random.org. Larger values mean less downloads but more numbers stored in memory. Can be 1-1000
    min: int = 0
        Minimum number for random generation, inclusive 
    max: int = 9
        Maximum number for random generation, inclusive 
    """
    def __init__( self, file_length: int = 1000, min: int = 0, max: int = 9 ):
        self.file_length = file_length
        self.min = min
        self.max = max
        self.i = 0
        self.npdata = np.empty( (0,), dtype=int )

    def next( self, n: int = 1 ):
        """
        Next random variable(s) from parameters
        """
        val = np.empty( (n) )
        n_generated = 0
        while n_generated < n:
            if self.i + ( n - n_generated ) < self.npdata.shape[ 0 ]:
                # No more downloads neccesary to fulfill request
                new_self_i = self.i + n - n_generated
                val[ n_generated: ] = self.npdata[ self.i:new_self_i ]
                self.i = new_self_i
                n_generated = n
            else:
                # Data length insufficient, download a new file and append it, then repeat
                new_n_generated = n_generated + self.npdata.shape[ 0 ] - self.i
                val[ n_generated:new_n_generated ] = self.npdata[ self.i: ]
                n_generated = new_n_generated
                self.refresh_data()

        return val
    
    def refresh_data( self ):
        """
        Get data from the random.org server to read
        """
        url = f'https://www.random.org/integers?num={self.file_length}&min={self.min}&max={self.max}&col=1&base=10&format=plain&rnd=new'
        filedata = None
        while filedata is None:
            try:
                filedata = "1\n2\n3"
                #with urllib.request.urlopen( url ) as url_data:
                #    filedata: str = url_data.read()
            except:
                print( 'Failed to download random data, trying again in 5 seconds...' )
                time.sleep( 5 )
        self.npdata = np.array( filedata.splitlines(), dtype=int )
        self.i = 0


    
