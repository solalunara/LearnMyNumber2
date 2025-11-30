import sys

if __name__ == '__main__':
    file = sys.argv[ 1 ]
    with open( file ) as f:
        data_str = f.read()
    new_data_str = "";
    for char in data_str:
        new_data_str = new_data_str + char + '\n'
    
    outfile = sys.argv[ 2 ]
    with open( outfile, 'w' ) as w:
        w.write( new_data_str )