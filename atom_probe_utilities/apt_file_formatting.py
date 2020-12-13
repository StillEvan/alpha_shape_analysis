def read_pos(fname):
    """
    Loads an APT .pos file and return a column-based data numpy array.

    Output formate is:
    [[x0, y0, z0, Da0],
     [x1, y1, z1, Da1],
     ...]
    """
    dt = np.dtype('>f4') #default pos file format is big endian byteorder four byte float. (check this in reference book)
    d = np.fromfile(fname, dtype=dt, count=-1) #use numpy from file to read a binary file. The d format is [x0,y0,z0,D0,x1,y1,z1,D1,...]
    data = np.reshape(d, (-1, 4)) # change the data shape to [m rows, 4 cols] from a total data size of 4m.

    print('Unpack finished.')
    return data

def read_epos(fname):
    """
    Loads an APT .epos file and return a column-based data numpy array.

    Output formate is:
    [[x0, y0, z0, Da0, ns0, kV0, X0, Y0, pulse count, ions per pulse],
     [x1, y1, z1, Da1],
     ...]
    """
    dt = np.dtype('>f4') #default pos file format is big endian byteorder four byte float. (check this in reference book)
    d = np.fromfile(fname, dtype=dt, count=-1) #use numpy from file to read a binary file. The d format is [x0,y0,z0,D0,x1,y1,z1,D1,...]
    data = np.reshape(d, (-1, 11)) # change the data shape to [m rows, 10 cols] from a total data size of 10m.

    print('Unpack finished.')
    return data

def read_ato(fname, f_format):
    """
    Loads an APT .ato file and return a column-based data numpy array.

    Output format is:
    [[x0, y0, z0, Da0, ClusterID, PulseNumber, DCkV, ToF, X0, Y0, PulsekV, Vvolt, fourierR, FourierI],
     [x1, y1, z1, Da1],
     ...]
    """
    dt = np.dtype(f_format) #default pos file format is big endian byteorder four byte float. (check this in reference book)
    d = np.fromfile(fname, dtype=dt, count=-1, offset = 8) #use numpy from file to read a binary file. The d format is [x0,y0,z0,D0,x1,y1,z1,D1,...]
    #data = d
    data = np.reshape(d, (-1, 14)) # change the data shape to [m rows, 10 cols] from a total data size of 14m.

    print('Unpack finished.')
    return data

def read_rrng(rrng_fname):
    """
    Loads a .rrng file (IVAS format). Returns an array contains unique 'ions', and an array for 'ranges'.
    Range file format for IVAS is inconsistent among known and unknown range names.

    For known range name (contains only ion names on element table), format is:

        Range1=low_m high_m Vol:some_float ion_species:number (ion_species_2:number ...) Color:hexdecimal_num

    For unknown range names (any name given that not on element table):
        Range1=low_m high_m Vol:some_float Name:some_name Color:hexdecimal_num
    
    rrng_fname:
        filename for the range file.
    
    return:
        (ions, rrng): ions is a numpy array for all ion species in the range file; 
        rrng is a structured numpy array [(low_m, high_m, vol, ion_type,.., color), ...]
        dt = np.dtype([('range_low', '>f4'), ('range_high', '>f4'), ('vol', '>f4'), ('ion_type', 'U16'), ('color', 'U16')])
    """
    with open(rrng_fname, 'r') as f:
        rf = f.readlines()
    
    # pattern is a compiled regular experssion pattern to match range file format. works for both known unknown names.
    # pattern maching for strings: 
    #       r: raw string literal flag, means '\' won't be treated as escape character.
    #       'Ion([0-9]+)=([A-Za-z0-9]+)': it matches the ion types in first section such as ion1=C, ion2=Ti1O1,...
    #                                     'Ion' will match exactly character; (...) means a group, which will be retrived; [0-9] means a set of characters, in
    #                                     this case, any numeric character between 0 to 9; + means one or more repetition of the preceding regular experssion.
    #       '|' means 'or', that is either the precede one or the trailing one is matched.
    #       '-?': to match 0 or more minus sign, just put here for test purpose (some times I use -1 as M/Z ratio for noise points in test data)
    #       '\d+.\d': to match a float number, such as 123.456
    #       '([a-zA-Z0-9:a-zA-Z0-9 ]+)' : to match ion types, such as 'Ti:1 O:1' or 'Name:Cluster1'. Note the last space within square paranthesis is important.
    #       'Color:([A-Za-z0-9]{6})': to match color hexdecimal number. It matches exactly 6 characters.
    pattern = re.compile(r'Ion([0-9]+)=([A-Za-z0-9]+)|Range([0-9]+)=(-?\d+.\d+) (-?\d+.\d+) +Vol:(\d+.\d+) +([a-zA-Z0-9_+:a-zA-Z0-9 ]+) +Color:([A-Za-z0-9]{6})')
        
    elements = []
    rrngs = []
    for line in rf:
        match_re = pattern.search(line)
        if match_re:
            if match_re.groups()[0] is not None:
                elements.append(list(match_re.groups())[1])
            else:
                rrngs.append(match_re.groups()[3:])
                
    dt = np.dtype([('range_low', '>f4'), ('range_high', '>f4'), ('vol', '>f4'), ('ion_type', 'U16'), ('color', 'U16')]) # Note that in python 3 all strings are unicode now.
    rrngs = np.array(rrngs, dtype=dt)
    elements = np.array(elements)
    
    pattern_unknown_ion = re.compile(r'(?<=Name)([A-Za-z0-9]+)') # To obtain correct ion name for unknown ions in range file. Separate the 'Name' from e.g. 'Name:Cluster1'.

    # To further process ion types, reorganize like 'Ti:1 O:1' to 'Ti1O1'.
    for idx in range(len(rrngs)):
        rrngs[idx][3] = rrngs[idx][3].replace(':', '')
        rrngs[idx][3] = rrngs[idx][3].replace(' ', '')

        n = pattern_unknown_ion.search(rrngs[idx][3])
        if n:
            rrngs[idx][3] = n.groups()[0]

    # check the validity of range, there should be no interval overlap.
    sorted_rng_idx = np.argsort(rrngs['range_low'], kind='mergesort')
    range_low = rrngs['range_low']
    range_low = range_low[sorted_rng_idx]
    range_high = rrngs['range_high']
    range_high = range_high[sorted_rng_idx]
    assert np.all(range_low < range_high), 'Invalid range file: range overlap detected.'

    return elements, rrngs[sorted_rng_idx]

# This will assign identities to Mass to Charge Ratios in the pandas data frame
def ion_identification(m2c, rrngs):
    """
    Filter the pos numpy array by range structured array.
    The function acts like a filter, to select ions from pos.

    m2c: a list, array of mass to charge ratios
    
    rng:
        a structured numpy array of format [(low_m, high_m, vol, ion_type,.., color), ...]
        dt = np.dtype([('range_low', '>f4'), ('range_high', '>f4'), ('vol', '>f4'), ('ion_type', 'U16'), ('color', 'U16')])

    Return:
        identity:
            a numpy array the same size as len(m2c).
    """
    # Note that range is pre-sorted during read
    range_low = rrngs['range_low']
    range_high = rrngs['range_high']
    #print(range_high)
    ion_stats = [(rrngs[i]['ion_type'], '#' + rrngs[i]['color']) for i in range(len(rrngs))]
    ion_stats.append(('Noise', '#000000'))
    #print(ion_stats)
    #ion_stats = set(ion_stats)
    #ion_stats = np.array(list(ion_stats))
    ion_stats = np.array(ion_stats)
    #print(ion_stats)
    ion_types = ion_stats[:, 0]
    ion_colors = ion_stats[:, 1]
    # binary search to find insertion idx, which could be used to decide if m/z in pos is within range, where 1 is True and 0 is False
    low_idx = np.searchsorted(range_low, m2c)#, side='right')
    high_idx = np.searchsorted(range_high, m2c)#, side='right')
    #print(m2c[high_idx])
    identity = ion_types[high_idx] # select potential ion idensities based on returned interval index. The contains element that does not fall within actual interval
                                # and will be discarded using the following logic array.
    identity_colors = ion_colors[high_idx]
    logic = np.array(low_idx-high_idx, dtype=bool)
    # Assign ion identity to all points in pos. Outside of range will be assigned 'Noise' type.
    for idx in range(len(identity)):
        if not logic[idx]:
            identity[idx] = 'Noise'
            identity_colors[idx] = '#000000'

    return identity, identity_colors

def write_pos(data, pos_fname):
    """
    Writing a numpy array data to APT readable .pos format.

    data:
        a n by 4 numpy array [[x0,y0,z0,Da0],
                              [x1,y1,z1,Da1],
                              ...]
    pos_fname:
        filename for the output pos.
    """
    if data.shape[1] != 4:
        sys.exit('data must be a numpy array of size m-by-4')

    assert data.dtype == '>f4' # very important, default float datatype will give corrupted result.

    flat_data = np.ndarray.flatten(data)
    flat_data.tofile(pos_fname) # Note to self, format in tofile dose not work for changing data type. 
                                   # So it need an assertation ahead to make sure data type is corrects.
    
    print('Pos file writing finished')   
    return
