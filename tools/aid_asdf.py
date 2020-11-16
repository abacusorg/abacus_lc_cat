import asdf
import numpy as np
import os

# save light cone catalog
def save_asdf(table,filename,header,cat_lc_dir,z_in,i_chunk=None):
    # cram into a dictionary
    data_dict = {}
    for j in range(len(table.dtype.names)):
        field = table.dtype.names[j]
        data_dict[field] = table[field]
        
    # create data tree structure
    data_tree = {
        "data": data_dict,
        "header": header,
    }

    # save the data and close file
    output_file = asdf.AsdfFile(data_tree)
    if i_chunk is not None:
        output_file.write_to(os.path.join(cat_lc_dir,"z%.3f"%z_in,filename+".%d.asdf"%(z_in,i_chunk)))
    else:
        output_file.write_to(os.path.join(cat_lc_dir,"z%.3f"%z_in,filename+".asdf"%(z_in)))
    output_file.close()
