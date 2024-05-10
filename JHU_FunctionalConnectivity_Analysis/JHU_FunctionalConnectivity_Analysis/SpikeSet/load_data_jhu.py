import ast
import os
import numpy as np
import pandas as pd
import h5py
from collections import defaultdict

def load_h5_to_pd_jhu(filename):
    f = h5py.File(filename, 'r')
    keys = f.keys() #save the keys from the data file
    wells = np.array(f['wells'])
    electrodes=[]
    df=[]
    i=0
    #since there are multiple wells per data file, i make a list of dfs per well
    for well in wells:
        try:
            recordings=f['recordings']['rec0000'][well]['spikes'] #frame, channel and amplitude data
            maps = np.array(f['recordings']['rec0000'][well]['settings']['mapping']) #electrode + x/y data
            maps=pd.DataFrame(maps.tolist())
            maps.columns=['channel', 'electrode', 'x', 'y'] 
            df.append(pd.DataFrame(np.array(recordings).tolist()))
            df[i].columns=['frame','channel','amplitude']
            df[i]['frame'] = df[i]['frame'] - df[i]['frame'].iloc[0]
            df[i] = df[i].merge(maps, on = 'channel', how = 'left') 
            #decode electrode data from bytes to dict:
            byte_str=f['assay']['inputs']['electrodes'][0]
            dict_str = byte_str.decode("UTF-8")
            electrodes.append(ast.literal_eval(dict_str)['electrodes'])
        except:
            print('Well' +str(i)+' does not exist')
            pass

        i+=1
    return df,keys,electrodes,f

def h5_to_pd_CCL(filename):
    f = h5py.File(filename, 'r')
    maps = np.array(f['mapping'])
    maps=pd.DataFrame(maps.tolist())
    #print(len(maps))
    maps.columns=['channel', 'electrode', 'x', 'y']
    proc0 = f['proc0']
    #print(len(proc0))
    ar = np.array(proc0['spikeTimes'])
    #print(len(ar))
    spikes=pd.DataFrame(ar.tolist())
    #print(len(spikes))
    spikes.columns=['frame', 'channel', 'amplitude']
    df = spikes.merge(maps, on = 'channel', how = 'left')
    df['frame'] = df['frame'] - df['frame'].iloc[0]
    return df

def save_h5_files_from_tags(tags, chipsToExclude=['11613' , '11623', '11618', '11597' , '7890' , '8790' , '7278' , '7282a' , '7282']):
    """
    Save h5s from loaded tags. Includes which chip IDs to exclude
    """
    print('Saving h5s')
    New_tags = defaultdict(dict)
    arrays = {}
    filenames= []
    k=0 
    chip =[]
    date =[]
    session_num =[]
    for dirs in [1,2,3,4]:
        if dirs ==1:
            dir = '../labnuc1/trace/'
        elif dirs ==2:
            dir = '../labnuc2/trace/'
        elif dirs == 3:
            dir = '../labnuc3/trace/'
        else:
            dir = '../aws-s3/trace/'
        for filename in os.listdir(dir): 
                if filename.endswith('.events.txt'):
                    filenames.append(filename)
                    arrays[filename] = open(os.path.join(dir, filename), 'r')
                    current_file = arrays[filename]
                    tocheck_file = current_file.read()

                    chip.append(filename.split('.')[0])
                    date.append(filename.split('.')[1])
                    session_num.append(filename.split('.')[2])
                    str = filename.replace('.events.txt', "")

                    if 'experiment tag' in tocheck_file:
                        for tag in tags:
                            if (tocheck_file.split('\n')[4]).split(':')[2][1:] == tag:
                                if not filename.split('.')[0] in chipsToExclude:
                                    New_tags[str]['tag'] = tag
                                    New_tags[str]['chip_id'] = filename.split('.')[0]
                                    New_tags[str]['date'] = filename.split('.')[1]
                                    New_tags[str]['session_num'] = filename.split('.')[2]
                                    k=k+1     

                    current_file.close()
    New_tags_filenames = {k+'.spikes.bin': v for k, v in New_tags.items()}
    #checker = 
    files=[]
    for dirs in [1,2,3,4]:
        if dirs ==1:
            dir = '../labnuc1/trace/'
        elif dirs ==2:
            dir = '../labnuc2/trace/'
        elif dirs == 3:
            dir = '../labnuc3/trace/'
        else:
            dir = '../aws-s3/trace/'

        for filename in os.listdir(dir): 
            if filename in New_tags_filenames:

                if not filename.startswith('.') and not filename.endswith('.ipynb') and not filename.endswith('.npy')and not filename.endswith('.txt'):
                    with open(os.path.join(dir, filename), 'rb') as file: # open in readonly mode # do your stuff
                        filename = file.name
                        name = filename.split('/')[-1]
                        chip_id = name.split('.')[0]
                        date = name.split('.')[1]
                        session_num = name.split('.')[2]
                        outputname = chip_id + "." + date + "." + session_num + ".h5"
                        files.append({'Filename':filename,'Outputname':outputname})
                        #Need to change paths here manually for now
                        # %run <dishpill Literate/raw_to_h5.py path> <save_path> -p <filename> -m <config file path>

    return files



# def grabTag(name, New_tags_filenames):
#     name = name.rsplit('.', 1)[0]
#     name = name + '.spikes.bin'
    
#     print(name)

#     chip = New_tags_filenames[name]
#     tagName = chip.get('tag')
    
#     return tagName
