import pandas as pd
import numpy as np
def wells(fileName = None,cols= ["E01","E02","E03"]):
    df = pd.read_csv(fileName) # read csv file that was defined in previous block
    dft = df.transpose() # transpose data
    df = dft 
    df.columns = df.iloc[0] # replace header with first row
    #df.drop(['Well','Content'])
    #df.loc["Well"]
    # stupid nan, I'm grabbing the column by using the name of the first column
    nameFirstCol = df.columns[9]
    print(nameFirstCol)
    wavelengths = np.array ( df[ nameFirstCol  ]  )
    wavelengths = wavelengths[2:,2].astype(float) # still has E01/Sample in first two entries, so skipping here
    well_condition = cols 
    print(wavelengths)
    print( np.shape( wavelengths ))  # create array to hold data
    numValues = np.shape( wavelengths )[0]
    numCtls = len(well_condition) # since its a string 

    data = np.zeros([numCtls,numValues])
    print(np.shape(data))
    for i, label in enumerate( well_condition ):
        print(i,label)

        idata = np.array( df[label])
        data[i,] = idata[2:]


    well_condition_avg = np.average(data,axis=0) # average the triplicates
    well_condition_standard= np.std(data,axis=0) #Standard deviation
    output= {"wavelengths":wavelengths,
             "average": well_condition_avg, 
             "standard_deviation": well_condition_standard}
    return output