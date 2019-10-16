def weighted_mean_of_masked_data(data_in,data_mask,data_cond):
    #data_in = input xarray data to have weighted mean
    #data_mask = nan mask eg. land values
    #LME mask T or F values
    global_attrs = data_in.attrs
    R = 6.37e6 #radius of earth in m
    grid_dy,grid_dx = (data_in.lat[0]-data_in.lat[1]).data,(data_in.lon[0]-data_in.lon[1]).data
    dϕ = np.deg2rad(grid_dy)
    dλ = np.deg2rad(grid_dx)
    dA = R**2 * dϕ * dλ * np.cos(np.deg2rad(ds.lat)) 
    pixel_area = dA.where(data_cond)  #pixel_area.plot()
    pixel_area = pixel_area.where(np.isfinite(data_mask))
    total_ocean_area = pixel_area.sum(dim=('lon', 'lat'))
    data_weighted_mean = (data_in * pixel_area).sum(dim=('lon', 'lat'),keep_attrs=True) / total_ocean_area
    data_weighted_mean.attrs = global_attrs  #save global attributes
    for a in data_in:                      #set attributes for each variable in dataset
        gatt = data_in[a].attrs
        data_weighted_mean[a].attrs=gatt
    return data_weighted_mean

def weighted_mean_of_data(data_in,data_cond):
    import numpy as np
    import xarray as xr
    #data_in = input xarray data to have weighted mean
    #data_mask = nan mask eg. land values
    #LME mask T or F values
    global_attrs = data_in.attrs
    R = 6.37e6 #radius of earth in m
    grid_dy,grid_dx = (data_in.lat[0]-data_in.lat[1]).data,(data_in.lon[0]-data_in.lon[1]).data
    dϕ = np.deg2rad(grid_dy)
    dλ = np.deg2rad(grid_dx)
    dA = R**2 * dϕ * dλ * np.cos(np.deg2rad(data_in.lat)) 
    pixel_area = dA.where(data_cond)  #pixel_area.plot()
    #pixel_area = pixel_area.where(np.isfinite(data_mask))
    sum_data=(data_in*pixel_area).sum(dim=('lon', 'lat'),keep_attrs=True)
    total_ocean_area = pixel_area.sum(dim=('lon', 'lat'))
    #print(sum_data)
    #print(total_ocean_area)
    data_weighted_mean = sum_data/total_ocean_area
    data_weighted_mean.attrs = global_attrs  #save global attributes
    for a in data_in:                      #set attributes for each variable in dataset
        gatt = data_in[a].attrs
        data_weighted_mean[a].attrs=gatt

    return data_weighted_mean


def get_filename(var):
    if (str(var).lower()=='sst') or (var==1):
        file='./utils/data/sst.mnmean.nc'
    if (str(var).lower()=='wind') or (var==2):
        file='./utils/data/wind.mnmean.nc'
    if (str(var).lower()=='current') or (var==3):
        file='./utils/data/cur.mnmean.nc'
    if (str(var).lower()=='chl') or (var==4):
        file='./utils/data/chl.mnmean.nc'
    return file
       
def get_pices_mask():
    import xarray as xr
    filename = './utils/data/PICES/PICES_all_mask360.nc'
    ds = xr.open_dataset(filename)
    ds.close()
    return ds

def get_lme_mask():
    import xarray as xr
    filename = './utils/data/LME/LME_all_mask.nc'
    ds = xr.open_dataset(filename)
    ds.close()
    return ds
    
def get_pices_data(var, ilme, initial_date,final_date):
    import xarray as xr
    import numpy as np
    import os
    
    print(var)
    file = get_filename(var)
    print('opening:',file)
    #print(os.getcwd())
    ds = xr.open_dataset(file)
    ds.close()
    
    #subset to time of interest
    ds = ds.sel(time=slice(initial_date,final_date))   
    
    if (str(var).lower()=='current') or (var==3):  #if current data need to mask
        m=ds.mask.sel(time=slice(initial_date,final_date)).min('time')
        ds = ds.where(m==1,np.nan)
        ds = ds.drop('mask')
       
    #read in pices LME mask
    ds_mask = get_pices_mask()
    #interpolate mask
    mask_interp = ds_mask.interp_like(ds,method='nearest')

    #create mean for pices region
    cond = (mask_interp.region_mask==ilme)
    tem = weighted_mean_of_data(ds,cond)
    data_mean=tem.assign_coords(region=ilme)

    #make climatology and anomalies using .groupby method
    data_climatology = data_mean.groupby('time.month').mean('time',keep_attrs=True)
    data_anomaly = data_mean.groupby('time.month') - data_climatology
    global_attributes = ds.attrs
    data_anomaly.attrs = global_attributes
    
    return data_mean, data_climatology, data_anomaly

def get_lme_data(var, ilme, initial_date,final_date):
    import xarray as xr
    
    file = get_filename(var)
    #print('opening:',file)]
    ds = xr.open_dataset(file)
    ds.close()
    
    #subset to time of interest
    ds = ds.sel(time=slice(initial_date,final_date))   
    
    #read in mask
    ds_mask = get_lme_mask()
    #interpolate mask
    mask_interp = ds_mask.interp_like(ds,method='nearest')

    #create mean for pices region
    cond = (mask_interp.region_mask==ilme)
    tem = weighted_mean_of_data(ds,cond)
    data_mean=tem.assign_coords(region=ilme)

    #make climatology and anomalies using .groupby method
    data_climatology = data_mean.groupby('time.month').mean('time')
    data_anomaly = data_mean.groupby('time.month') - data_climatology

    return data_mean, data_climatology, data_anomaly

def analyze_PICES_Region(region,var,initial_date,final_date):
      
    import sys
    sys.path.append('./utils/subroutines/')
    from pices import get_pices_data
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.simplefilter('ignore') # filter some warning messages
    lmenames = ['California Current','Gulf of Alaska','East Bering Sea','North Bering Sea','Aleutian Islands','West Bering Sea',
            'Sea of Okhotsk','Oyashio Current','Sea of Japan','Yellow Sea','East China Sea','Kuroshio Current',
            'West North Pacific','East North Pacific']

    # check values
    
    # assign variables
    lmei = region
    lmename = lmenames[lmei-11]
    var = var.lower() # variable name in lower case

    # data aquisition
    dtmean, dtclim, dtanom = get_pices_data(var, lmei, initial_date, final_date)
    
    # extract and assign data
    allvars = dtmean.data_vars
    
    ## wind and currents
    if (var=='wind') or (var=='current'):
        
        # name in dataset
        ok = 0
        for key,val in allvars.items():
            if ok==0:
                nvaru = key  
                ok += 1
            else:
                nvarv = key
        
        # short and long name for variable
        svar = var.lower().capitalize()
        if var == 'wind':
            datasetname = dtmean.attrs['title']
        else:
            datasetname = dtmean.attrs['description']
        units = dtmean[nvaru].attrs['units']
        lvaru = dtmean[nvaru].attrs['long_name']
        lvarv = dtmean[nvarv].attrs['long_name']
        svaru = var.capitalize() +'_U'
        svarv = var.capitalize() +'_V'
            
        ## Data information
        print('\n\nRegion = '+str(lmei)+' - '+lmename)
        print('Data = '+lvaru)
        print('Data = '+lvarv)
        print('Units = '+units)
        print('Period = '+initial_date+' : '+final_date)
        print('Dataset = '+datasetname)
        if var=='wind':
            act='blowing'
        else:
            act='flowing'
        print('\nNote: Sign indicates where the '+svar+' is '+act+' towards to\n')

        # displaying time series data
        plt.figure(figsize=(10,4))
        plt.plot(dtmean.time,dtmean[nvaru],label=svaru, alpha=0.8)
        plt.plot(dtmean.time,dtmean[nvarv],label=svarv, alpha=0.8)
        if (np.sign(dtmean[nvaru].min())!=np.sign(dtmean[nvaru].max())) or (np.sign(dtmean[nvarv].min())!=np.sign(dtmean[nvarv].max())):
            plt.axhline(color='k',zorder=0)
        plt.grid(True)
        plt.ylabel(svar+' ('+units+')')
        plt.title(lmename+' '+svar+' values')
        plt.legend(loc=0,fontsize='small')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.show()
        
        # display climatology
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.stem(dtclim.month, dtclim[nvaru],markerfmt='d')
        plt.grid(True)
        plt.ylabel(svaru+' ('+units+')')
        plt.xticks(range(1,13),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],rotation=45)
        plt.title(lmename+' '+svaru+' climatology')
        if (np.sign(dtclim[nvaru].min())!=np.sign(dtclim[nvaru].max())):
            plt.axhline(color='k',zorder=0)
        plt.subplot(1,2,2)
        plt.stem(dtclim.month, dtclim[nvarv],markerfmt='d')
        plt.grid(True)
        plt.ylabel(svarv+' ('+units+')')
        plt.xticks(range(1,13),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],rotation=45)
        plt.title(lmename+' '+svarv+' climatology')
        if (np.sign(dtclim[nvarv].min())!=np.sign(dtclim[nvarv].max())):
            plt.axhline(color='k',zorder=0)
        plt.tight_layout()
        plt.show()
    
        ## display statistics
        print('\nMean '+svaru+' value = ', round(dtmean[nvaru].values.mean(),2),units)
        print('Median '+svaru+' value = ', round(np.median(dtmean[nvaru].values),2),units)
        print(svaru+' Standard deviation = ', round(dtmean[nvaru].values.std(),2),units)
        print('\n')
        print('Maximum '+svaru+' value = ', round(dtmean[nvaru].values.max(),2),units)
        print('Minimum '+svaru+' value = ', round(dtmean[nvaru].values.min(),2),units)
        print('\n')
        print('Maximum '+svaru+' anomalies value = ', round(dtanom[nvaru].values.max(),2),units)
        print('Minimum '+svaru+' anomalies value = ', round(dtanom[nvaru].values.min(),2),units)

        ## display statistics
        print('\n\nMean '+svarv+' value = ', round(dtmean[nvarv].values.mean(),2),units)
        print('Median '+svarv+' value = ', round(np.median(dtmean[nvarv].values),2),units)
        print(svarv+' Standard deviation = ', round(dtmean[nvarv].values.std(),2),units)
        print('\n')
        print('Maximum '+svarv+' value = ', round(dtmean[nvarv].values.max(),2),units)
        print('Minimum '+svarv+' value = ', round(dtmean[nvarv].values.min(),2),units)
        print('\n')
        print('Maximum '+svarv+' anomalies value = ', round(dtanom[nvarv].values.max(),2),units)
        print('Minimum '+svarv+' anomalies value = ', round(dtanom[nvarv].values.min(),2),units)

        # display density plots
        plt.figure(figsize=(10,9))
        plt.subplot(2,2,1)
        sns.distplot(dtmean[nvaru], hist=True, kde=True, bins=30,
                     kde_kws={'linewidth': 2})
        plt.title(svaru+' values density plot')
        plt.grid(True)
        plt.xlabel(svaru+' ('+units+')')
        plt.subplot(2,2,2)
        sns.distplot(dtanom[nvaru], hist=True, kde=True, bins=30,
                     kde_kws={'linewidth': 2})
        plt.title(svaru+' anomalies density plot')
        plt.grid(True)
        plt.xlabel(svaru+' ('+units+')')
        
        plt.subplot(2,2,3)
        sns.distplot(dtmean[nvarv], hist=True, kde=True, bins=30,
                     kde_kws={'linewidth': 2})
        plt.title(svarv+' values density plot')
        plt.grid(True)
        plt.xlabel(svarv+' ('+units+')')
        plt.subplot(2,2,4)
        sns.distplot(dtanom[nvarv], hist=True, kde=True, bins=30,
                     kde_kws={'linewidth': 2})
        plt.title(svarv+' anomalies density plot')
        plt.grid(True)
        plt.xlabel(svarv+' ('+units+')')
        plt.tight_layout()
        plt.show()
        
        # display anomalies
        plt.figure(figsize=(12,8),dpi=180)
        plt.subplot(2,1,1)
        p=dtanom.where(dtanom[nvaru]>=0)
        n=dtanom.where(dtanom[nvaru]<0)
        plt.bar(p.time.values,p[nvaru], width=30, color='darkred',alpha=0.8, edgecolor=None,zorder=2)
        plt.bar(n.time.values,n[nvaru], width=30, color='darkblue',alpha=0.8, edgecolor=None,zorder=3)
        plt.grid(True,zorder=1)
        plt.ylabel(svaru+' ('+units+')')
        plt.title(lmename+' '+svaru+' anomalies')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.axhline(color='k',zorder=1)
        plt.subplot(2,1,2)
        p=dtanom.where(dtanom[nvarv]>=0)
        n=dtanom.where(dtanom[nvarv]<0)
        plt.bar(p.time.values,p[nvarv], width=30, color='darkred',alpha=0.8, edgecolor=None,zorder=2)
        plt.bar(n.time.values,n[nvarv], width=30, color='darkblue',alpha=0.8, edgecolor=None,zorder=3)
        plt.grid(True,zorder=1)
        plt.axhline(color='k',zorder=1)
        plt.ylabel(svarv+' ('+units+')')
        plt.title(lmename+' '+svarv+' anomalies')
        plt.autoscale(enable=True, axis='x', tight=True)
        #save anomalies
        plt.savefig('./User_Data_And_Figures/PICESregion'+str(lmei)+'_'+svar+'_anomalies_'+initial_date+'_'+final_date+'.png')
        plt.tight_layout()
        plt.show()
        print('Anomalies calculated based on the entire data period')
        
         # build data set and save
        dta  ={'Year':pd.to_datetime(dtanom.time.values).year.values,'Month':pd.to_datetime(dtanom.time.values).month.values,svaru:dtanom[nvaru].values,svarv:dtanom[nvarv].values}
        df = pd.DataFrame(data=dta)
        df.to_csv('./User_Data_And_Figures/PICESregion'+str(lmei)+'_'+svar+'_anomalies_'+initial_date+'_'+final_date+'.csv')
        
    ## SST and Chl
    else:
        # name in dataset
        for key,val in allvars.items():
            nvar = key
        # short and long name for variable
        svar = var.upper()
        if svar=='SST':
            lvar = dtmean[nvar].attrs['long_name']
        elif svar=='CHL':
            lvar = dtmean.attrs['parameter']

        datasetname = dtmean.attrs['title']

        units = dtmean[nvar].attrs['units']    

        ## Data information
        print('\n\nRegion = '+str(lmei)+' - '+lmename)
        print('Data = '+lvar)
        print('Units = '+units)
        print('Period = '+initial_date+' : '+final_date)
        print('Dataset = '+datasetname)

        # displaying time series data
        plt.figure(figsize=(10,4))
        plt.plot(dtmean.time,dtmean[nvar])
        plt.grid(True)
        plt.ylabel(svar+' ('+units+')')
        plt.title(lmename+' '+svar+' values')
        plt.autoscale(enable=True, axis='x', tight=True)
        if (np.sign(dtmean[nvar].min())!=np.sign(dtmean[nvar].max())):
            plt.axhline(color='k',zorder=0)
        plt.show()

        # display climatology
        plt.figure(figsize=(5,4))
        plt.plot(dtclim.month, dtclim[nvar],'+-',color='k')
        plt.grid(True)
        plt.ylabel(svar+' ('+units+')')
        plt.xticks(range(1,13),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],rotation=45)
        plt.title(lmename+' '+svar+' climatology')
        if (np.sign(dtclim[nvar].min())!=np.sign(dtclim[nvar].max())):
            plt.axhline(color='k',zorder=0)
        plt.show()

        ## display statistics
        print('\nMean '+svar+' value = ', round(dtmean[nvar].values.mean(),2),units)
        print('Median '+svar+' value = ', round(np.median(dtmean[nvar].values),2),units)
        print(svar+' Standard deviation = ', round(dtmean[nvar].values.std(),2),units)
        print('\n')
        print('Maximum '+svar+' value = ', round(dtmean[nvar].values.max(),2),units)
        print('Minimum '+svar+' value = ', round(dtmean[nvar].values.min(),2),units)
        print('\n')
        print('Maximum '+svar+' anomalies value = ', round(dtanom[nvar].values.max(),2),units)
        print('Minimum '+svar+' anomalies value = ', round(dtanom[nvar].values.min(),2),units)

        # display density plots
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        sns.distplot(dtmean[nvar], hist=True, kde=True, bins=30,
                     kde_kws={'linewidth': 2})
        plt.title(svar+' values density plot')
        plt.grid(True)
        plt.xlabel(svar+' ('+units+')')
        plt.subplot(1,2,2)
        sns.distplot(dtanom[nvar], hist=True, kde=True, bins=30,
                     kde_kws={'linewidth': 2})
        plt.title(svar+' anomalies density plot')
        plt.grid(True)
        plt.xlabel(svar+' ('+units+')')
        plt.show()

        # display anomalies
        plt.figure(figsize=(12,4),dpi=180)
        p=dtanom.where(dtanom>=0)
        n=dtanom.where(dtanom<0)
        plt.bar(p.time.values,p[nvar], width=30, color='darkred',alpha=0.8, edgecolor=None,zorder=2)
        plt.bar(n.time.values,n[nvar], width=30, color='darkblue',alpha=0.8, edgecolor=None,zorder=3)
        plt.grid(True,zorder=1)
        plt.axhline(color='k',zorder=0)
        plt.ylabel(svar+' ('+units+')')
        plt.title(lmename+' '+svar+' anomalies')
        plt.autoscale(enable=True, axis='x', tight=True)
        # save anomalies
        plt.savefig('./User_Data_And_Figures/PICESregion'+str(lmei)+'_'+svar+'_anomalies_'+initial_date+'_'+final_date+'.png')
        plt.show()
        print('Anomalies calculated based on the entire data period')
        
        # build data set and save
        dta  ={'Year':pd.to_datetime(dtanom.time.values).year.values,'Month':pd.to_datetime(dtanom.time.values).month.values,svar:dtanom[nvar].values}
        df = pd.DataFrame(data=dta)
        df.to_csv('./User_Data_And_Figures/PICESregion'+str(lmei)+'_'+svar+'_anomalies_'+initial_date+'_'+final_date+'.csv')
        