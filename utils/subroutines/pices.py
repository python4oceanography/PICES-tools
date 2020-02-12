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
    import os
    home_dir=os.getcwd()
    var=str(var).lower()
    
    if (var=='sst') or (var==1):
        file=home_dir+'/utils/data/sst.mnmean.nc'
    if (var=='wind') or (var==2):
        file=home_dir+'/utils/data/wind.mnmean.nc'
    if (var=='current') or (var==3):
        file=home_dir+'/utils/data/cur.mnmean_aviso.nc'
    if (var=='chl') or (var==4):
        file=home_dir+'/utils/data/chl.mnmean.nc'
    if (var=='sla') or (var==5):
        file=home_dir+'/utils/data/adt.mnmean_aviso.nc'  ### !!!! here
    if (var=='adt') or (var==6):
        file=home_dir+'/utils/data/sla.mnmean_aviso.nc'  ### !!!! here
    #if (var=='current_oscar') or (var==7):
     #   file=home_dir+'/utils/data/cur.mnmean.nc'
    return file
       
def get_pices_mask():
    import xarray as xr
    import os
    home_dir=os.getcwd()
    filename = home_dir+'/utils/data/PICES/PICES_all_mask360.nc'
    ds = xr.open_dataset(filename)
    ds.close()
    return ds

def get_lme_mask():
    import xarray as xr
    import os
    home_dir=os.getcwd()
    filename = home_dir+'/utils/data/LME/LME_all_mask.nc'
    ds = xr.open_dataset(filename)
    ds.close()
    return ds
    
def get_pices_data(var, ilme, initial_date,final_date):
    import xarray as xr
    import numpy as np
    import os
   
    if 'current' in var:
        var='current'
    elif 'wind' in var:
        var='wind'
    
    file = get_filename(var)
    #print('opening:',file)
    #print(os.getcwd())
    ds = xr.open_dataset(file)
    ds.close()
    
    if (var=='current') :
        #read in mask from aviso data (to all? or just to currents?)
        file = get_filename(var)
        ds_aviso = xr.open_dataset(file)
        ds_aviso.close()
        
        #apply aviso mask
        ds_aviso2 = ds_aviso.interp(lat=ds.lat,lon=ds.lon)
        for key in ds.data_vars:
            ds[key]=ds[key].where(np.isfinite(ds_aviso2.u[0,:,:]))

    #subset to time of interest
    ds = ds.sel(time=slice(initial_date,final_date))   
       
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

def select_propervar(dtmean, dtclim, dtanom, var):
    import numpy as np
    if (var=='wind_v'):
        dtmean=dtmean.drop('u_mean')
        dtmean=dtmean.rename({'v_mean':'wind_v'})
        dtclim=dtclim.drop('u_mean')
        dtclim=dtclim.rename({'v_mean':'wind_v'})
        dtanom=dtanom.drop('u_mean')
        dtanom=dtanom.rename({'v_mean':'wind_v'})
    elif (var=='wind_u'):
        dtmean=dtmean.drop('v_mean')
        dtmean=dtmean.rename({'u_mean':'wind_u'})
        dtclim=dtclim.drop('v_mean')
        dtclim=dtclim.rename({'u_mean':'wind_u'})
        dtanom=dtanom.drop('v_mean')
        dtanom=dtanom.rename({'u_mean':'wind_u'})
    elif (var=='wind_speed'):
        dtmean['wind_speed']=np.sqrt(dtmean.v_mean**2+dtmean.u_mean**2)
        dtclim['wind_speed']=np.sqrt(dtclim.v_mean**2+dtclim.u_mean**2)
        dtanom['wind_speed']=dtmean['wind_speed'].groupby('time.month')-dtclim['wind_speed']  
        dtmean=dtmean.drop('u_mean')
        dtclim=dtclim.drop('u_mean')
        dtanom=dtanom.drop('u_mean')
        dtmean=dtmean.drop('v_mean')
        dtclim=dtclim.drop('v_mean')
        dtanom=dtanom.drop('v_mean')
    elif (var=='current_v'):
        dtmean=dtmean.drop('u')
        dtmean=dtmean.rename({'v':'current_v'})
        dtclim=dtclim.drop('u')
        dtclim=dtclim.rename({'v':'current_v'})
        dtanom=dtanom.drop('u')
        dtanom=dtanom.rename({'v':'current_v'})
    elif (var=='current_u'):
        dtmean=dtmean.drop('v')
        dtmean=dtmean.rename({'u':'current_u'})
        dtclim=dtclim.drop('v')
        dtclim=dtclim.rename({'u':'current_u'})
        dtanom=dtanom.drop('v')
        dtanom=dtanom.rename({'u':'current_u'})
    elif (var=='current_speed'):
        dtmean['current_speed']=np.sqrt(dtmean.v**2+dtmean.u**2)
        dtclim['current_speed']=np.sqrt(dtclim.v**2+dtclim.u**2)
        dtanom['current_speed']=dtmean['current_speed'].groupby('time.month')-dtclim['current_speed']  
        dtmean=dtmean.drop('u')
        dtclim=dtclim.drop('u')
        dtanom=dtanom.drop('u')
        dtmean=dtmean.drop('v')
        dtclim=dtclim.drop('v')
        dtanom=dtanom.drop('v')
    return dtmean, dtclim, dtanom

def print_var_info(dtmean, var, lmei, lmename, initial_date, final_date):
    # short and long name for variable
    svar = var.upper()
    #print(dtmean.attrs)
    
    if (svar=='SST'):
        print(dtmean)
        lvar = dtmean.attrs['title']
        #units = dtmean.attrs['units']
        units = 'C'
    elif (svar=='SLA') or (svar=='ADT'):
        lvar = dtmean.attrs['comment']
        units = dtmean.attrs['geospatial_vertical_units']    
    elif svar=='CHL':
        lvar = dtmean.attrs['parameter']
        #units = dtmean.attrs['units']  
        units = 'mg m-3'
    else:
        lvar = svar.replace('_',' ')
        svar = lvar
        units='m/s'
        
    datasetname = dtmean.attrs['title']   

    ## Data information
    print('\n\nRegion = '+str(lmei)+' - '+lmename)
    print('Data = '+lvar)
    print('Units = '+units)
    print('Period = '+initial_date+' : '+final_date)
    print('Dataset = '+datasetname)
    
    return svar, units


def make_plot(plot_type, ds, ds2, var, svar, units, lmei, lmename, initial_date, final_date):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import os
    home_dir=os.getcwd()   

    if plot_type == 'timeseries':
        plt.figure(figsize=(10,4))
        plt.plot(ds.time,ds[var])
        plt.grid(True)
        plt.ylabel(svar+' ('+units+')')
        plt.title(lmename+' '+svar+' values')
        plt.autoscale(enable=True, axis='x', tight=True)
        if (np.sign(ds[var].values.min()))!=(np.sign(ds[var].values.max())):
            print('im here')
            plt.axhline(color='k',zorder=0)
        plt.savefig(home_dir+'/User_Data_And_Figures/PICESregion'+
                        str(lmei)+'_'+svar+'_timeseries_'+
                        initial_date+'_'+final_date+'.png')
        plt.show()
                 
    elif plot_type == 'climatology':
        plt.figure(figsize=(5,4))
        plt.plot(ds.month, ds[var],'+-',color='k')
        plt.grid(True)
        plt.ylabel(svar+' ('+units+')')
        plt.xticks(range(1,13),
               ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec'],
                rotation=45)
        plt.title(lmename+' '+svar+' climatology')
        if (np.sign(ds[var].min())!=np.sign(ds[var].max())):
            plt.axhline(color='k',zorder=0)
        plt.savefig(home_dir+'/User_Data_And_Figures/PICESregion'+str(lmei)+'_'+svar+'_climatology_'+initial_date+'_'+final_date+'.png')
        plt.show()
        
    elif plot_type =='density':
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        sns.distplot(ds[var], hist=True, kde=True, bins=30,
                     kde_kws={'linewidth': 2})
        plt.title(svar+' values density plot')
        plt.grid(True)
        plt.xlabel(svar+' ('+units+')')
        plt.subplot(1,2,2)
        sns.distplot(ds2[var], hist=True, kde=True, bins=30,
                     kde_kws={'linewidth': 2})
        plt.title(svar+' anomalies density plot')
        plt.grid(True)
        plt.xlabel(svar+' ('+units+')')
        plt.savefig(home_dir+'/User_Data_And_Figures/PICESregion'+str(lmei)+'_'+svar+'_densityplots_'+initial_date+'_'+final_date+'.png')
        plt.show()
        
    elif plot_type == 'anomalies':
        plt.figure(figsize=(12,4),dpi=180)
        p=ds.where(ds>=0)
        n=ds.where(ds<0)
        plt.bar(p.time.values,p[var], width=30, color='darkred',alpha=0.8, edgecolor=None,zorder=2)
        plt.bar(n.time.values,n[var], width=30, color='darkblue',alpha=0.8, edgecolor=None,zorder=3)
        plt.grid(True,zorder=1)
        plt.axhline(color='k',zorder=0)
        plt.ylabel(svar+' ('+units+')')
        plt.title(lmename+' '+svar+' anomalies')
        plt.autoscale(enable=True, axis='x', tight=True)
        # save anomalies
        plt.savefig(home_dir+'/User_Data_And_Figures/PICESregion'+
                    str(lmei)+'_'+svar+'_anomalies_'+initial_date+
                    '_'+final_date+'.png')
        plt.show()
        print('Anomalies calculated based on the entire data period')
        
def analyze_PICES_Region(region,var,initial_date,final_date):
    import sys
    import os
    home_dir=os.getcwd()
    sys.path.append(home_dir+'/utils/subroutines/')
#    sys.path.append('/home/jovyan/utils/subroutines/')
    from pices import get_pices_data
    import numpy as np
    import pandas as pd
    import warnings
    warnings.simplefilter('ignore') # filter some warning messages
    lmenames = ['California Current','Gulf of Alaska','East Bering Sea',
                'North Bering Sea','Aleutian Islands','West Bering Sea',
                'Sea of Okhotsk','Oyashio Current','R19','Yellow Sea',
                'East China Sea','Kuroshio Current',
                'West North Pacific','East North Pacific']

    # assign variables
    lmei = region
    lmename = lmenames[lmei-11]
    var = var.lower() # variable name in lower case

    # data aquisition
    dtmean, dtclim, dtanom = get_pices_data(var, lmei, initial_date, final_date)
    #print(dtmean)
    
    # extract and assign data
    if ('wind' in var) or ('current' in var):
        # isolate the proper variable
        dtmean, dtclim, dtanom = select_propervar(dtmean, dtclim, dtanom, var)
    
    # print information
    svar, units = print_var_info(dtmean, var, lmei, lmename, initial_date, final_date)

    if var=='chl':
        var='CHL1_mean'
        
    # displaying time series data
    make_plot('timeseries', dtmean, dtanom, var, svar, units, lmei, lmename, initial_date, final_date)

    # display climatology
    make_plot('climatology', dtclim, dtanom, var, svar, units, lmei, lmename, initial_date, final_date)
    
    ## display statistics
    print('\nMean '+svar+' value = ', round(dtmean[var].values.mean(),2),units)
    print('Median '+svar+' value = ', round(np.median(dtmean[var].values),2),units)
    print(svar+' Standard deviation = ', round(dtmean[var].values.std(),2),units)
    print('\n')
    print('Maximum '+svar+' value = ', round(dtmean[var].values.max(),2),units)
    print('Minimum '+svar+' value = ', round(dtmean[var].values.min(),2),units)
    print('\n')
    print('Maximum '+svar+' anomalies value = ', round(dtanom[var].values.max(),2),units)
    print('Minimum '+svar+' anomalies value = ', round(dtanom[var].values.min(),2),units)

    # display density plots
    make_plot('density',dtmean, dtanom, var, svar, units, lmei, lmename, initial_date, final_date)

    # display anomalies
    make_plot('anomalies',dtanom, dtanom, var, svar, units, lmei, lmename, initial_date, final_date)

    # build data set and save
    dta  ={'Year':pd.to_datetime(dtanom.time.values).year.values,'Month':pd.to_datetime(dtanom.time.values).month.values,svar:dtanom[var].values}
    df = pd.DataFrame(data=dta)
    df.to_csv(home_dir+'/User_Data_And_Figures/PICESregion'+str(lmei)+'_'+svar+'_anomalies_'+initial_date+'_'+final_date+'.csv')
        
