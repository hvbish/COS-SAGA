import numpy as np
from astropy.table import Table
import astropy
c_in_km_s = astropy.constants.c.to('km/s').value

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'

import os
import sys
import io

def plot_lines(gal_vhel, gal_name, qso_name, line_info, all_line_files, figname=None, 
               model_filenames=None, vcomps_exclude=None, vmin=-300, vmax=200, reduced_chi2=None): 
    """
    plot all specs out to take a quick look 

    gal_vhel: heliocentric velocity of host galaxy
    
    """
    nline = len(line_info)
    line_list = line_info['name'].values # note that line_info and all_line_files may contain different set of lines
    fontsize = 12
    fig, axes = plt.subplots(nline, 1, sharex=True, figsize=(6, nline*2.5)) # Originally figsize=(3.5, nline*1.2)
    if nline == 1:
        axes = [axes]  # Ensure axes is always a list, even if there is only one subplot
    for iline in range(nline):
        line_name = line_list[iline]
        line_wrest = line_info.query("(name == @line_name)")['wrest'].values[0]
        spec = Table.read(all_line_files[line_name] , format='ascii')
        spec_vhel = (spec['Wave']-line_wrest)/line_wrest*c_in_km_s
        spec_v_in_gal = spec_vhel - gal_vhel # shift to galaxy redshift
        # let's plot the spec out and take a look 
        ax = axes[iline]
        plt_x = spec_v_in_gal
        plt_y = spec['NormFlux']
        plt_ye = spec['NormErr']
        ax.plot(plt_x, plt_y, ds='steps-mid', color='k', lw=1)
        ax.plot(plt_x, plt_ye, ds='steps-mid', color='k', lw=0.5)
        ax.annotate(line_name, (0.05, 0.08), xycoords='axes fraction', fontsize=fontsize)
    
        # find the corresponding voigt model to plot on 
        if model_filenames is not None: 
            model_filename = model_filenames[line_name]
            voigt_model_fluxes = Table.read(model_filename, format='ascii')
            line_name_short = line_name.replace(' ', '')

            model_wave = voigt_model_fluxes['Wave']
            model_vhel = (model_wave - line_wrest)/line_wrest*c_in_km_s
            model_v_in_gal = model_vhel - gal_vhel
            for key in voigt_model_fluxes.colnames: 
                if 'flux_comp' in key: 
                    if 'all' in key: # this is total flux
                        total_flux = voigt_model_fluxes[key]
                        ax.plot(model_v_in_gal, total_flux, color='r', lw=1.2)
                    else: 
                        comp_flux = voigt_model_fluxes[key]
                        ax.plot(model_v_in_gal, comp_flux, color=plt.cm.Blues(0.6), lw=1, ls='--')
    
        # also indicate the range that is masked 
        if vcomps_exclude is not None: 
            if line_name in vcomps_exclude.keys(): 
                for vrange in vcomps_exclude[line_name]: 
                    indv = np.all([plt_x>=vrange[0], plt_x<=vrange[1]], axis=0)
                    ax.plot(plt_x[indv], plt_y[indv], ds='steps-mid', color=plt.cm.Greys(0.3), lw=1.4)
    for ax in axes:
        ax.set_ylim(0, 1.5)
        ax.set_xlim(vmin, vmax)
        ax.axhline(1, ls='--', lw=0.5,color=plt.cm.Greys(0.8))
        ax.set_ylabel('Norm Flux', fontsize=fontsize)
        ax.minorticks_on()
        ax.axvline(0, ls='--', lw=0.5, color=plt.cm.Greys(0.8))
        ax.tick_params(labelsize=fontsize)
    axes[-1].set_xlabel('v_in_gal (km/s)', fontsize=fontsize)
    if reduced_chi2 is not None: 
        axes[0].set_title(r'%s, %s, $\chi^2_{\rm re}=$%s'%(gal_name, qso_name, reduced_chi2), fontsize=fontsize)
    else: 
        axes[0].set_title(f'{gal_name}, {qso_name}', fontsize=fontsize)
    fig.tight_layout()
    if figname is not None: 
        fig.savefig(figname)
        print('Figure saved to: ', figname)

def parse_output_to_astropy_table(voigt_output, popt, gal_vhel, tbname=None):
    """
    This is to take the output from VoigtFit and rewrite it into a more readable astropy 
    table format and save it (if tbname is not None)
    """
    total_ncomps = 0
    for key in voigt_output.keys():
        if 'logN' in key:
            total_ncomps += 1
    
    # record the data
    param_bestfit = Table()
    param_bestfit['ion'] = [' '*5]*total_ncomps
    param_bestfit['z'] = np.zeros(total_ncomps)+np.nan
    param_bestfit['z_e'] = np.zeros(total_ncomps)+np.nan
    param_bestfit['vhel'] = np.zeros(total_ncomps)+np.nan
    param_bestfit['vhel_e'] = np.zeros(total_ncomps)+np.nan
    param_bestfit['v_in_gal'] = np.zeros(total_ncomps)+np.nan
    param_bestfit['v_in_gal_e'] = np.zeros(total_ncomps)+np.nan
    param_bestfit['b'] = np.zeros(total_ncomps)+np.nan
    param_bestfit['b_e'] = np.zeros(total_ncomps)+np.nan
    param_bestfit['flag'] = ['N/A']*total_ncomps
    param_bestfit['logN'] = np.zeros(total_ncomps)+np.nan
    param_bestfit['logN_e'] = np.zeros(total_ncomps)+np.nan

    # get ions that are fitted 
    fitted_ion_list = []
    for key in voigt_output.keys(): 
        if 'logN' in key: 
            fitted_ion_list.append(key.split('_')[1])
    fitted_ion_list = np.unique(fitted_ion_list)

    previous_ion_ncomps = 0
    for ion in fitted_ion_list: 
        # find out how many components are fitted for this ion 
        ion_ncomps = 0
        for key in voigt_output.keys(): 
            if ('logN' in key) and (ion == key.split('_')[1]): 
                ion_ncomps += 1
        for i in range(ion_ncomps):
            ind = i+previous_ion_ncomps
            param_bestfit['ion'][ind] = ion
            param_bestfit['z'][ind] = np.around(voigt_output[f'z{i}_{ion}'].value, decimals=6)
            param_bestfit['z_e'][ind] = np.around(voigt_output[f'z{i}_{ion}'].stderr, decimals=6)
            param_bestfit['b'][ind] = np.around(voigt_output[f'b{i}_{ion}'].value, decimals=2)
            param_bestfit['b_e'][ind] = np.around(voigt_output[f'b{i}_{ion}'].stderr, decimals=2)
            param_bestfit['logN'][ind] = np.around(voigt_output[f'logN{i}_{ion}'].value, decimals=2)
            param_bestfit['logN_e'][ind] = np.around(voigt_output[f'logN{i}_{ion}'].stderr, decimals=2)
        previous_ion_ncomps += ion_ncomps  

    param_bestfit['vhel'] = np.around(param_bestfit['z']*c_in_km_s, decimals=2)
    param_bestfit['vhel_e'] = np.around(param_bestfit['z_e']*c_in_km_s, decimals=2)

    param_bestfit['v_in_gal'] = np.around(param_bestfit['vhel'] - gal_vhel, decimals=1)
    param_bestfit['v_in_gal_e'] = np.around(param_bestfit['vhel_e'], decimals=2)

    param_bestfit.meta['comments'] = [f'chi2={popt.chisqr:.2f}', f'reduced_chi2={popt.redchi:.2f}']
    if tbname is not None: 
        from astropy.io import ascii
        ascii.write(param_bestfit, tbname, overwrite=True, delimiter=',')
        print('Saving best fit parameters to: ', tbname)

    return param_bestfit

# def truncate_kernel(kernel, alpha):
#     kernel = kernel / kernel.sum()
#     ckernel = np.cumsum(kernel)
# 
#     left, right= np.searchsorted(ckernel, [0.5*alpha, 1-0.5*alpha])
#     left = max(left-1, 0)
#     right = min(right, kernel.size-1)
#     return kernel[left:right]


def construct_voigt_profile(line, line_lam, wave, logN, b, z, lsf_file):

    """
    reconstruct voigt profile using component parameters

    Input:
    line: e.g., 'SiIV 1393'
    wave: wavelength array you want to contruct the voigt profile for, in unit of AA
    logN: logrithmic column density value
    b: doppler parameter
    z: redshift
    """
    
    import io
    import sys
    from contextlib import redirect_stdout 
    buffer = io.StringIO()
    # this is to capture the print out of the linetool process
    with redirect_stdout(buffer):
        # first construct a voigt profile according to the input line information
        import astropy.units as u
        from linetools.spectralline import AbsLine
        line_comp = AbsLine(trans=line, verbose=False)
        line_comp.attrib['N'] = 10**logN*u.cm**(-2)
        line_comp.attrib['b'] = b*u.km/u.s
        line_comp.setz(z)
        # set fwhm to a very small value, infinite res 
        line_comp_voigt = line_comp.generate_voigt(wave=wave*u.AA, fwhm=1.)

    voigt_wave = line_comp_voigt.wavelength.value
    voigt_flux = line_comp_voigt.flux
   

    # get the average LSF
    lsf_tb = Table.read(lsf_file, format='ascii', header_start=0)

    # find the closest lsf kernel
    kernel_wvs = np.array([float(key) for key in lsf_tb.colnames])
    ind_w = np.argmin(np.abs(kernel_wvs - line_lam))
    kernel_key = lsf_tb.colnames[ind_w]
    lsf_kernel = lsf_tb[kernel_key]/lsf_tb[kernel_key].sum()

    # convolve theoretical flux with lsf
    from astropy.convolution import convolve
    cos_flux = convolve(voigt_flux, lsf_kernel)
    
    return voigt_wave, cos_flux

def reproduce_voigt_model_fluxes(lines_to_fit_info, param_bestfit, all_line_files, savename, gal_vhel,
                                 plt_vmin=-400, plt_vmax=400, cos_sat_flux_limit=0.2, data_nbin=None, tbname=None): 
    """
    Make the modeled voigt profiles and save the model spectral flux, note that the 
    flux will always be calculated based on the original wavelength array

    lines_to_fit_info: a pandas table with line atomic hata
    param_bestfit: an astropy table that has bestfit param from VoigtFit
    all_line_files: a list of filenames 
    savename: the filename where you save the modeled voigt profile fluxes
    plt_vmin, plt_vmax: the velocity range around the host galaxy
    cos_sat_flux_limit: this is the flux limit of COS, below which we'd consider the line to be saturated
 
    """
    save_filenames = {}
    ions_to_fit = np.unique(lines_to_fit_info['species'].values)
    for fit_ion in ions_to_fit: 
        sub_fit_info = lines_to_fit_info.query('(species == @fit_ion)')
        min_f = np.min(sub_fit_info['f'].values) # we'll use the weakest line (with the smallest f) to calculate flux at line center
        for iline in range(len(sub_fit_info)): 
           # get line information 
           line_name = sub_fit_info['name'].iloc[iline]
           line_lam = sub_fit_info['wrest'].iloc[iline]
           line_f = sub_fit_info['f'].iloc[iline]
           line_name_short = line_name.replace(' ', '')
    
           # get line data file and get model wavelength 
           line_file = all_line_files[line_name]
           line_spec = Table.read(line_file, format='ascii')
           line_wave = line_spec['Wave']
    
           # line spread function 
           cos_lsf_file = determine_cos_lsf_file(line_name, data_nbin=data_nbin)
    
           # a container to store the data 
           result_tb = line_spec.copy()
           
           # go through the best fit parameter table and find the right ion components
           n_comp = 0
           model_flux_total = np.ones(line_wave.size)
           for i in range(len(param_bestfit)):
               if param_bestfit['ion'][i] != fit_ion:
                   continue
               else:
                   comp_z = param_bestfit['z'][i]
                   comp_b = param_bestfit['b'][i]
                   comp_logN = param_bestfit['logN'][i]
                   from voigtfit_mod_cgm import construct_voigt_profile
                   _, cos_flux = construct_voigt_profile(line_name, line_lam, line_wave,
                                                          comp_logN, comp_b, comp_z,
                                                          cos_lsf_file)
                   result_tb[f'flux_comp{n_comp}'] = cos_flux
                   model_flux_total *= cos_flux # combine all line fluxes together 
                   n_comp += 1
                    
                   #if this is the weakest line,  decide whether the line is saturated
                   if line_f == min_f:
                       # print('this is the weakest line:', line_name, line_f, min_f, line_lam) 
                       plt_wmin = line_lam*((plt_vmin+gal_vhel)/c_in_km_s+1)
                       plt_wmax = line_lam*((plt_vmax+gal_vhel)/c_in_km_s+1)
                       indv = np.all([line_wave>=plt_wmin, line_wave<=plt_wmax], axis=0)
                       min_flux = np.around(np.nanmin(cos_flux[indv]), decimals=2)
      
                       if min_flux < cos_sat_flux_limit: 
                           param_bestfit['flag'][i] = '>'
                       else: 
                           param_bestfit['flag'][i] = '='

           # record the total fluxes
           result_tb[f'flux_comp-all'] = model_flux_total
    
           # save result for this line 
           from astropy.io import ascii
           result_tbname = f'{savename}_{line_name_short}_vfit-flux.txt'
           ascii.write(result_tb, result_tbname, overwrite=True)
           print('Save voigt fit fluxes to: ', result_tbname)
           save_filenames[line_name] = result_tbname

    # save the bestfit param
    if tbname is not None: 
        ascii.write(param_bestfit, tbname, overwrite=True, delimiter=',') 

    return save_filenames

def aod_logN_profiles(wave_arr, flux_arr, flux_err_arr, line_lambda, line_f,
                      line_str, aod_vrange=[-200, 500]):
    """
    Calculate apparent column density, EW, and b for an input line arrays

    Input:
    wave_arr: wavelength array, in unit of A
    flux_arr: normalized flux array
    flux_err_arr: normalized error array
    line_lambda: rest wavelength of line
    line_f: oscillator strength of line
    aod_vrange: a wider velocity range for aod profile
    """
    import numpy as np
    import astropy.units as u
    import astropy.constants as const
    aod_profile = {} # library to record data

    vel_arr = ((wave_arr-line_lambda)/line_lambda* const.c).to(u.km/u.s).value # km/s

    # we only care about stuff within some velocity range as indicated by aod_vrange
    vmin, vmax = aod_vrange[0], aod_vrange[1]
    indv = np.all([vel_arr>=vmin, vel_arr<=vmax], axis=0)
    vel_arr = vel_arr[indv]
    wave_arr = wave_arr[indv]
    flux_arr = flux_arr[indv]
    flux_err_arr = flux_err_arr[indv]

    idx_zero = flux_arr == 0
    if len(flux_arr[idx_zero]) != 0:
        flux_arr[idx_zero] = np.nan
        flux_err_arr[idx_zero] = np.nan

    if len(vel_arr) == 0:
        print(r'Not enough data for aod profile in line %s, return {}'%(line_lambda))
        aod_profile = {}
        return aod_profile

    aod_profile['line_str'] = line_str
    aod_profile['wave_arr'] = wave_arr.copy()
    aod_profile['vel_arr'] = vel_arr.copy()
    aod_profile['flux_arr'] = flux_arr.copy()
    aod_profile['flux_err_arr'] = flux_err_arr.copy()

    # if flux is lower than the err, this line is saturated, then
    # we take the error value instead.
    # See line 90-92 in COS-Halos IDL package, Lines/eqwrange.pro
    # this line is necessary for the ullyses sightlines
    flux_less_than_err = flux_arr<=flux_err_arr
    if len(flux_arr[flux_less_than_err]) != 0:
        flux_arr[flux_less_than_err] = flux_err_arr[flux_less_than_err]

    # get the velocity and wave_arr intevals
    dv = vel_arr[1:]-vel_arr[:-1]
    dv = np.concatenate([dv, [dv[-1]]])
    dlambda = wave_arr[1:] - wave_arr[:-1]
    dlambda = np.concatenate([dlambda, [dlambda[-1]]])
    aod_profile['dv_arr'] = dv
    aod_profile['dlambda_arr'] = dlambda

    # get the column density using apparent optical depth method
    # line 145 and 163 in COS-Halos IDL package, Lines/eqwrange.pro
    # flux_arr[flux_arr<=0.] = 0.0001
    tau_v = -np.log(flux_arr) # iflux_arr = I_obs / I_continuum; line 134
    N_v = 3.768e14*tau_v/line_lambda/line_f # atom cm-2/(kms-1); line 145
    aod_profile['tau_arr'] = tau_v.copy()
    aod_profile['Na_arr'] = N_v.copy()

    # get column density err_arror
    # line 160 in COS-Halos IDL package, Lines/eqwrange.pro
    tauerr_v = np.abs(flux_err_arr/flux_arr) # line 158
    Nerr_v = 3.768e14*tauerr_v/line_lambda/line_f  # line 160
    aod_profile['tau_err_arr'] = tauerr_v.copy()
    aod_profile['Na_err_arr'] = Nerr_v.copy()

    return aod_profile

def aod_integrated_values(wave_arr, flux_arr, err_arr, line_lambda, line_f, line_str,
                          aod_vrange=[-200, 500], intg_vrange=[175, 375]):
    """
    Calculate apparent column density, EW, and b for an input line arrays

    Input:
    wave_arr: wavelength array, in unit of A
    flux_arr: normalized flux array
    err_arr: normalized error array
    line_lambda: rest wavelength of a line, in unit of A
    line_f: oscillator strength
    line_str: the number str of a line, such as line_str = 1253 for "SII 1253"
    aod_vrange: a wider velocity range for aod profile, for plotting purpose
    intg_vrange: the velocity range to make aod integration
    Update as of July 12, 2023

    """
    ## initiate a dict to put all info in
    aod_vals = {'line_str': line_str,
                'intg_vrange': intg_vrange}

    import numpy as np
    # check if the input spec is wide enough for integration
    del_w = np.asarray(intg_vrange)/3e5*line_lambda
    wmin = np.nanmin(wave_arr)
    wmax = np.nanmax(wave_arr)
    if (wmin>line_lambda+del_w[0]) or (wmax<line_lambda+del_w[1]):
        print(r'Not enough data for aod integration in line %s, return {}'%(line_lambda))
        aod_vals = {}
        return aod_vals

    # from yztools.aod.aod import aod_logN_profiles
    aod_profile = aod_logN_profiles(wave_arr, flux_arr, err_arr, line_lambda,
                                    line_f, line_str, aod_vrange=aod_vrange)
    vel_arr = aod_profile['vel_arr']
    intg_indv = np.all([vel_arr>=intg_vrange[0], vel_arr<=intg_vrange[1]], axis=0)

    # get centroid velocity, as weighted by apparent optical depth
    # as Zheng+2019
    tau_arr = aod_profile['tau_arr'][intg_indv]
    tau_err_arr = aod_profile['tau_err_arr'][intg_indv]

    # COS-Halos IDL package, Lines/eqwrange.pro
    vel_arr = vel_arr[intg_indv]
    Na_arr = aod_profile['Na_arr'][intg_indv]
    Na_err_arr = aod_profile['Na_err_arr'][intg_indv]
    dv_arr = aod_profile['dv_arr'][intg_indv]
    # Na = np.fabs(np.nansum(Na_arr*dv_arr)) # line 163
    Na = np.nansum(Na_arr*dv_arr)
    Na_err = np.sqrt(np.nansum((Na_err_arr*dv_arr)**2))  # line 164
    aod_vals['Na'] = np.around(Na, decimals=2)
    aod_vals['Na_err'] = np.around(Na_err, decimals=2)

    #print(line_str, aod_vals['Na'], aod_vals['Na_err'])
    # calculate line SNR, decide whether saturated or not
    snr = Na/Na_err
    aod_vals['SNR'] = np.around(snr, decimals=1)
    #print(line_str, snr)
    if snr <3:
        aod_vals['line_flag'] = '<(3sig)'
        logN_3sig = np.log10(3*Na_err)
        #logN_3sig = np.log10(Na + 3*Na_err)
        # define as 3 times the 1sigma error derived for column density assumign optically thin
        aod_vals['logNa_3sig'] = np.around(logN_3sig, decimals=3)
    else:
        if np.nanmax(tau_arr-tau_err_arr) >1:
            aod_vals['line_flag'] = '>'
        else:
            aod_vals['line_flag'] = '='
        logNa = np.log10(Na)
        logNa_err = Na_err/(Na*np.log(10)) # through error propagation
        # note that COS-Halos (line 172 in Lines/eqwrange.pro) use
        # logNerr = np.log10(N+Nerr)-logN # line 172
        aod_vals['logNa'] = np.around(logNa, decimals=3)
        aod_vals['logNa_err'] = np.around(logNa_err, decimals=3)

    ###### get Equivalent width over the same velocity range
    # See Eq. 1 and 5 in ISM review by Savage+1996
    flux_arr = aod_profile['flux_arr'][intg_indv]
    flux_err_arr = aod_profile['flux_err_arr'][intg_indv]
    dlambda_arr = aod_profile['dlambda_arr'][intg_indv]
    ew_A = np.nansum(dlambda_arr*(1-flux_arr)) # A, line 111 in YongIDL/Lines/eqwrange.pro
    ewerr_A = np.sqrt(np.nansum((dlambda_arr*flux_err_arr)**2)) # mA, line 120 in YongIDL/Lines/eqwrange.pro
    N_ew = 1.13e17*ew_A*1000/(line_f*line_lambda**2)
    Nerr_ew = np.abs(1.13e17*ewerr_A*1000/(line_f*line_lambda**2))
    aod_vals['EW_A'] = np.around(ew_A, decimals=4)
    aod_vals['EWerr_A'] = np.around(ewerr_A, decimals=4)

    if N_ew > 0:
        aod_vals['logNa_from_ew'] = np.around(np.log10(N_ew), decimals=3)
        if aod_vals['line_flag'] == '<(3sig)':
            aod_vals['logNa_3sig_from_ew'] = np.around(np.log10(3*Nerr_ew), decimals=3)

    ###### Now adding in velocity and line width information for detected/saturated lines###
    # calculate centroid velocity
    if aod_vals['line_flag'] != '<(3sig)':
        aod_vals = calc_aod_vc_fwhm(aod_vals, aod_profile, intg_vrange)
    else:
        aod_vals['vc'] = np.nan
        aod_vals['vcerr'] = np.nan
        aod_vals['sigma_v'] = np.nan
        aod_vals['sigma_verr'] = np.nan
    return aod_vals

def calc_aod_vc_fwhm(aod_vals, aod_profile, intg_vrange):
    """
    aod_vals: a dict to store vc and fwhm information
    aod_profile: from yztools.aod.aod import aod_logN_profiles
                  aod_profile = aod_logN_profiles(wave_arr, flux_arr, err_arr, line_lambda,
                                    line_f, line_str, aod_vrange=aod_vrange)
    intg_vrange: integration range of velocity you want to calculate vc and fwhm
    return aod_vals
    """
    import numpy as np
    vel_arr = aod_profile['vel_arr']
    intg_indv = np.all([vel_arr>=intg_vrange[0], vel_arr<=intg_vrange[1]], axis=0)

    # get centroid velocity, as weighted by apparent optical depth
    # as Zheng+2019
    tau_arr = aod_profile['tau_arr'][intg_indv]
    tau_err_arr = aod_profile['tau_err_arr'][intg_indv]

    # COS-Halos IDL package, Lines/eqwrange.pro
    vel_arr = vel_arr[intg_indv]
    dv_arr = aod_profile['dv_arr'][intg_indv]

    vc = np.nansum(vel_arr*tau_arr*dv_arr)/np.fabs(np.nansum(tau_arr*dv_arr))
    aod_vals['vc'] = np.around(vc, decimals=2) # apparent optical depth weighted

    # error for vc_tauwt, same as Zheng+2019
    pA = np.nansum(vel_arr*tau_arr*dv_arr)
    sigpA = np.sqrt(np.nansum((vel_arr*tau_err_arr*dv_arr)**2))
    # note that tauerr/tau = Nerr/N
    Na_err = aod_vals['Na_err']
    Na = aod_vals['Na']
    vcerr = np.fabs(vc)*np.sqrt((sigpA/pA)**2 + (Na_err/Na)**2)
    aod_vals['vcerr'] = np.around(vcerr, decimals=2)
    # note that COS-Halos defineds vc as where cumulative tau reaches 50%

    # get sigma v and line width from the profile
    # based on line 138-143; and page 2 in Heckman et al. 2002, ApJ, 577:691
    top_part = np.nansum((vel_arr-vc)**2 * tau_arr * dv_arr)
    bot_part = np.nansum(tau_arr * dv_arr)
    # print(top_part, bot_part)
    sigma_v = np.sqrt(np.abs(top_part)/bot_part)
    aod_vals['sigma_v'] = np.around(sigma_v, decimals=2)
    # doppler_b = np.sqrt(2) * sigma_v

    # find the undertainty for sigma_v and b using error proporgation
    x = np.nansum((vel_arr - vc)**2 * tau_arr * dv_arr)
    y = np.nansum(tau_arr*dv_arr)
    sig_x = np.sqrt(np.nansum((tau_err_arr*dv_arr)**2))
    sig_y = np.sqrt(np.nansum((2*(vel_arr-vc)*vcerr*tau_arr*dv_arr)**2 +
                            ((vel_arr-vc)**2*tau_err_arr*dv_arr)**2))
    err_sigma_v = 1/(2*sigma_v)*np.sqrt((sig_x/x)**2 + (sig_y/y)**2)
    aod_vals['sigma_verr'] = np.around(err_sigma_v, decimals=2)
    # err_b = np.sqrt(2)*err_sigma_v

    return aod_vals

# def bin_spec(x1, y1, bins):
    # import numpy as np
    
    # b1 = np.mgrid[0:len(x1):bins]
    # dg1 = np.digitize(np.mgrid[0:len(x1):1], b1)
    
    # dg_x1 = np.array([np.mean(x1[dg1==j]) for j in np.arange(len(b1)+1)[1:]])
    # dg_y1 = np.array([np.mean(y1[dg1==j]) for j in np.arange(len(b1)+1)[1:]])
    
    # return dg_x1, dg_y1
    # note @ 20240916, this bin method doesn not take into account how to take into account errors

def bin_spec(wavelengths, fluxes, flux_errors, bin_size=3):
    # Ensure all arrays are numpy arrays
    wavelengths = np.array(wavelengths)
    fluxes = np.array(fluxes)
    flux_errors = np.array(flux_errors)

    # Calculate the number of bins
    n_bins = len(wavelengths) // bin_size

    # Reshape the arrays to group by bin_size
    reshaped_wavelengths = wavelengths[:n_bins*bin_size].reshape(-1, bin_size)
    reshaped_fluxes = fluxes[:n_bins*bin_size].reshape(-1, bin_size)
    reshaped_flux_errors = flux_errors[:n_bins*bin_size].reshape(-1, bin_size)

    # Calculate binned wavelengths (take central value of each bin)
    binned_wavelengths = reshaped_wavelengths[:, bin_size//2]

    # Calculate binned fluxes (average of each bin)
    binned_fluxes = np.mean(reshaped_fluxes, axis=1)

    # Calculate binned flux errors (propagate errors)
    binned_flux_errors = np.sqrt(np.sum(reshaped_flux_errors**2, axis=1)) / bin_size

    return binned_wavelengths, binned_fluxes, binned_flux_errors

# let's estimate the (logN, v, b) values for each component
def estimate_init_guess_params(vcomps, lines_to_fit_info, all_line_files, gal_vhel): 
    """
    This is to provide initial guesses on ion velocity components' velocities, Doppler widths, and column densities
    based on input velocity range using the AOD method

    vcomps: a library with velocity component ranges for ions of interested. Velocities in the the galaxy's rest frame
            Example: vcomps = {'SiIV': [[-130, -80], [-80, -20]], 
                               'SiII': [[-130, -80], [-80, -20]]}
    lines_to_fit_info: a pandas table with the atomic information for the lines of interested 
    all_line_files: a library which includes the path to the corresponding line data files 
    gal_vhel: heliocentric velocity of host galaxy 
    """
    # initial guess based on the velocity range specified in vcomps 
    # for the same ion, we just need to use one of the line file for this initial guess
    param_ion = []
    param_vmin = []
    param_vmax = []
    param_z = []  
    param_vhel = [] 
    param_vsys = []
    param_b = []
    param_logN = []
    
    for iline in range(len(lines_to_fit_info)):
        fit_ion = lines_to_fit_info['species'].iloc[iline]
        if fit_ion in param_ion: # for each ion, we just need estimate from one line
            continue
        else: 
            # get line information 
            line_name = lines_to_fit_info['name'].iloc[iline]
            line_lam = lines_to_fit_info['wrest'].iloc[iline]
            line_f = lines_to_fit_info['f'].iloc[iline]
        
            # get line file information 
            fit_file = all_line_files[line_name]
            fit_spec = Table.read(fit_file, format='ascii')
            spec_wave = fit_spec['Wave']
            spec_flux = fit_spec['NormFlux']
            spec_err = fit_spec['NormErr']
    
            # find the velocity range that we'd use to calculate with aod 
            for vrange in vcomps[fit_ion]: 
                # we'll shift the line_lam into the galaxy's rest frame
                gal_z = gal_vhel/c_in_km_s
                line_lam_at_gal_frame = line_lam*(1+gal_z)
                aod_res = aod_integrated_values(spec_wave, spec_flux, spec_err, 
                                                line_lam_at_gal_frame, line_f, line_name,
                                                intg_vrange=vrange, 
                                                aod_vrange=[-500, 500]) # just need to be large enough than vrange
                # absorber aod result 
                try: 
                    abs_logN = np.around(aod_res['logNa'], decimals=2)
                    abs_vsys = np.around(aod_res['vc'], decimals=1) # velocity in the rest frame of the galaxy
                    abs_b = np.around(aod_res['sigma_v']*1.4, decimals=1)
                except: 
                    abs_logN = np.around(aod_res['logNa_3sig'], decimals=2)
                    abs_vsys = np.around(np.mean(vrange), decimals=1) # velocity in the rest frame of the galaxy 
                    abs_b = np.around((vrange[1]-vrange[0])/2./1.7, decimals=1) # only an estimate 
               
                abs_vhel = abs_vsys+gal_vhel # shift to the heliocentric frame 
                abs_z = np.around(abs_vhel/c_in_km_s, decimals=6)
                
                # record data
                param_ion.append(fit_ion)
                param_vmin.append(vrange[0])
                param_vmax.append(vrange[1])
                param_z.append(abs_z)
                param_vhel.append(abs_vhel) 
                param_vsys.append(abs_vsys)
                param_b.append(abs_b)
                param_logN.append(abs_logN)
    
    param_guess = Table()
    param_guess['ion'] = param_ion
    param_guess['vmin'] = param_vmin
    param_guess['vmax'] = param_vmax
    param_guess['z'] = param_z 
    param_guess['vhel'] = param_vhel
    param_guess['v_in_gal'] = param_vsys
    param_guess['b'] = param_b
    param_guess['logN'] = param_logN

    return param_guess

def determine_cos_lsf_file(line_name, data_nbin=None): 
    # this is to determine which LSF file to be used for the vogit profile fitting later
    # doesn't matter if line_name is 'SII1250', 'SII 1250', or 'SII_1250'
    import re
    comps = re.split(r'(\d+)', line_name)
    rough_wrest = float(comps[1])

    if rough_wrest < 1440:
        if data_nbin is not None:  
            cos_lsf_file = f'lsf/cos_lsf_g130m_average_lp1-4_bin{data_nbin}_yz.txt'
        else: 
            cos_lsf_file = 'lsf/cos_lsf_g130m_average_lp1-4_yz.txt'
    else:
        if data_nbin is not None: 
            cos_lsf_file = f'lsf/cos_lsf_g160m_average_lp1-4_bin{data_nbin}_yz.txt'
        else:  
            cos_lsf_file = 'lsf/cos_lsf_g160m_average_lp1-4_yz.txt'
           
    # test if data file exist
    if os.path.isfile(cos_lsf_file):  
        return cos_lsf_file
    else: 
        print('No lsf file exist, please check: ', cos_lsf_file)
        return None

def voigtfit_dataset_add_component(dataset, input_tb, tie_ions_list=[], close_comp_maxv=30):
    """
    tie_ions_list: a nested list, such as [['SII', 'SiII'], ['SiIII', 'SiIV']]
                   will tie the ions within each sublist together
    close_comp_maxv: maximum velocity allowed to tie velocity components of different ions together. 
                   if the components have velocity offset by more than the designated value, 
                   we won't consider them to be the same component, and they won't be tied together. 
    History: 
    02/13/2025: update from previous code, added functions when tie_ions_list doesn't cover all ions, 
                the rest of the ions will still be added. 
                and when tieing components, if two components' estimated velocity are too far from each
                other, e.g., 30 km/s, we don't consider them to be related, and won't tie them. 
    """
    added_ions = []
    ### if you have ions you want to tie together
    for tie_ions in tie_ions_list:
        if 'CIIa' in tie_ions:
            tie_ions.remove('CIIa')  # will be taken care of later, to tie to CII in b and v

        for i, ion in enumerate(tie_ions):
            if i == 0: # first ion is the reference ion
                ref_ion = ion
                ref_tb = Table.from_pandas(input_tb.to_pandas().query('ion == @ref_ion'))
                for j in range(len(ref_tb)):
                    dataset.add_component(ref_tb['ion'][j], ref_tb['z'][j], ref_tb['b'][j], ref_tb['logN'][j])
                added_ions.append(ref_ion)
            else: # for other ions, tie to the first one in velocity/redshift space
                tie_tb = Table.from_pandas(input_tb.to_pandas().query('ion == @ion'))
                for j in range(len(tie_tb)):
                    jcomp_v = tie_tb['v_in_gal'][j]
                    # find the closest velocity in reference ion 
                    indv = np.argmin(np.abs(ref_tb['v_in_gal']-jcomp_v))
                    if np.abs(ref_tb['v_in_gal'][indv]-jcomp_v)>close_comp_maxv:
                        # if the closest component is 30 km/s away (2-3 sig COS v res), we don't consider them the same compo
                        dataset.add_component(tie_tb['ion'][j], tie_tb['z'][j], tie_tb['b'][j], tie_tb['logN'][j])
                    else:
                        tie_z = f'z{indv}_{ref_ion}'
                        print(j, tie_tb['ion'][j], tie_tb['v_in_gal'][j], tie_z, ref_tb['v_in_gal'][indv])
                        dataset.add_component(tie_tb['ion'][j], tie_tb['z'][j], tie_tb['b'][j], tie_tb['logN'][j], tie_z=tie_z)
                added_ions.append(ion)

    # take care of CIIa
    if 'CIIa' in input_tb['ion']:
        ref_ion = 'CII'
        ref_tb = Table.from_pandas(input_tb.to_pandas().query('ion == @ref_ion'))
        if 'CII' not in added_ions:
            for j in range(len(ref_tb)):
                dataset.add_component(ref_tb['ion'][j], ref_tb['z'][j], ref_tb['b'][j], ref_tb['logN'][j])

        tie_ion = 'CIIa'
        tie_tb = Table.from_pandas(input_tb.to_pandas().query('ion == @tie_ion'))
        for j in range(len(tie_tb)):
            jcomp_v = tie_tb['v_in_gal'][j]
            # find the closest velocity in reference ion 
            indv = np.argmin(np.abs(ref_tb['v_in_gal']-jcomp_v))
            tie_z = f'z{indv}_{ref_ion}'
            tie_b = f'b{indv}_{ref_ion}'
            print(j, tie_tb['ion'][j], tie_tb['v_in_gal'][j], tie_z, tie_b, ref_tb['v_in_gal'][indv])
            dataset.add_component(tie_tb['ion'][j], tie_tb['z'][j], tie_tb['b'][j], tie_tb['logN'][j], tie_b=tie_b, tie_z=tie_z)
            #print(j, tie_tb['ion'][j], tie_tb['v_in_gal'][j], tie_z, ref_tb['v_in_gal'][indv])
            #dataset.add_component(tie_tb['ion'][j], tie_tb['z'][j], tie_tb['b'][j], tie_tb['logN'][j], tie_z=tie_z) 
        added_ions.append('CIIa')
        added_ions.append('CII')

    # go through the input_tb and see if there are ions leftover                
    for i in range(len(input_tb)):
        ion = input_tb['ion'][i]
        if ion in added_ions:
            continue
        else:
            dataset.add_component(ion, input_tb['z'][i], input_tb['b'][i], input_tb['logN'][i])
    return dataset

# def voigtfit_dataset_add_component(dataset, input_tb, tie_ions_list=None): 
#     if tie_ions_list is None: 
#         for i in range(len(input_tb)): 
#             dataset.add_component(input_tb['ion'][i], input_tb['z'][i], input_tb['b'][i], input_tb['logN'][i])
#     else: # if you want to tie components together 
#         for tie_ions in tie_ions_list: 
#             for i, ion in enumerate(tie_ions): 
#                 if i == 0: # first ion is the reference ion
#                     ref_ion = ion 
#                     ref_tb = Table.from_pandas(input_tb.to_pandas().query('ion == @ref_ion'))
#                     for j in range(len(ref_tb)): 
#                         dataset.add_component(ref_tb['ion'][j], ref_tb['z'][j], ref_tb['b'][j], ref_tb['logN'][j])
#                 else: # for other ions, tie to the first one in velocity/redshift space
#                     tie_tb = Table.from_pandas(input_tb.to_pandas().query('ion == @ion'))
#                     for j in range(len(tie_tb)): 
#                         jcomp_v = tie_tb['v_in_gal'][j]
#                         indv = np.argmin(np.abs(ref_tb['v_in_gal']-jcomp_v))
#                         tie_z = f'z{indv}_{ref_ion}'
#                         print(j, tie_tb['ion'][j], tie_z, tie_tb['v_in_gal'][j], ref_tb['v_in_gal'][indv])
#                         dataset.add_component(tie_tb['ion'][j], tie_tb['z'][j], tie_tb['b'][j], tie_tb['logN'][j], tie_z=tie_z)
#                         # find the closest velocity in reference ion 
#     return dataset
