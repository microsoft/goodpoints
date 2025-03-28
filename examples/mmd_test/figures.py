import matplotlib
matplotlib.use('Agg')

import glob
import numpy as np
import os
import pickle
import argparse
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import util_classes
import util_tests

plt.style.use('seaborn-v0_8-white')

def plot_line(rejection_rate, rejection_rate_upper, rejection_rate_lower, times, labels, legend_text, marker, markersize, color, linestyle, xytext, log_time_scale, small_times):
    if log_time_scale:
        plt.semilogx(times, rejection_rate, label=legend_text, marker=marker, markersize=markersize, color=color, linestyle=linestyle)
    else:
        plt.plot(times, rejection_rate, label=legend_text, marker=marker, markersize=markersize, color=color, linestyle=linestyle)
        #if small_times:
            #plt.xlim([0,1200])
    plt.fill_between(times, rejection_rate_lower, rejection_rate_upper, alpha=.3, color=color)
    for i in range(len(rejection_rate_lower)):
        print(f'rejection_rate_upper[i]-rejection_rate_lower[i]: {rejection_rate_upper[i]-rejection_rate_lower[i]}')
    
    for x,y,z in zip(times, rejection_rate, labels):
        plt.annotate(z, # this is the text
                     (x,y), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=xytext, # distance from text to points (x,y)
                     ha='center',
                     size=8) #,fontweight='bold') # horizontal alignment can be left, right or center

def cut_times(args,rejection_rate,rejection_rate_upper,rejection_rate_lower,times,labels,group,no_compute):
    cut_idx = 0
    while cut_idx < len(times[group]) and times[group][cut_idx] < args.long_times:
        cut_idx += 1
    print(f'cut_idx for {group}: {cut_idx}')
    print(f'times[group]: {times[group]}')
    print(f'args.long_times: {args.long_times}')
    if cut_idx == len(times[group]):
        print(f'no_compute for {group} set to False because cut_idx={cut_idx} is equal to len(times)={len(times)}')
        no_compute[group] = True
    else:
        rejection_rate[group] = rejection_rate[group][cut_idx:]
        rejection_rate_upper[group] = rejection_rate_upper[group][cut_idx:]
        rejection_rate_lower[group] = rejection_rate_lower[group][cut_idx:]
        times[group] = times[group][cut_idx:]
        labels[group] = labels[group][cut_idx:]
        
def remove_violations(args,rejection_rate,rejection_rate_upper,rejection_rate_lower,times,labels,group):
    if args.n == 262144:
        if args.name == 'gaussians':
            if group == 'block_asymp_b':
                cut_idx = 3
            elif group == 'block_asymp_c':
                cut_idx = 6
            elif group == 'incomplete_asymp_b':
                cut_idx = 5
            else:
                cut_idx = len(rejection_rate[group])
        elif args.name == 'EMNIST':
            if group == 'block_asymp_b':
                cut_idx = 3
            elif group == 'block_asymp_c':
                cut_idx = 5
            elif group == 'incomplete_asymp_b':
                cut_idx = 2
            else:
                cut_idx = len(rejection_rate[group])
        else:
            cut_idx = len(rejection_rate[group])
    else:
        if args.name == 'gaussians':
            if group == 'block_asymp_b':
                cut_idx = 4
            elif group == 'block_asymp_c':
                cut_idx = 6
            elif group == 'incomplete_asymp_b':
                cut_idx = 5
            else:
                cut_idx = len(rejection_rate[group])
        elif args.name == 'EMNIST':
            if group == 'block_asymp_b':
                cut_idx = 1
            elif group == 'block_asymp_c':
                cut_idx = 5
            elif group == 'incomplete_asymp_b':
                cut_idx = 7
            else:
                cut_idx = len(rejection_rate[group])
        else:
            cut_idx = len(rejection_rate[group])
        
    rejection_rate[group] = rejection_rate[group][:cut_idx]
    rejection_rate_upper[group] = rejection_rate_upper[group][:cut_idx]
    rejection_rate_lower[group] = rejection_rate_lower[group][:cut_idx]
    times[group] = times[group][:cut_idx]
    labels[group] = labels[group][:cut_idx]

def pplot(ax=None):
    if ax is None:
        plt.grid(True, alpha=0.5)
        axoff(plt.gca())
    else:
        ax.grid(True, alpha=0.5)
        axoff(ax)
    return

def axoff(ax, keys=['top', 'right']):
    for k in keys:
        ax.spines[k].set_visible(False)
    return

def fig_rejection(args, test_groups, joint_group_results_dict, aggregated=False):
    
    rejection_rate = dict()
    rejection_rate_upper = dict()
    rejection_rate_lower = dict()
    times = dict()
    labels = dict()
    
    #linestyle, markers, marker sizes and colors
    lss = ['-', '-.',  ':', '--',  '--', '-.', ':', '-', '--', '-.', ':', '-']*2
    mss = ['>','s', 'o', 'D', '+', '*', 'x', '>', '<', '^', 'v']*2 
    #['>', 's', 'o', 'D', '+', '*',  '>', 's', 'o', 'D', '>', 's', 'o', 'D']*2
    ms_size = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]*2
    colors = ['#e41a1c', 'cyan', '#0000cd', '#4daf4a', 'magenta', 'gray' ,'orange','yellow', 'black']*2
    
    null_distribution = (args.name == 'gaussians' and args.mean_diff == 0) or (args.name == 'blobs' and args.epsilon == 1) or (args.name == 'EMNIST' and args.p_even == 0.5) or (args.name == 'Higgs' and args.p_poisoning == 1) or (args.name == 'sine' and args.omega == 0)
    
    std_multiple = scipy.stats.norm.ppf((1+args.wilson_size)/2)
        
    #Compute rejection rates, rejection rate bars, times and labels for CTT/ACTT
    if not no_compute['ctt']:
        rejection_rate['ctt'], rejection_rate_upper['ctt'], rejection_rate_lower['ctt'], times['ctt'], labels['ctt'] = joint_group_results_dict['ctt'].get_lists(wilson_intervals=args.wilson_intervals,z=std_multiple)
        if aggregated:
            input_labels = ['']*len(args.block_g_list)
            rejection_rate['ctt_median'], rejection_rate_upper['ctt_median'], rejection_rate_lower['ctt_median'], times['ctt_median'], labels['ctt_median'] = joint_group_results_dict['ctt'].get_lists(wilson_intervals=args.wilson_intervals,z=std_multiple,median_rate=True,labels=input_labels)
    
    #Compute rejection rates, rejection rate bars, times and labels for RFF
    if not no_compute['rff']:
        input_labels = ['']*len(args.n_features_list)
        if not null_distribution:
            input_labels[0] = f'r={args.n_features_list[0]}'
            input_labels[-1] = f'r={args.n_features_list[-1]}'
        rejection_rate['rff'], rejection_rate_upper['rff'], rejection_rate_lower['rff'], times['rff'], labels['rff'] = joint_group_results_dict['rff'].get_lists(wilson_intervals=args.wilson_intervals,z=std_multiple,labels=input_labels)
        if aggregated:
            input_labels = ['']*len(args.n_features_list)
            rejection_rate['rff_median'], rejection_rate_upper['rff_median'], rejection_rate_lower['rff_median'], times['rff_median'], labels['rff_median'] = joint_group_results_dict['rff'].get_lists(wilson_intervals=args.wilson_intervals,z=std_multiple,median_rate=True,labels=input_labels)
            
    #Compute rejection rates, rejection rate bars, times and labels for Low Rank CTT
    if not no_compute['ctt_rff']:
        #for g in args.block_g_list_ctt_rff:
        #    dict_key = 'ctt_rff'+str(g)
        rejection_rate['ctt_rff'], rejection_rate_upper['ctt_rff'], rejection_rate_lower['ctt_rff'], times['ctt_rff'], labels['ctt_rff'] = joint_group_results_dict['ctt_rff'].get_lists(wilson_intervals=args.wilson_intervals,z=std_multiple)
        if aggregated:
            input_labels = ['']*len(args.n_features_list)
            rejection_rate['ctt_rff_median'], rejection_rate_upper['ctt_rff_median'], rejection_rate_lower['ctt_rff_median'], times['ctt_rff_median'], labels['ctt_rff_median'] = joint_group_results_dict['ctt_rff'].get_lists(wilson_intervals=args.wilson_intervals,z=std_multiple,median_rate=True,labels=input_labels)
    
    no_compute_block_wb = no_compute['block_wb']
    print(f'no_compute_block_wb: {no_compute_block_wb}, aggregated: {aggregated}')
    #Compute rejection rates, rejection rate bars, times and labels for block WB
    if not no_compute['block_wb'] and not aggregated:
        print(f'Plot block_wb')
        input_labels = []
        for size_num, size in enumerate(args.wb_block_size_list):
            if args.log_time_scale and null_distribution and args.show_exact:
                empty_label = True
            else:
                empty_label = (size_num < len(args.wb_block_size_list) - 1)
            if empty_label:
                input_labels.append('')
            else:
                log_size = int(np.log2(size))
                if log_size%2 == 0:
                    log4_size = int(log_size/2)
                    input_labels.append(f'$B=4^{log4_size}$')
                else:
                    log4_size = int((log_size+1)/2)
                    input_labels.append(f'$B=4^{log4_size}/2$')
        print(f'input labels block_wb: {input_labels}')
        rejection_rate['block_wb'], rejection_rate_upper['block_wb'], rejection_rate_lower['block_wb'], times['block_wb'], labels['block_wb'] = joint_group_results_dict['block_wb'].get_lists(wilson_intervals=args.wilson_intervals,z=std_multiple, labels=input_labels)
        
        cut_times(args, rejection_rate, rejection_rate_upper, rejection_rate_lower, times, labels, 'block_wb', no_compute)
        
        times_block_wb = times['block_wb']
        print(f'times_block_wb: {times_block_wb}')
        
        no_compute_block_wb = no_compute['block_wb']
        print(f'no_compute_block_wb: {no_compute_block_wb}, aggregated: {aggregated}')
    
    #Compute rejection rates, rejection rate bars, times and labels for incomplete WB
    if not no_compute['incomplete_wb']:
        print(f'Plot incomplete_wb')
        input_labels = []
        for l_num, l in enumerate(args.wb_incomplete_list_multiples):
            if args.log_time_scale and null_distribution and args.show_exact:
                input_labels.append('')
            elif args.name in ['gaussians','EMNIST','sine'] and l_num < len(args.wb_incomplete_list_multiples) - 1: 
                input_labels.append('')
            elif args.name in ['blobs','Higgs'] and l_num < len(args.wb_incomplete_list_multiples) - 2: 
                input_labels.append('')
            else:
                log_l = int(np.log2(l))
                if l == int(l):
                    if log_l%2 == 0:
                        log4_l = int(log_l/2)
                        input_labels.append(r'$\ell=$'+f'$4^{log4_l}n$')
                    else:
                        log4_l = int((log_l+1)/2)
                        input_labels.append(r'$\ell=$'+f'$4^{log4_l}$'+r'$\frac{n}{2}$')
                else:
                    if args.n == 262144:
                        input_labels.append(r'$\ell=$'+r'$\frac{4^9-1}{2}$'+r'$n$')
                    else:
                        input_labels.append(r'$\ell=$'+r'$\frac{4^7-1}{2}$'+r'$n$')
        rejection_rate['incomplete_wb'], rejection_rate_upper['incomplete_wb'], rejection_rate_lower['incomplete_wb'], times['incomplete_wb'], labels['incomplete_wb'] = joint_group_results_dict['incomplete_wb'].get_lists(wilson_intervals=args.wilson_intervals,z=std_multiple,labels=input_labels)
        
        cut_times(args, rejection_rate, rejection_rate_upper, rejection_rate_lower, times, labels, 'incomplete_wb', no_compute)
        
        if aggregated:
            input_labels_median = []
            for l_num, l in enumerate(args.wb_incomplete_list_multiples):
                input_labels_median.append('')
            rejection_rate['incomplete_wb_median'], rejection_rate_upper['incomplete_wb_median'], rejection_rate_lower['incomplete_wb_median'], times['incomplete_wb_median'], labels['incomplete_wb_median'] = joint_group_results_dict['incomplete_wb'].get_lists(wilson_intervals=args.wilson_intervals,z=std_multiple,median_rate=True,labels=input_labels_median)
            cut_times(args, rejection_rate, rejection_rate_upper, rejection_rate_lower, times, labels, 'incomplete_wb_median', no_compute)
    
    #Compute rejection rates, rejection rate bars, times and labels for block asymp.
    if not no_compute['block_asymp'] and not aggregated:
        order_a = []
        order_b = []
        order_c = []
        input_labels = []
        n_var = np.minimum(2048,args.n)
        for size_num, size in enumerate(args.asymptotic_block_size_list):
            order_a.append('bk'+str(size)+'nv'+str(n_var))
            order_b.append('bk'+str(size)+'nv'+str(n_var)+'b')
            order_c.append('bk'+str(size)+'nv'+str(n_var)+'c')
            
            if null_distribution:
                if args.n == 262144:
                    empty_label = ((args.name == 'gaussians' and size_num < 7) or (args.name == 'EMNIST' and size_num < 7)) 
                else:
                    empty_label = ((args.name == 'gaussians' and size_num < 7) or (args.name == 'EMNIST' and size_num < 7))
            else:
                if args.n == 262144:
                    empty_label = ((args.name == 'gaussians' and size_num < 6) or args.name == 'EMNIST') 
                else:
                    empty_label = ((args.name == 'gaussians' and size_num < 7) or args.name == 'EMNIST')
            if args.log_time_scale and null_distribution and args.show_exact:
                empty_label = True
            if empty_label:
                input_labels.append('')
            else:
                log_size = int(np.log2(size))
                if log_size%2 == 0:
                    log4_size = int(log_size/2)
                    input_labels.append(f'$B=4^{log4_size}$')
                else:
                    log4_size = int((log_size+1)/2)
                    input_labels.append(f'$B=4^{log4_size}/2$')
        print(f'input labels block_asymp: {input_labels}')
        rejection_rate['block_asymp_a'], rejection_rate_upper['block_asymp_a'], rejection_rate_lower['block_asymp_a'], times['block_asymp_a'], labels['block_asymp_a'] = joint_group_results_dict['block_asymp'].get_lists(order=order_a, labels=input_labels,wilson_intervals=args.wilson_intervals,z=std_multiple)
        rejection_rate['block_asymp_b'], rejection_rate_upper['block_asymp_b'], rejection_rate_lower['block_asymp_b'], times['block_asymp_b'], labels['block_asymp_b'] = joint_group_results_dict['block_asymp'].get_lists(order=order_b, labels=input_labels,wilson_intervals=args.wilson_intervals,z=std_multiple)
        if null_distribution:
            input_labels = ['']*len(input_labels)
        rejection_rate['block_asymp_c'], rejection_rate_upper['block_asymp_c'], rejection_rate_lower['block_asymp_c'], times['block_asymp_c'], labels['block_asymp_c'] = joint_group_results_dict['block_asymp'].get_lists(order=order_c, labels=input_labels,wilson_intervals=args.wilson_intervals,z=std_multiple)
        print(labels['block_asymp_b'])
        print(labels['block_asymp_c'])
        
        cut_times(args, rejection_rate, rejection_rate_upper, rejection_rate_lower, times, labels, 'block_asymp_a', no_compute)
        cut_times(args, rejection_rate, rejection_rate_upper, rejection_rate_lower, times, labels, 'block_asymp_b', no_compute)
        cut_times(args, rejection_rate, rejection_rate_upper, rejection_rate_lower, times, labels, 'block_asymp_c', no_compute)
        
        print(labels['block_asymp_b'])
        print(labels['block_asymp_c'])
        
        if args.no_violations:
            remove_violations(args, rejection_rate, rejection_rate_upper, rejection_rate_lower, times, labels, 'block_asymp_a')
            remove_violations(args, rejection_rate, rejection_rate_upper, rejection_rate_lower, times, labels, 'block_asymp_b')
            remove_violations(args, rejection_rate, rejection_rate_upper, rejection_rate_lower, times, labels, 'block_asymp_c')
        
        print(labels['block_asymp_b'])
        print(labels['block_asymp_c'])
    
    #Compute rejection rates, rejection rate bars, times and labels for incomplete asymp.
    if not no_compute['incomplete_asymp'] and not aggregated:
        order_a = []
        order_b = []
        input_labels = []
        n_var = np.minimum(2048,args.n)
        for l in args.asymptotic_incomplete_list:
            order_a.append('i'+str(l)+'nv'+str(n_var))
            order_b.append('i'+str(l)+'nv'+str(n_var)+'b')
        for l_num, l in enumerate(args.asymptotic_incomplete_list_multiples):
            if null_distribution:
                if args.log_time_scale and args.show_exact:
                    empty_label = True
                else:
                    if args.n == 262144:
                        empty_label = ((args.name == 'gaussians' and l_num < 5) or (args.name == 'EMNIST' and l_num < 5)) 
                    else:
                        empty_label = ((args.name == 'gaussians' and l_num < 7) or (args.name == 'EMNIST' and l_num < 7))
                if l_num == 0: 
                    print(f'l_num = 0, empty_label={empty_label}')
            else:
                empty_label = ((args.name == 'gaussians' and l_num < 6) or args.name == 'EMNIST') #not args.log_time_scale and
            if empty_label: 
                input_labels.append('')
            else:
                log_l = int(np.log2(l))
                if l == int(l):
                    if log_l%2 == 0:
                        log4_l = int(log_l/2)
                        input_labels.append(r'$\ell=$'+f'$4^{log4_l}n$')
                    elif log_l == 15:
                        input_labels.append(r'$\ell=$'+r'$\frac{4^8}{2}n$')
                    else:
                        log4_l = int((log_l+1)/2)
                        input_labels.append(r'$\ell=$'+f'$4^{log4_l}$'+r'$\frac{n}{2}$')
                else:
                    input_labels.append(r'$\ell=$'+r'$\frac{4^9-1}{2}$'+r'$n$')
        rejection_rate['incomplete_asymp_a'], rejection_rate_upper['incomplete_asymp_a'], rejection_rate_lower['incomplete_asymp_a'], times['incomplete_asymp_a'], labels['incomplete_asymp_a'] = joint_group_results_dict['incomplete_asymp'].get_lists(order=order_a, labels=input_labels, wilson_intervals=args.wilson_intervals, z=std_multiple)
        rejection_rate['incomplete_asymp_b'], rejection_rate_upper['incomplete_asymp_b'], rejection_rate_lower['incomplete_asymp_b'], times['incomplete_asymp_b'], labels['incomplete_asymp_b'] = joint_group_results_dict['incomplete_asymp'].get_lists(order=order_b, labels=input_labels, wilson_intervals=args.wilson_intervals, z=std_multiple)
        
        cut_times(args, rejection_rate, rejection_rate_upper, rejection_rate_lower, times, labels, 'incomplete_asymp_a', no_compute)
        cut_times(args, rejection_rate, rejection_rate_upper, rejection_rate_lower, times, labels, 'incomplete_asymp_b', no_compute)
        
        if args.no_violations:
            remove_violations(args, rejection_rate, rejection_rate_upper, rejection_rate_lower, times, labels, 'incomplete_asymp_a')
            remove_violations(args, rejection_rate, rejection_rate_upper, rejection_rate_lower, times, labels, 'incomplete_asymp_b')
           
    #Plot settings    
    title_size = 20
    fix_plot_settings = True
    if fix_plot_settings:
        plt.rc('font', family='serif')
        plt.rc('text', usetex=False)
        label_size = 9
        legend_size = 8
        mpl.rcParams['xtick.labelsize'] = label_size 
        mpl.rcParams['ytick.labelsize'] = label_size 
        mpl.rcParams['axes.labelsize'] = label_size
        mpl.rcParams['axes.titlesize'] = label_size
        mpl.rcParams['figure.titlesize'] = label_size
        mpl.rcParams['lines.markersize'] = label_size
        mpl.rcParams['grid.linewidth'] = 1.5
        mpl.rcParams['legend.fontsize'] = legend_size
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        pylab.rcParams['xtick.major.pad'] = 5
        pylab.rcParams['ytick.major.pad'] = 5
    
    #Title settings
    if args.name == 'gaussians': 
        if args.n == 262144:
            plt.title(f'Gaussian (mean separation$=${args.mean_diff}, $n=4^9$)')
        else:
            plt.title(f'Gaussian (mean separation$=${args.mean_diff}, $n=4^7$)')
    elif args.name == 'blobs':
        if args.n == 262144:
            plt.title(f'Blobs ($\epsilon=${args.epsilon}, $n=4^9$)')
        else:
            if args.epsilon == 1:
                plt.title(f'Blobs (null, $n=4^7$)')
            else:
                plt.title(f'Blobs ($\epsilon=${args.epsilon}, $n=4^7$)')
    elif args.name == 'EMNIST':
        if args.n == 262144:
            plt.title(f'Downsampled EMNIST ('+r'$p_{even}=$'+f'{args.p_even}, $n=4^9$)')
        else:
            plt.title(f'Downsampled EMNIST ('+r'$p_{even}=$'+f'{args.p_even}, $n=4^7$)')
    elif args.name == 'Higgs': 
        if args.n == 262144:
            plt.title(r'Higgs ($p_{p}=$'+f'${args.p_poisoning}$'+r', $n=4^9$)')
        else:
            if args.p_poisoning == 1:
                plt.title(r'Higgs (null'+r', $n=4^7$)')
            elif args.p_poisoning == 0:
                plt.title(r'Higgs ('+'$n=4^7$)')
            else:
                plt.title(r'Higgs Mixture ($p_{m}=$'+f'${args.p_poisoning}$'+r', $n=4^7$)')
    if args.name == 'sine': 
        if args.n == 262144:
            plt.title(f'Sine ($\omega=${args.omega}, $n=4^9$)')
        else:
            plt.title(f'Sine ($\omega=${args.omega}, $n=4^7$)')
    
    line_number = 0
        
    #Plot CTT/ACTT line
    if not no_compute['ctt']:
        if not aggregated:
            legend_content = 'CTT'
            label_position = (-8,7)
        else:
            legend_content = 'ACTT'
            label_position = (-8,4)
        plot_line(rejection_rate['ctt'], rejection_rate_upper['ctt'], rejection_rate_lower['ctt'], times['ctt'], labels['ctt'], legend_text=legend_content, marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1
        
    if args.show_exact or no_compute['ctt']:
        line_number = 1
        
    #Plot block WB line
    no_compute_block_wb = no_compute['block_wb']
    print(f'no_compute_block_wb: {no_compute_block_wb}, aggregated: {aggregated}')
    if not no_compute['block_wb'] and not aggregated:
        print('labels[block_wb]', labels['block_wb'])
        print('plot line block_wb')
        label_position = (-14,0)
        plot_line(rejection_rate['block_wb'], rejection_rate_upper['block_wb'], rejection_rate_lower['block_wb'], times['block_wb'], labels['block_wb'], legend_text='W-Block', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1
        
    if args.name == 'blobs' or 'Higgs':
        line_number = 2
        
    #Plot incomplete WB line
    if not no_compute['incomplete_wb']:
        if args.log_time_scale:
            if args.name == 'gaussians':
                label_position = (21,-4)
            elif args.name == 'blobs':
                label_position = (-25,0)
            elif args.name == 'Higgs':
                if args.p_poisoning == 0.0:
                    label_position = (-6,-13)
                elif args.p_poisoning == 0.5:
                    label_position = (-6,10)
                else:
                    label_position = (-6,-13)
            else:
                label_position = (10,-14)
        else:
            label_position = (-8,-20)
        if aggregated:
            legend_content = 'Agg. W-Incomp.'
        else:
            legend_content = 'W-Incomp.'
        print('plot line incomplete_wb')
        plot_line(rejection_rate['incomplete_wb'], rejection_rate_upper['incomplete_wb'], rejection_rate_lower['incomplete_wb'], times['incomplete_wb'], labels['incomplete_wb'], legend_text=legend_content, marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1
        
    if null_distribution:
        line_number = 3    
        
    #Plot block asymp. line
    if not no_compute['block_asymp'] and not aggregated:
        if not no_compute['asymp_a']:
            label_position = (0,3)
            plot_line(rejection_rate['block_asymp_a'], rejection_rate_upper['block_asymp_a'], rejection_rate_lower['block_asymp_a'], times['block_asymp_a'], labels['block_asymp_a'], legend_text=f'Asymp. Block (a), n_v={2048}', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1
        if null_distribution:
            label_position = (18,4)
        else:
            label_position = (0,4)
        plot_line(rejection_rate['block_asymp_b'], rejection_rate_upper['block_asymp_b'], rejection_rate_lower['block_asymp_b'], times['block_asymp_b'], labels['block_asymp_b'], legend_text=f'A-Block I', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1
        if args.log_time_scale:
            label_position = (-8,2)#(0,2)
        else:
            if null_distribution:
                label_position = (-23,-5)
            else:
                label_position = (-12,5)
        plot_line(rejection_rate['block_asymp_c'], rejection_rate_upper['block_asymp_c'], rejection_rate_lower['block_asymp_c'], times['block_asymp_c'], labels['block_asymp_c'], legend_text=f'A-Block II', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1
        
    #Plot incomplete asymp. line
    if not no_compute['incomplete_asymp'] and not aggregated:
        if not no_compute['asymp_a']:
            plot_line(rejection_rate['incomplete_asymp_a'], rejection_rate_upper['incomplete_asymp_a'], rejection_rate_lower['incomplete_asymp_a'], times['incomplete_asymp_a'], labels['incomplete_asymp_a'], legend_text=f'Asymp. Incomplete (a), n_v={2048}', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=(0,3), log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1
        if null_distribution:
            label_position = (-8,-25)
        else:
            label_position = (12,4) #(0,-19)
        plot_line(rejection_rate['incomplete_asymp_b'], rejection_rate_upper['incomplete_asymp_b'], rejection_rate_lower['incomplete_asymp_b'], times['incomplete_asymp_b'], labels['incomplete_asymp_b'], legend_text=f'A-Incomp.', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1
        
    #If aggregated, plot median CTT line
    if not no_compute['ctt']:
        if aggregated:
            legend_content = r'CTT (median $\lambda$)'
            label_position = (8,-4)
            plot_line(rejection_rate['ctt_median'], rejection_rate_upper['ctt_median'], rejection_rate_lower['ctt_median'], times['ctt_median'], labels['ctt_median'], legend_text=legend_content, marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1
    
    #If aggregated, plot median incomplete WB line
    if not no_compute['incomplete_wb']:
        if aggregated:
            if args.log_time_scale:
                if args.name == 'gaussians':
                    label_position = (21,-4)#(-4,-20)
                elif args.name == 'blobs':
                    label_position = (-25,0)
                else:
                    label_position = (10,-14)
            else:
                label_position = (-8,-20)
            if args.name == 'Higgs' and args.p_poisoning == 0.5:
                legend_content = r'W-Incomp. (med. $\lambda$)'
            else:
                legend_content = r'W-Incomp. (median $\lambda$)'
            plot_line(rejection_rate['incomplete_wb_median'], rejection_rate_upper['incomplete_wb_median'], rejection_rate_lower['incomplete_wb_median'], times['incomplete_wb_median'], labels['incomplete_wb_median'], legend_text=legend_content, marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1
            
    #Plot RFF line
    if not no_compute['rff']:
        if not aggregated:
            legend_content = 'RFF'
            label_position = (10,-9)
        else:
            legend_content = 'Agg. RFF'
            label_position = (-8,4)
        plot_line(rejection_rate['rff'], rejection_rate_upper['rff'], rejection_rate_lower['rff'], times['rff'], labels['rff'], legend_text=legend_content, marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1
        
    #If aggregated, plot median RFF line
    if not no_compute['rff']:
        if aggregated:
            legend_content = r'RFF (median $\lambda$)'
            label_position = (8,-4)
            plot_line(rejection_rate['rff_median'], rejection_rate_upper['rff_median'], rejection_rate_lower['rff_median'], times['rff_median'], labels['rff_median'], legend_text=legend_content, marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1
            
    num_features_ctt_rff = len(args.n_features_list_ctt_rff)
    num_block_g_ctt_rff = len(args.block_g_list_ctt_rff)
    labels_n_features = [1,2,4,8,16,32,64,128,256,512,1024,2048] #[16,64,256,1024]
    #Plot RFF line
    if not no_compute['ctt_rff']:
        if not aggregated:
            legend_content = 'LR-CTT-RFF'
            label_position = (-8,7)
        else:
            legend_content = 'LR-CTT-RFF'
            label_position = (-8,4)
        for g_num, g in enumerate(args.block_g_list_ctt_rff):
            #g_num = 0
            #g = 0
            #labels['ctt_rff'][g_num*num_features_ctt_rff:(g_num+1)*num_features_ctt_rff]
            time_seq0 = times['ctt_rff'][g_num*num_features_ctt_rff:(g_num+1)*num_features_ctt_rff]
            print(f'time_seq0: {time_seq0}')
            time_seq = [time_seq0[i] for i in range(len(time_seq0)) if time_seq0[i] != 0]
            rr_seq = [rejection_rate['ctt_rff'][g_num*num_features_ctt_rff:(g_num+1)*num_features_ctt_rff][i] for i in range(len(time_seq0)) if time_seq0[i] != 0]
            rru_seq = [rejection_rate_upper['ctt_rff'][g_num*num_features_ctt_rff:(g_num+1)*num_features_ctt_rff][i] for i in range(len(time_seq0)) if time_seq0[i] != 0]
            rrl_seq = [rejection_rate_lower['ctt_rff'][g_num*num_features_ctt_rff:(g_num+1)*num_features_ctt_rff][i] for i in range(len(time_seq0)) if time_seq0[i] != 0]
            #labels_seq = [labels_n_features[i] for i in range(len(time_seq0)) if time_seq0[i] != 0]
            labels_seq = ['' for i in range(len(time_seq0))]
            if g==3 and not null_distribution:
                labels_seq[0] = f'r={args.n_features_list_ctt_rff[0]}'
                labels_seq[-1] = f'r={args.n_features_list_ctt_rff[-1]}'
            #print(f'time_seq: {time_seq}')
            #labels_empty = ['']*num_features_ctt_rff
            #plot_line(rejection_rate['ctt_rff'][g_num*num_features_ctt_rff:(g_num+1)*num_features_ctt_rff], rejection_rate_upper['ctt_rff'][g_num*num_features_ctt_rff:(g_num+1)*num_features_ctt_rff], rejection_rate_lower['ctt_rff'][g_num*num_features_ctt_rff:(g_num+1)*num_features_ctt_rff], times['ctt_rff'][g_num*num_features_ctt_rff:(g_num+1)*num_features_ctt_rff], labels_empty, legend_text=legend_content+' g='+str(g), marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            plot_line(rr_seq, rru_seq, rrl_seq, time_seq, labels_seq, legend_text=legend_content+' g='+str(g), marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1
        
    #If aggregated, plot median RFF line
    if not no_compute['ctt_rff']:
        if aggregated:
            legend_content = r'Low Rank CTT (median $\lambda$)'
            label_position = (8,-4)
            plot_line(rejection_rate['ctt_rff_median'], rejection_rate_upper['ctt_rff_median'], rejection_rate_lower['ctt_rff_median'], times['ctt_rff_median'], labels['ctt_rff_median'], legend_text=legend_content, marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1
        
    #Plot level line
    if not args.no_nominal_level:
        plt.axhline(y=args.alpha, label='Level 0.05', color='orange', linestyle=':')
    
    if args.name == 'gaussians': 
        plt.xlabel('Total computation time (s)')
    elif args.name == 'blobs': 
        plt.xlabel('Total computation time (s)')
    elif args.name == 'EMNIST': 
        plt.xlabel('Total computation time (s)')
    elif args.name == 'Higgs': 
        plt.xlabel('Total computation time (s)')
    
    if null_distribution:
        plt.ylabel('Size (Type I error)')
    else:
        plt.ylabel('Power (1 - Type II error)')
    
    #Adjust plot limits and axis labels
    if args.name == 'gaussians': 
        if args.log_time_scale:
            if null_distribution and args.show_exact and no_compute['ctt_rff']:
                if args.n == 262144:
                    mpl.rcParams["legend.loc"] = 'upper left'
                    plt.xlim([15,8000])
                    plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000], 
                               ['20', '50', '100', '200', '500', '1k', '2k', '5k'])
                elif args.n == 16384:
                    mpl.rcParams["legend.loc"] = 'upper right'
                    plt.xlim([0.1,32])
                    plt.xticks([0.2, 0.5, 1, 2, 5, 10, 20], 
                               ['0.2', '0.5', '1', '2', '5', '10', '20'])
                    #plt.xlim([0.075,500])
                    #plt.xticks([0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200],
                    #            ['0.2', '0.5', '1', '2', '5', '10', '20', '50', '100', '200'])
            elif null_distribution and not args.show_exact and no_compute['ctt_rff']:
                if args.n == 262144:
                    mpl.rcParams["legend.loc"] = 'upper left'
                    plt.xlim([8,8000])
                    plt.xticks([10, 20, 50, 100, 200, 500, 1000, 2000, 5000], 
                               ['10', '20', '50', '100', '200', '500', '1k', '2k', '5k'])
                elif args.n == 16384:
                    mpl.rcParams["legend.loc"] = 'upper left'
                    plt.xlim([0.075,28])
                    plt.xticks([0.2, 0.5, 1, 2, 5, 10, 20], ['0.2', '0.5', '1', '2', '5', '10', '20'])
            elif not no_compute['ctt_rff']:
                if args.n == 262144:
                    if null_distribution:
                        mpl.rcParams["legend.loc"] = 'upper left'
                    else:
                        mpl.rcParams["legend.loc"] = 'lower right'
                    #plt.xlim([0.005,50])
                    plt.xlim([0.005,215])
                    plt.xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100], 
                               ['0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '50', '100'])
                    #plt.xticks([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000], 
                    #           ['1', '2', '5', '10', '20', '50', '100', '200', '500', '1k', '2k', '5k'])
                else:
                    mpl.rcParams["legend.loc"] = 'lower right'
                    #plt.xlim([0.001,5])
                    plt.xlim([0.0008,20])
                    plt.xticks([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10], 
                               ['0.002', '0.005', '0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10'])
            else:
                if args.n == 262144:
                    mpl.rcParams["legend.loc"] = 'lower right'
                    plt.xlim([0.27,23000])
                    #plt.xlim([5,23000])
                    plt.xticks([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000], 
                               ['1', '2', '5', '10', '20', '50', '100', '200', '500', '1k', '2k', '5k'])
                    #plt.xticks([10, 20, 50, 100, 200, 500, 1000, 2000, 5000], 
                    #           ['10', '20', '50', '100', '200', '500', '1k', '2k', '5k'])
                else:
                    mpl.rcParams["legend.loc"] = 'lower right'
                    plt.xlim([0.017,70])
                    plt.xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50], 
                               ['0.02', '0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '50'])
        else:
            plt.xlim([-475,5900])
    elif args.name == 'EMNIST':
        if args.log_time_scale:
            if null_distribution and args.show_exact and no_compute['ctt_rff']:
                if args.n == 262144:
                    mpl.rcParams["legend.loc"] = 'upper left'
                    plt.xlim([15,15000])
                    plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000], 
                               ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k'])
                else:
                    mpl.rcParams["legend.loc"] = 'upper right'
                    plt.xlim([0.16,65])
                    plt.xticks([0.2, 0.5, 1, 2, 5, 10, 20, 50], 
                               ['0.2', '0.5', '1', '2', '5', '10', '20', '50'])
            elif null_distribution and not args.show_exact and no_compute['ctt_rff']:
                if args.n == 262144:
                    mpl.rcParams["legend.loc"] = 'upper left'
                    plt.xlim([18,15000])
                    plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000], 
                               ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k'])
                else:
                    mpl.rcParams["legend.loc"] = 'upper left'
                    plt.xlim([0.3,80])
                    plt.xticks([0.5, 1, 2, 5, 10, 20, 50], ['0.5', '1', '2', '5', '10', '20', '50'])
            elif not no_compute['ctt_rff']:
                if args.n == 262144:
                    if null_distribution:
                        mpl.rcParams["legend.loc"] = 'upper left'
                    else:
                        mpl.rcParams["legend.loc"] = 'lower right'
                    #plt.xlim([0.005,50])
                    plt.xlim([0.02,250])
                    plt.xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100], 
                               ['0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '50', '100'])
                    #plt.xticks([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000], 
                    #           ['1', '2', '5', '10', '20', '50', '100', '200', '500', '1k', '2k', '5k'])
                else:
                    if null_distribution:
                        mpl.rcParams["legend.loc"] = 'upper left'
                        plt.ylim(0.034,0.091)
                    else:
                        mpl.rcParams["legend.loc"] = 'lower right'
                    #plt.xlim([0.001,5])
                    plt.xlim([0.0018,10])
                    plt.xticks([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5], 
                               ['0.002', '0.005', '0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1', '2', '5'])
            else:
                if args.n == 262144:
                    mpl.rcParams["legend.loc"] = 'lower right'
                    plt.xlim([0.65,23000])
                    plt.xticks([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000], 
                               ['1', '2', '5', '10', '20', '50', '100', '200', '500', '1k', '2k', '5k', '10k'])
                else:
                    mpl.rcParams["legend.loc"] = 'lower right'
                    plt.xlim([0.04,110])
                    plt.xticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100], 
                               ['0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '50', '100'])
        else:
            plt.xlim([-800,10800])
    elif args.name == 'blobs': #args.small_times:
        if args.log_time_scale:
            if null_distribution:
                plt.xlim([0.4,800])
                plt.xticks([1, 2, 5, 10, 20, 50, 100, 200, 500], ['1', '2', '5', '10', '20', '50', '100', '200', '500'])
            else:
                plt.xlim([0.025,800])
                plt.xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500], 
                           ['0.05', '0.1', '0.2', '0.5','1','2','5','10', '20', '50', '100', '200', '500'])
        else:
            plt.xlim([-5000,68250])
    elif args.name == 'Higgs': #args.small_times:
        if args.log_time_scale:
            if null_distribution:
                plt.xlim([0.02,800])
                plt.ylim(-0.06,1.04)
                plt.xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500], 
                           ['0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '50', '100', '200', '500'])
            elif args.p_poisoning == 0.5:
                plt.xlim([0.01,800])
                plt.xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500], 
                           ['0.02', '0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '50', '100', '200', '500'])
            else:
                plt.xlim([0.02,800])
                plt.xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500], 
                           ['0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '50', '100', '200', '500'])
        else:
            plt.xlim([-5000,68250])
    plt.legend(handletextpad=0.0)
    
def format_int_list(mylist):
    formatted_list = ""
    for i in range(len(mylist)-1):
        formatted_list += str(mylist[i]) + '_'
    formatted_list += str(mylist[len(mylist)-1])
    return formatted_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMD tests figures')
    
    #General arguments
    parser.add_argument('--name', default='gaussians', help='experiment name')
    parser.add_argument('--n', type=int, default=262144, help='number of samples')
    parser.add_argument('--d', type=int, default=49, help='dimension')
    parser.add_argument('--B', type=int, default=39, help='number of permutations/Rademacher variables used')
    parser.add_argument('--alpha', type=float, default=0.05, help='level of the test')
    parser.add_argument('--n_tests', type=int, default=1, help='number of tests')
    parser.add_argument('--total_n_tests', type=int, default=200, help='number of tests')
    parser.add_argument('--aggregated', action='store_true', help='if True, compute aggregated test, else compute single test')
    parser.add_argument('--error_bars_percentile', action='store_true', help='use percentiles 10 and 90 for statistics error bars')
    parser.add_argument('--no_nominal_level', action='store_true', help='if passed do not plot nominal test level (alpha)')
    parser.add_argument('--log_time_scale', action='store_true', help='if passed use log scale for time (x axis)')
    parser.add_argument('--small_times', action='store_true', help='if passed restrict time axis to the first 1200 seconds')
    parser.add_argument('--long_times', type=float, default=0.0, help='if passed restrict time axis to more than this value')
    parser.add_argument('--wilson_intervals', action='store_true', help='if passed, use Wilson confidence intervals in plots')
    parser.add_argument('--wilson_size', type=float, default=0.95, help='size of Wilson intervals')
    parser.add_argument('--no_violations', action='store_true', help='if passed, remove points for asymp. tests for which the nominal level is not respected')
    parser.add_argument('--show_exact', action='store_true', help='if passed, show only permutations/WB test for size plots')
    parser.add_argument('--s', type=int, default=16, help='number of bins for CTT')
    parser.add_argument('--s_rff', type=int, default=16, help='number of bins for Low Rank CTT compression')
    parser.add_argument('--s_permute', type=int, default=16, help='number of bins for Low Rank CTT permutation')
    
    #Argument for gaussians
    parser.add_argument('--mean_diff', type=float, default=0.024, help='covariance eigenvalue (for blobs)')
    
    #Arguments for blobs
    parser.add_argument('--grid_size', type=int, default=3, help='dimension of the grid of the distribution (for blobs)')
    parser.add_argument('--epsilon', type=float, default=2, help='covariance eigenvalue (for blobs)')
    
    #Arguments for MNIST and EMNIST
    parser.add_argument('--p_even', type=float, default=0.49, help='joint probability of all even digits (for MNIST and EMNIST)')
    
    #Arguments for Higgs
    parser.add_argument('--mixing', action='store_true', help='if passed use test mixing between classes')
    parser.add_argument('--null', action='store_true', help='if passed use null hypothesis, else use alternative')
    parser.add_argument('--n_components', type=int, default=4, help='number of dimensions to use')
    parser.add_argument('--p_poisoning', type=float, default=0.9, help='poisoning probability of class 1 with class 0')
    
    #Argument for sine
    parser.add_argument('--omega', type=float, default=0.05, help='sine frequency')
    
    #Arguments for aggregated tests
    parser.add_argument('--n_bandwidths', type=int, default=5, help='number of bandwidths used in the aggregated test')
    parser.add_argument('--B_2', type=int, default=100, help='number of permutations used for Monte Carlo estimation in agg.')
    parser.add_argument('--B_3', type=int, default=15, help='number of bisection iterations for aggregated test')
    parser.add_argument('--different_compression', action='store_true', help='if passed use different compression calls for different bandwidths')
    
    #Arguments to avoid plotting specific test groups
    parser.add_argument('--no_block_wb', action='store_true', help='if passed do not plot block_wb tests')
    parser.add_argument('--no_incomplete_wb', action='store_true', help='if passed do not plot incomplete_wb tests')
    parser.add_argument('--no_block_asymp', action='store_true', help='if passed do not plot block_asymp tests')
    parser.add_argument('--no_incomplete_asymp', action='store_true', help='if passed do not plot incomplete_asymp tests')
    parser.add_argument('--no_ctt', action='store_true', help='if passed do not plot ctt tests')
    parser.add_argument('--no_rff', action='store_true', help='if passed do not plot rff tests')
    parser.add_argument('--no_ctt_rff', action='store_true', help='if passed do not plot ctt_rff tests')
    
    #Argument to avoid plotting asymptotic block (a) and asymptotic incomplete (a)
    parser.add_argument('--no_asymp_a', action='store_true', help='if passed do not plot asymptotic (a) tests')
    
    args = parser.parse_args()
    
    #Reset default values depending on args.name
    if args.name == 'gaussians':
        args.d = 10
        
    if args.name == 'blobs':
        args.d = 2
        
    if args.name == 'Higgs':
        args.d = args.n_components
        if args.p_poisoning > 0:
            args.mixing = True
        
    if args.name == 'sine':
        args.d = 10
    
    args.interactive = False
    
    util_tests.get_attributes_tests(args)
            
    #Build list of test groups
    if args.aggregated:
        test_groups = ['incomplete_wb', 'ctt', 'rff']
    else:
        test_groups = ['block_wb', 'incomplete_wb', 'block_asymp', 'incomplete_asymp', 'ctt', 'rff', 'ctt_rff']
    
    #Store no-compute choices for each test group
    no_compute = dict()
    no_compute['block_wb'] = args.no_block_wb
    no_compute['incomplete_wb'] = args.no_incomplete_wb
    no_compute['block_asymp'] = args.no_block_asymp
    no_compute['incomplete_asymp'] = args.no_incomplete_asymp
    no_compute['asymp_a'] = args.no_asymp_a
    no_compute['ctt'] = args.no_ctt
    no_compute['rff'] = args.no_rff
    no_compute['ctt_rff'] = args.no_ctt_rff
    
    if ((args.name == 'gaussians' and args.mean_diff == 0) or (args.name == 'blobs' and args.epsilon == 1) or (args.name == 'EMNIST' and args.p_even == 0.5) or (args.name == 'Higgs' and args.p_poisoning == 1)) and not args.show_exact and not args.aggregated:
        no_compute['block_wb'] = True
        no_compute['incomplete_wb'] = True
        no_compute['ctt'] = True
        no_compute['rff'] = True
        no_compute['ctt_rff'] = True
    if args.name == 'Higgs' or args.name == 'blobs':
        no_compute['rff'] = True
        no_compute['ctt_rff'] = True
        no_compute['block_asymp'] = True
        no_compute['incomplete_asymp'] = True
    #elif ((args.name == 'gaussians' and args.mean_diff == 0) or (args.name == 'blobs' and args.epsilon == 1) or (args.name == 'EMNIST' and args.p_even == 0.5) or (args.name == 'Higgs' and args.p_poisoning == 1)) and args.show_exact:
        #print(f'null distribution and args.show_exact')
        #no_compute['block_asymp'] = True
        #no_compute['incomplete_asymp'] = True
        
    for group in test_groups:
        print(f'{group}: no_compute={no_compute[group]}')
    
    used_test_groups = []
    for group in test_groups:
        if not no_compute[group]:
            used_test_groups.append(group)
    if (not no_compute['block_asymp'] or not no_compute['incomplete_asymp']) and no_compute['asymp_a']:
        used_test_groups.append('no_asymp_a')
    if args.log_time_scale:
        used_test_groups.append('log_time_scale')
    if args.small_times:
        used_test_groups.append('small_times')
    if args.wilson_intervals:
        used_test_groups.append('wilson')
    if args.aggregated:
        used_test_groups.append('aggregated')
    if args.no_violations:
        used_test_groups.append('no_violations')
    if args.long_times != 0:
        if args.long_times.is_integer():
            str_long_times = str(int(args.long_times))
        else:
            str_long_times = str(args.long_times)
        used_test_groups.append('long_times'+'_'+str_long_times)
    formatted_used_test_groups = util_classes.format_int_list(used_test_groups)
        
    joint_resdir = util_classes.get_joint_group_directories(args, test_groups, aggregated=args.aggregated)
    
    joint_filename = util_classes.get_joint_filename(args, test_groups, aggregated=args.aggregated)
    
    joint_fname = util_classes.get_fname_joint(args,test_groups,joint_resdir,joint_filename)
    
    joint_group_results_dict = dict()
    
    for group in test_groups:
        if not no_compute[group]:
            print(f'joint_fname[group]: {joint_fname[group]}')
            joint_group_results_dict[group] = pickle.load(open(joint_fname[group], 'rb'))
            
    if not os.path.exists('figures'+'_'+args.name+'_camera_ready'):
        os.makedirs('figures'+'_'+args.name+'_camera_ready')
            
    plt.figure(figsize=(5.5,3))
    fig_rejection(args, test_groups, joint_group_results_dict, aggregated=args.aggregated)
    
    if args.name == 'gaussians':
        fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.mean_diff)+'_'+formatted_used_test_groups+'.pdf'

    elif args.name == 'blobs':
        fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.grid_size)+'_'+str(args.epsilon)+'_'+formatted_used_test_groups+'.pdf' 

    elif args.name == 'MNIST' or args.name == 'EMNIST':
        fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.p_even)+'_'+formatted_used_test_groups+'.pdf' 
        
    elif args.name == 'Higgs':
        if args.mixing:
            fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.mixing)+'_'+str(args.p_poisoning)+'_'+formatted_used_test_groups+'.pdf'
        else:
            fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.mixing)+'_'+str(args.null)+'_'+formatted_used_test_groups+'.pdf'
      
    elif args.name == 'sine':
        fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.omega)+'_'+formatted_used_test_groups+'.pdf'
        
    #Here new
    pplot()
    plt.tight_layout()  
    #End of new
            
    print('Figure file:'+'figures'+'_'+args.name+'_camera_ready/'+'sm='+str(args.s_rff)+'_sp='+str(args.s_permute)+'_'+fig_file)
    
    if not no_compute['ctt_rff']:
        plt.savefig(f'figures'+'_'+args.name+'_camera_ready/'+'sm='+str(args.s_rff)+'_sp='+str(args.s_permute)+'_'+fig_file, bbox_inches='tight', pad_inches=0)
    elif not no_compute['ctt']:
        plt.savefig(f'figures'+'_'+args.name+'_camera_ready/'+'sp='+str(args.s_permute)+'_'+fig_file, bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(f'figures'+'_'+args.name+'_camera_ready/'+fig_file, bbox_inches='tight', pad_inches=0)
            
            
            
