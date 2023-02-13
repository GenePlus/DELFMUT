# -*- coding: utf-8 -*-
"""
Created on May 26 2022

@author: wuguiying
"""

import argparse
import logging
from numpy import minimum, delete, floor, savez, nanmax, std, mean, median, intersect1d, logical_and, logical_or, sum, sort, zeros, cumsum, asarray, where, ceil, random, hstack, concatenate, array, arange   
import numpy as np
import pandas as pd
import os
from re import split
from scipy.optimize import fsolve
from functools import  partial
from timeit import default_timer
from multiprocessing import Pool as mp_Pool
from multiprocessing import cpu_count as mp_cpu_count

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family']='sans-serif'
rcParams['font.sans-serif']=['Arial']
import seaborn as sns



class Template():
    TempID = 0
    TempType = ''
    mutLabel = 0
    R1_ID_array = array([])
    R1_count = 0
    R2_ID_array = array([])
    R2_count = 0
    
    def __init__(self, TempID, TempType, mutLabel, R1_ID_array, R1_count, R2_ID_array, R2_count):

        self.TempID = TempID
        self.TempType = TempType
        self.mutLabel = mutLabel
        self.R1_ID_array = R1_ID_array
        self.R1_count = R1_count
        self.R2_ID_array = R2_ID_array
        self.R2_count = R2_count



def posNegBino_mean_fun(x, mean_no0, alpha_with0):
    f = x / (1 - (1 + alpha_with0 * x) ** (-1 / alpha_with0) ) - mean_no0
    return f


def posNB_rand_fun(n_rand, mean_no0, alpha_with0_NB=0.3):
    mean_with0_NB = fsolve(posNegBino_mean_fun, x0=2.0, args=(mean_no0,alpha_with0_NB))[0] 
    var_with0_NB = mean_with0_NB * (1 + alpha_with0_NB * mean_with0_NB)
    
    p_NB = mean_with0_NB / var_with0_NB  
    r_NB = 1 / alpha_with0_NB 
    
    rand_array_with0 = random.negative_binomial(n=r_NB, p=p_NB, size=n_rand)
    rand_array_no0 = delete(rand_array_with0, obj=where(rand_array_with0==0)[0]) 
    
    n_lack = n_rand - len(rand_array_no0)
    
    while n_lack > 0:
        rand_array_with0 = random.negative_binomial(n=r_NB, p=p_NB, size=n_lack)
        temp_rand_array_no0 = delete(rand_array_with0, obj=where(rand_array_with0==0)[0])  
        rand_array_no0 = hstack(( rand_array_no0, temp_rand_array_no0 ))
        n_lack = n_rand - len(rand_array_no0)
    
    return rand_array_no0


def fine_tune_no0(rand_array, n_rand, expect_sum):    
    if len(rand_array) != n_rand:
        return False
    
    if sum(rand_array) == expect_sum:
        return rand_array
    
    else:
        rand_array = ceil( rand_array * (expect_sum/sum(rand_array)) ).astype(int)
        
        if sum(rand_array) > expect_sum:
            n_diff = sum(rand_array) - expect_sum
           
            while n_diff > 0:
                index_array = where(rand_array>1)[0]

                n_candidate = len(index_array)

                if n_diff <= n_candidate:
                    sampled_index_array = random.choice(index_array, n_diff, replace=False) 
                    rand_array[sampled_index_array] = rand_array[sampled_index_array] - 1 
                    n_diff = 0
                   
                else:
                    rand_array[index_array] = rand_array[index_array] - 1 
                    n_diff = n_diff - n_candidate
                   
            return rand_array
                   
        elif sum(rand_array) == expect_sum:
            return rand_array
        else:
            return False




def Templates_init_FUN(T, n_DT, n_R1T, n_R2T, R1, R2):
    T_list = [] 
    for i in arange(0,T):
        
        if i >= 0 and i <= n_DT-1:
            T_init = Template(TempID=i, TempType='DoubleStrand', mutLabel=0, 
                              R1_ID_array=array([]), R1_count=0, R2_ID_array=array([]), R2_count=0)

        elif i >= n_DT and i <= n_R1T-1:
            T_init = Template(TempID=i, TempType='SingleStrandPos', mutLabel=0, 
                              R1_ID_array=array([]), R1_count=0, R2_ID_array=array([]), R2_count=0)

        else:
            T_init = Template(TempID=i, TempType='SingleStrandNeg', mutLabel=0, 
                              R1_ID_array=array([]), R1_count=0, R2_ID_array=array([]), R2_count=0)
        
        T_list.append(T_init)
            

    E_R1_perTemp_no0 = R1/n_R1T    
    E_R2_perTemp_no0 = R2/n_R2T   
    
    rand1_array = posNB_rand_fun(n_rand=n_R1T, mean_no0=E_R1_perTemp_no0, alpha_with0_NB=0.3)  
    rand2_array = posNB_rand_fun(n_rand=n_R2T, mean_no0=E_R2_perTemp_no0, alpha_with0_NB=0.3)  
    
    n_read1_array = fine_tune_no0(rand_array=rand1_array, n_rand=n_R1T, expect_sum=R1)
    n_read2_array = fine_tune_no0(rand_array=rand2_array, n_rand=n_R2T, expect_sum=R2)
    
    cumsum_read1_array = cumsum(n_read1_array)        
    cumsum_read2_array = R1 + cumsum(n_read2_array)   
    
    T1_index_array = arange(0,n_R1T)
    for i in arange(0,n_R1T):
        T1_index = T1_index_array[i]
        if i == 0:   
            T_list[T1_index].R1_ID_array = arange(0, cumsum_read1_array[i])
        else:
            T_list[T1_index].R1_ID_array = arange(cumsum_read1_array[i-1], cumsum_read1_array[i])
    
        T_list[T1_index].R1_count = n_read1_array[i]
    
    T2_index_array = hstack(( arange(0,n_DT), arange(n_R1T,T) ))  
    for i in arange(0,n_R2T):   
        T2_index = T2_index_array[i]
        if i == 0:
            T_list[T2_index].R2_ID_array = arange(R1, cumsum_read2_array[i])   
        else:
            T_list[T2_index].R2_ID_array = arange(cumsum_read2_array[i-1], cumsum_read2_array[i])
    
        T_list[T2_index].R2_count = n_read2_array[i]

    return T_list




def mutLabel_init_FUN(T_list, T, Tm):   
    Tm_ID_array = sort( random.choice(arange(0,T), Tm, replace=False) )
    
    Rm1 = 0
    Rm2 = 0
    for i in arange(0, T):   
        if not i in Tm_ID_array:
            T_list[i].mutLabel = 0
        else:
            T_list[i].mutLabel = 1
            Rm1 = Rm1 + T_list[i].R1_count
            Rm2 = Rm2 + T_list[i].R2_count
                        
    Rm = Rm1 + Rm2
    
    return (T_list, Tm_ID_array, Rm, Rm1, Rm2) 



def intersect2D(array1D, array2D):     
    if len(array1D) <= 40:   
        N_rowIntersect_array = zeros(array2D.shape[0]).astype(int)
        for x in array1D:
            N_rowIntersect_array += sum(x == array2D, axis=1)
            
    else:
        N_rowIntersect_array = array([len(intersect1d(array1D,x)) for x in array2D])

    return N_rowIntersect_array



def DS_and_stat_FUN(T_list, targetDEP_array, n_DSrepeat, DEP, Tm, Tm_ID_array, rule_df):
    n_targetDEP = len(targetDEP_array)
    n_rules = rule_df.shape[0] 
    detect_freq_D3array = zeros((n_rules, 1, n_targetDEP)) 
    
    if Tm >= 1 and Tm == floor(Tm).astype(int): 
    
        for targetDEP_i in arange(0, n_targetDEP):
            targetDEP = targetDEP_array[targetDEP_i]
            
            DSread_ID_D2array = zeros((n_DSrepeat, targetDEP)).astype(int) - 1  
            for repeat_j in arange(0, n_DSrepeat):
                DSread_ID_D2array[repeat_j,] = sort( random.choice(arange(0,DEP), targetDEP, replace=False) )              
            
            DSread1_count_D2array = zeros((n_DSrepeat, Tm)).astype(int)
            DSread2_count_D2array = zeros((n_DSrepeat, Tm)).astype(int)
        
            for Tm_k in arange(0, Tm):       
                
                Tm_ID = Tm_ID_array[Tm_k]                
                
                DSread1_count_D2array[:,Tm_k] = intersect2D(T_list[Tm_ID].R1_ID_array, DSread_ID_D2array) 
                DSread2_count_D2array[:,Tm_k] = intersect2D(T_list[Tm_ID].R2_ID_array, DSread_ID_D2array) 
               
            for rule_m in arange(0, n_rules): 
                temp_rule = rule_df.iloc[rule_m,:]
                rule_type = temp_rule[0]
                c = int(temp_rule[1])
                n1 = int(temp_rule[2])
                n2 = int(temp_rule[3])
                
                if rule_type == "inclusion":
                    
                    DS_binary_D2array = logical_or( logical_and(DSread1_count_D2array >= n1, DSread2_count_D2array >= n2),
                                                    logical_and(DSread1_count_D2array >= n2, DSread2_count_D2array >= n1) )   
                
                elif rule_type == "exclusion":
                    
                    if ( (n1 == 1 and n2 == 0) or (n1 == 0 and n2 == 1) ):
                        DS_binary_D2array = DSread1_count_D2array + DSread2_count_D2array >= 2
                        
                    elif ( (n1 == 2 and n2 == 0) or (n1 == 0 and n2 == 2) ):
                        DS_binary_D2array = logical_or( (DSread1_count_D2array + DSread2_count_D2array >= 3),
                                                        logical_and(DSread1_count_D2array == 1, DSread2_count_D2array == 1) )
                        
                    else:
                        logging.error("Only support 'exclusion_1-1+0' and 'exclusion_1-2+0' rules !")
                        return False
                    
                else:
                    logging.error("Please input the right 'rule_type' parameter: 'inclusion' or 'exclusion' !")
                    return False
                    
                DS_nTm_array = sum(DS_binary_D2array, axis=1)  
                detect_freq_D3array[rule_m,0,targetDEP_i] = sum(DS_nTm_array>=c)/len(DS_nTm_array)  
            
        return detect_freq_D3array

    elif Tm == 0:
        return detect_freq_D3array
    else:
        logging.error("Please input the right 'Tm' parameter: a natural number !")
        return False
            


def whole_process_FUN(repeat_whole, T, n_DT, n_R1T, n_R2T, DEP, R1, R2,
                      Tm_float, n_TmSamp_repeat,
                      targetDEP_array, n_DSrepeat, 
                      rule_df, n_rules):   
    
    print("\n\n---- Repeat time of the whole process:", repeat_whole+1, "----")
    
    T_list = Templates_init_FUN(T=T, n_DT=n_DT, n_R1T=n_R1T, n_R2T=n_R2T, R1=R1, R2=R2)
       
    repeatTm_detect_freq_D3array = zeros((n_rules, n_TmSamp_repeat, len(targetDEP_array)))
    
    Tm_int = floor(Tm_float).astype(int)  
    Tm_deci = Tm_float - Tm_int 

    n_TmIntPlusOne = round(n_TmSamp_repeat * Tm_deci)  
    n_TmInt = n_TmSamp_repeat - n_TmIntPlusOne 

    print("\nTm:", Tm_int, "(", n_TmInt, "times ); ", Tm_int+1, "(", n_TmIntPlusOne, "times )")
        
    for repeat_i in arange(0, n_TmSamp_repeat):
    
        if repeat_i < n_TmInt:
            Tm = Tm_int
        else:
            Tm = Tm_int + 1

        T_list, Tm_ID_array, Rm, Rm1, Rm2 = mutLabel_init_FUN(T_list=T_list, T=T, Tm=Tm)
        print('\n\n-- Repeat time of the mutated templates sampling:', repeat_i+1, '/', n_TmSamp_repeat, '--')
        print('Tm_ID_array:', Tm_ID_array)
        print('Rm:', Rm)
        print('Rm1:', Rm1)
        print('Rm2:', Rm2)
    
        temp_detect_freq_D3array = DS_and_stat_FUN(T_list=T_list, targetDEP_array=targetDEP_array, n_DSrepeat=n_DSrepeat, 
                                                   DEP=DEP, Tm=Tm, Tm_ID_array=Tm_ID_array, 
                                                   rule_df=rule_df)
        
        repeatTm_detect_freq_D3array[:,repeat_i,:] = temp_detect_freq_D3array[:,0,:] 
        print('\ntemp_detect_freq_D3array:\n', temp_detect_freq_D3array)

    return repeatTm_detect_freq_D3array



def  one_MASS_VAF_FUN(MASS, T, VAF, DEP=80000,
                      s=0.6, t=0.2, R_bias=0.9, 
                      rule_list=["inclusion_2-1+0"], targetDEP_array=array([2000]), 
                      n_whole_repeat=10, n_TmSamp_repeat=10, n_DSrepeat=100,
                      n_cpu=1):
   
    print("\nDEP:", DEP)
    print("T:", T)
    
    R1 = round( R_bias / (1+R_bias) * DEP ) 
    R2 = DEP - R1    

    n_DT = round(s*T)
    n_R1T = round((s+t)*T)
    n_PosST = n_R1T - n_DT
    n_NegST = T - n_R1T
    n_R2T = T - n_PosST
    
    print("\nn_DT:", n_DT)
    print("n_PosST:", n_PosST)
    print("n_NegST:", n_NegST)

    if R1 < n_R1T or R2 < n_R2T:
        logging.error('The number of reads is smaller than the number of templates: R1 {} < n_R1T {} or R2 {} < n_R2T {}'.format(R1, n_R1T, R2, n_R2T))
        return False 
    
    Tm_float = VAF*T
    print("\nTm_float:", Tm_float)
       
    n_rules = len(rule_list)
    rule_D2list = [split("_|-|\+", x) for x in rule_list]
    rule_df = pd.DataFrame(rule_D2list)
    rule_df.columns = ["rule_type", "c", "n1", "n2"]
    
    print("\ntargetDEP_array:", targetDEP_array)
    
    print("n_whole_repeat:", n_whole_repeat)
    print("n_TmSamp_repeat:", n_TmSamp_repeat)
    print("n_DSrepeat:", n_DSrepeat)

    if Tm_float == 0:  
        n_targetDEP = len(targetDEP_array)
        mean_detect_freq_D2array = zeros((n_rules, n_targetDEP))  
        median_detect_freq_D2array = zeros((n_rules, n_targetDEP))     
        sd_detect_freq_D2array = zeros((n_rules, n_targetDEP))     

    elif Tm_float > 0:
        n_parallel = min(mp_cpu_count()-1, n_cpu, n_whole_repeat)
        print('\n%% Number of running processers:', n_parallel, '%%')
        pool = mp_Pool()
        
        repeatWhole_detect_freq_D3array_list = pool.map(partial(whole_process_FUN, 
                                                                T=T, n_DT=n_DT, n_R1T=n_R1T, n_R2T=n_R2T, DEP=DEP, R1=R1, R2=R2,
                                                                Tm_float=Tm_float, n_TmSamp_repeat=n_TmSamp_repeat,
                                                                targetDEP_array=targetDEP_array, n_DSrepeat=n_DSrepeat, 
                                                                rule_df=rule_df, n_rules=n_rules), 
                                                        arange(0,n_whole_repeat)) 
        
        pool.close()
        pool.join()

        for whole_i in arange(0, n_whole_repeat):
            if whole_i == 0:
                repeatWhole_detect_freq_D3array = repeatWhole_detect_freq_D3array_list[0]
            elif whole_i < n_whole_repeat:
                repeatWhole_detect_freq_D3array = concatenate( (repeatWhole_detect_freq_D3array, 
                                                                repeatWhole_detect_freq_D3array_list[whole_i]), 
                                                              axis=1)   
                
        mean_detect_freq_D2array = mean(repeatWhole_detect_freq_D3array, axis=1)   
        median_detect_freq_D2array = median(repeatWhole_detect_freq_D3array, axis=1)     
        sd_detect_freq_D2array = std(repeatWhole_detect_freq_D3array, axis=1)  

    else: 
        logging.error("Please check whether the input VAF and T are correct !")
        return False

    print("\n\n----Statistical results for different detection rules----")
    for rule_j in arange(0, n_rules): 
        temp_rule = rule_list[rule_j]
        print("\n--Detection rule:", temp_rule)

        rule_j_mean_detect_freq_array = mean_detect_freq_D2array[rule_j,:]   
        rule_j_median_detect_freq_array = median_detect_freq_D2array[rule_j,:]     
        rule_j_sd_detect_freq_array = sd_detect_freq_D2array[rule_j,:]     

        print("(Input) targetDEP_array:", targetDEP_array)
    
        print('(Output) mean_detect_freq_array:', rule_j_mean_detect_freq_array)
        print('(Output) median_detect_freq_array:', rule_j_median_detect_freq_array)
        print('(Output) sd_detect_freq_array:', rule_j_sd_detect_freq_array)
    
    return (mean_detect_freq_D2array, median_detect_freq_D2array, sd_detect_freq_D2array)




def linePlot_FUN(detect_freq_D4array, 
                 MASS_array, VAF_array, rule_list, targetDEP_array,
                 VAFcolor_array,
                 output_path, flag):
    
    y_max = nanmax(detect_freq_D4array) 
    
    n_MASS = len(MASS_array)
    n_rule = len(rule_list)
    n_VAF = len(VAF_array)
    
    plt.figure(figsize=(6*n_rule, 4*n_MASS))

    result_df = pd.DataFrame()
    
    ylabel_flag = split("-", flag)[0] 
    
    for MASS_i in arange(0, n_MASS):
        MASS = MASS_array[MASS_i]
        T = T_array[MASS_i]

        for rule_j in arange(0, n_rule):
            rule = rule_list[rule_j]   
            temp_rule = split("_|-|\+", rule)
            rule_type = temp_rule[0]
            c = temp_rule[1]
            n1 = temp_rule[2]
            n2 = temp_rule[3]
            if rule_type=="inclusion":
                ylabel = ylabel_flag + " of detection frequency (" + c + "-" + n1 + "+" + n2 + ")"               
            elif rule_type=="exclusion":
                ylabel = ylabel_flag + " of detection frequency \n (" + c + "-exclude" + n1 + "+" + n2 + ")"
            
            plt.subplot(n_MASS, n_rule, MASS_i * n_rule + rule_j+1)
            
            for VAF_k in arange(0, n_VAF):
                VAF = VAF_array[VAF_k]
                
                P_array = detect_freq_D4array[MASS_i,VAF_k,rule_j,:]

                plt.plot(targetDEP_array, P_array, c=VAFcolor_array[VAF_k], linewidth=3, label='VAF={}'.format(VAF))
        
                temp_result_df = pd.DataFrame({'MASS':MASS, 'T':T, 'VAF':VAF, 'rule':rule, 'targetDEP':targetDEP_array, 'P':P_array})
                result_df = pd.concat([result_df, temp_result_df], axis=0, ignore_index=True)  
        
            plt.ylim((-0.05,y_max+0.05))
            plt.ylabel(ylabel, fontsize=15)
            
            if y_max <= 1:  
                plt.axhline(y=0.95,ls="--",c='gray',linewidth=1.5)
                
            plt.xticks(fontsize=13, rotation=0)
            plt.yticks(fontsize=13)
            plt.grid(color='gray', ls=':', lw=1, alpha=0.4)
            
            if rule_j == 0:
                plt.title('DNA input: {}ng'.format(MASS), fontsize=17, fontweight='bold')
            
            if rule_j == 0 and MASS_i == 0:
                plt.legend(loc='center right', facecolor='none')
    
                
            if MASS_i == n_MASS - 1:
                plt.xlabel("Sequencing depth", fontsize=15)
    
    plt.tight_layout(h_pad=3, w_pad=1.5)
    
    max_targetDEP = max(targetDEP_array)
    plt.savefig(f'{output_path}/resultPLOT_maxTargetDep%d_%dMASS_%drule_%dVAF_%s.pdf' % (max_targetDEP,n_MASS,n_rule,n_VAF,flag)) 
    plt.savefig(f'{output_path}/resultPLOT_maxTargetDep%d_%dMASS_%drule_%dVAF_%s.png' % (max_targetDEP,n_MASS,n_rule,n_VAF,flag), dpi=600) 
    plt.close()
    
    result_df.to_csv(f'{output_path}/resultDF_maxTargetDep%d_%dMASS_%drule_%dVAF_%s.tsv' % (max_targetDEP,n_MASS,n_rule,n_VAF,flag), index=False, sep='\t') 

    return result_df



def meanSD_linePlot_FUN(mean_detect_freq_D4array, sd_detect_freq_D4array,
                         MASS_array, VAF_array, rule_list, targetDEP_array,
                         VAFcolor_array,
                         output_path, flag):
    
    y_max = min(1.0, nanmax(mean_detect_freq_D4array + sd_detect_freq_D4array)) 
    
    n_MASS = len(MASS_array)
    n_rule = len(rule_list)
    n_VAF = len(VAF_array)
    
    plt.figure(figsize=(6*n_rule, 4*n_MASS))

    result_df = pd.DataFrame()
        
    for MASS_i in arange(0, n_MASS):
        MASS = MASS_array[MASS_i]
        T = T_array[MASS_i]

        for rule_j in arange(0, n_rule):
            rule = rule_list[rule_j]   
            temp_rule = split("_|-|\+", rule)
            rule_type = temp_rule[0]
            c = temp_rule[1]
            n1 = temp_rule[2]
            n2 = temp_rule[3]
            if rule_type=="inclusion":
                ylabel = "Detection frequency (" + c + "-" + n1 + "+" + n2 + ")"               
            elif rule_type=="exclusion":
                ylabel = "Detection frequency \n (" + c + "-exclude" + n1 + "+" + n2 + ")"
            
            plt.subplot(n_MASS, n_rule, MASS_i * n_rule + rule_j+1)
            
            for VAF_k in arange(0, n_VAF):
                VAF = VAF_array[VAF_k]
                
                mean_P_array = mean_detect_freq_D4array[MASS_i,VAF_k,rule_j,:]
                sd_P_array = sd_detect_freq_D4array[MASS_i,VAF_k,rule_j,:]
                lower_error_P_array = minimum(mean_P_array, sd_P_array)    
                upper_error_P_array = minimum(1 - mean_P_array, sd_P_array)    
                error_P_array = concatenate( (lower_error_P_array.reshape(1,-1), 
                                              upper_error_P_array.reshape(1,-1)), 
                                              axis=0)    

                plt.plot(targetDEP_array, mean_P_array, c=VAFcolor_array[VAF_k], linewidth=4, label='VAF={}'.format(VAF))
                plt.errorbar(targetDEP_array, mean_P_array, yerr=error_P_array, 
                             ecolor=VAFcolor_array[VAF_k], elinewidth=2, alpha=0.5,
                             fmt='none',
                             capsize=5, capthick=2
                             )
        
                temp_result_df = pd.DataFrame({'MASS':MASS, 'T':T, 'VAF':VAF, 'rule':rule, 'targetDEP':targetDEP_array, 'mean_P':mean_P_array, 'sd_P':sd_P_array})
                result_df = pd.concat([result_df, temp_result_df], axis=0, ignore_index=True)  
        
            plt.ylim((-0.05,y_max+0.05))
            plt.ylabel(ylabel, fontsize=15)
            
            if y_max <= 1:  
                plt.axhline(y=0.95,ls="--",c='gray',linewidth=1.5)
                
            plt.xticks(fontsize=13, rotation=0)
            plt.yticks(fontsize=13)
            plt.grid(color='gray', ls=':', lw=1, alpha=0.4)
            
            if rule_j == 0:
                plt.title('DNA input: {}ng'.format(MASS), fontsize=17, fontweight='bold')
            
            if rule_j == 0 and MASS_i == 0:
                plt.legend(loc='center right', facecolor='none')
                
            if MASS_i == n_MASS - 1:
                plt.xlabel("Sequencing depth", fontsize=15)
    
    plt.tight_layout(h_pad=3, w_pad=1.5)
    
    max_targetDEP = max(targetDEP_array)
    plt.savefig(f'{output_path}/resultPLOT_maxTargetDep%d_%dMASS_%drule_%dVAF_%s.pdf' % (max_targetDEP,n_MASS,n_rule,n_VAF,flag)) 
    plt.savefig(f'{output_path}/resultPLOT_maxTargetDep%d_%dMASS_%drule_%dVAF_%s.png' % (max_targetDEP,n_MASS,n_rule,n_VAF,flag), dpi=600) 
    plt.close()
    
    result_df.to_csv(f'{output_path}/resultDF_maxTargetDep%d_%dMASS_%drule_%dVAF_%s.tsv' % (max_targetDEP,n_MASS,n_rule,n_VAF,flag), index=False, sep='\t') 

    return result_df




def heatmapPlot_FUN(mean_detect_freq_D4array, 
                    MASS_array, VAF_array, rule_list, targetDEP_array,
                    output_path, flag,
                    heatmap_colormap="Blues"):

    n_rule = len(rule_list)
    n_VAF = len(VAF_array)
    n_MASS = len(MASS_array)    

    for rule_j in arange(0, n_rule):
        rule = rule_list[rule_j]   
        temp_rule = split("_|-|\+", rule)
        rule_type = temp_rule[0]
        c = temp_rule[1]
        n1 = temp_rule[2]
        n2 = temp_rule[3]
        if rule_type=="inclusion":
            rule_label = c + "-" + n1 + "+" + n2             
        elif rule_type=="exclusion":
            rule_label = c + "-exclude" + n1 + "+" + n2
            
        plt.figure(figsize=(9, 3*n_VAF))
        
        for VAF_k in arange(0, n_VAF):
            VAF = VAF_array[VAF_k]
            
            plt.subplot(n_VAF, 1, VAF_k+1)
            
            P_D2array = mean_detect_freq_D4array[:,VAF_k,rule_j,:]  
            P_df = pd.DataFrame(P_D2array, index=MASS_array, columns=targetDEP_array)
                       
            sns.heatmap(P_df, 
                        annot=True, fmt='.2f', annot_kws={"fontsize":14},
                        vmin=0, vmax=1, 
                        square=True, 
                        cmap=heatmap_colormap, cbar_kws={"pad":0.01, "aspect":15})            
                
            plt.xticks(fontsize=14, rotation=0)
            plt.yticks(fontsize=14)
            
            plt.xlabel("Sequencing depth", fontsize=15)
            plt.ylabel("DNA input (ng)", fontsize=15)
            
            plt.title('VAF = {};  {}'.format(VAF, rule_label), fontsize=16, fontweight='bold')           
    
        plt.tight_layout(h_pad=3)
        
        max_targetDEP = max(targetDEP_array)
        plt.savefig(f'{output_path}/decisionPLOT_maxTargetDep%d_%dMASS_%dVAF_%s_Rule_%s.pdf' % (max_targetDEP, n_MASS, n_VAF, flag, rule)) 
        plt.savefig(f'{output_path}/decisionPLOT_maxTargetDep%d_%dMASS_%dVAF_%s_Rule_%s.png' % (max_targetDEP, n_MASS, n_VAF, flag, rule), dpi=600) 
        plt.close()
    
    return True



def main(output_path, 
         s=0.4, R_bias=1, T_bias=1,  
         MASS_array=array([30, 50, 80]), T_array=array([5159, 7787, 12233]), 
         VAF_array=array([0.0002, 0.0005, 0.001, 0.002, 0.005, 0.008]), VAFcolor_array=array(['orangered','goldenrod','limegreen','hotpink','deepskyblue','darkviolet']),
         rule_list=["inclusion_2-1+0", "inclusion_4-1+0", "exclusion_1-1+0", "exclusion_1-2+0", "inclusion_1-2+1", "inclusion_2-2+1", "inclusion_1-2+2", "inclusion_1-3+3"], 
         targetDEP_array=array([2000,5000,8000,10000,15000,20000,25000,30000,35000,40000,45000]), 
         DEP=80000,
         n_whole_repeat=10, n_TmSamp_repeat=10, n_DSrepeat=10, 
         n_cpu=1,
         heatmap_colormap="hot_r"):
    
    os.makedirs(output_path, exist_ok=True)

    if s > 1 or s < 0:
        logging.error('The ratio of Double-stranded Templates: s {} > 1 or s {} < 0'.format(s, s))
        return False   
    
    if R_bias < 0:
        logging.error('Reads-level strand bias: R_bias {} < 0'.format(R_bias))
        return False
           
    if T_bias < s or T_bias > 1 / s:
        logging.error('Template-level strand bias: T_bias {} < the ratio of Double-stranded Template s {} or > the reciprocal of ratio of Double-stranded Template 1/s {}'.format(T_bias, s, 1/s))
        return False

    t = (T_bias-s) / (1+T_bias) 
    
    n_MASS = len(MASS_array)
    n_VAF = len(VAF_array)
    n_rule = len(rule_list)
    n_targetDEP = len(targetDEP_array)
    
    print("\nMASS_array(ng):", MASS_array)
    print("T_array:", T_array)
    print("VAF_array:", VAF_array)
    print("rule_list:", rule_list)
    
    print("\ns:", s)
    print("R_bias:", R_bias)
    print("T_bias:", T_bias)
    
    if n_MASS != len(T_array):
        logging.error('The length of "MASS_array": {} should be the same with the length of "T_array": {} '.format(n_MASS, len(T_array)))
        return False
    
    
    if n_VAF != len(VAFcolor_array):
        logging.error('The length of "VAF_array": {} should be the same with the length of "VAFcolor_array": {} '.format(n_VAF, len(VAFcolor_array)))
        return False
    
    max_targetDEP = max(targetDEP_array)
    if DEP < max_targetDEP + 10000:
        logging.error('Paremeter "DEP": {} should be >= max(targetDEP_array)+10000: {} '.format(DEP, max_targetDEP+10000))
        return False    

    mean_detect_freq_D4array = zeros(( n_MASS, n_VAF, n_rule, n_targetDEP ))
    median_detect_freq_D4array = zeros(( n_MASS, n_VAF, n_rule, n_targetDEP ))
    sd_detect_freq_D4array = zeros(( n_MASS, n_VAF, n_rule, n_targetDEP ))
    for MASS_i in arange(0, n_MASS):
        MASS = MASS_array[MASS_i]
        T = T_array[MASS_i]
        print("\n\n\n**** MASS(ng):", MASS, ", in MASS_array", MASS_array, "****")
        print("**** T:", T, ", in T_array", T_array, "****")
        
        for VAF_j in arange(0, n_VAF):
            VAF = VAF_array[VAF_j]
            print("\n\n**** VAF:", VAF, ", in VAF_array", VAF_array, "****")
    

            mean_detect_freq_D2array, median_detect_freq_D2array, sd_detect_freq_D2array = one_MASS_VAF_FUN(MASS=MASS, T=T, VAF=VAF, DEP=DEP,
                                                                                            s=s, t=t, R_bias=R_bias, 
                                                                                            rule_list=rule_list, targetDEP_array=targetDEP_array, 
                                                                                            n_whole_repeat=n_whole_repeat, n_TmSamp_repeat=n_TmSamp_repeat, n_DSrepeat=n_DSrepeat,
                                                                                            n_cpu=n_cpu)
            mean_detect_freq_D4array[MASS_i,VAF_j,:,:] = mean_detect_freq_D2array
            median_detect_freq_D4array[MASS_i,VAF_j,:,:] = median_detect_freq_D2array
            sd_detect_freq_D4array[MASS_i,VAF_j,:,:] = sd_detect_freq_D2array
   
    np.seterr(divide='ignore', invalid='ignore')   
    CV_detect_freq_D4array = sd_detect_freq_D4array / mean_detect_freq_D4array  

    savez( f'{output_path}/detectFreq_D4array_maxTargetDepth%d_%dMASS_%drule_%dVAF.npz' % (max_targetDEP, n_MASS, n_rule, n_VAF),
            mean_detect_freq_D4array=mean_detect_freq_D4array,
            median_detect_freq_D4array=median_detect_freq_D4array,
            sd_detect_freq_D4array=sd_detect_freq_D4array,
            CV_detect_freq_D4array=CV_detect_freq_D4array,
            Dimnames_D0MASS_D1VAF_D2detectRule_D3targetDEP_tuple=array((MASS_array, VAF_array, rule_list, targetDEP_array), dtype=object),
            parameters_dict={ 'output_path': output_path, 
                              's': s, 'R_bias': R_bias, 'T_bias': T_bias,  
                              'MASS_array': MASS_array, 'T_array': T_array, 
                              'VAF_array': VAF_array, 'VAFcolor_array': VAFcolor_array,
                              'rule_list': rule_list, 
                              'targetDEP_array': targetDEP_array, 
                              'DEP': DEP,
                              'n_whole_repeat': n_whole_repeat, 'n_TmSamp_repeat': n_TmSamp_repeat, 'n_DSrepeat': n_DSrepeat, 
                              'n_cpu': n_cpu,
                              'heatmap_colormap': heatmap_colormap
                            }
          )

    plotData_list = [mean_detect_freq_D4array, median_detect_freq_D4array, sd_detect_freq_D4array, CV_detect_freq_D4array]
    flag_list = ["Mean", "Median", "SD", "CV"]
    print('\n\n\n----------Plotting----------')
    for i in arange(0, len(plotData_list)):
        flag = flag_list[i] + "-" + str(n_whole_repeat) + "-" + str(n_TmSamp_repeat) + "-" + str(n_DSrepeat) 
        linePlot_FUN(detect_freq_D4array=plotData_list[i], 
                 MASS_array=MASS_array, VAF_array=VAF_array, rule_list=rule_list, targetDEP_array=targetDEP_array,
                 VAFcolor_array=VAFcolor_array,
                 output_path=output_path, flag=flag)
        print('\n\n--Detection frequency (', flag, '):\n', plotData_list[i])        
        
    print('\n\n\n----------Mean with error bar plotting----------')
    flag_meanSD = "meanSD" + "-" + str(n_whole_repeat) + "-" + str(n_TmSamp_repeat) + "-" + str(n_DSrepeat)
    
    meanSD_linePlot_FUN(mean_detect_freq_D4array=mean_detect_freq_D4array, sd_detect_freq_D4array=sd_detect_freq_D4array,
                        MASS_array=MASS_array, VAF_array=VAF_array, rule_list=rule_list, targetDEP_array=targetDEP_array,
                        VAFcolor_array=VAFcolor_array,
                        output_path=output_path, flag=flag_meanSD)

    print('\n\n\n----------Heatmap plotting----------')
    flag_heatmap = "Mean-" + str(n_whole_repeat) + "-" + str(n_TmSamp_repeat) + "-" + str(n_DSrepeat)
    
    heatmapPlot_FUN(mean_detect_freq_D4array=mean_detect_freq_D4array, 
                    MASS_array=MASS_array, VAF_array=VAF_array, rule_list=rule_list, targetDEP_array=targetDEP_array,
                    output_path=output_path, flag=flag_heatmap,
                    heatmap_colormap=heatmap_colormap)  

    return True
   
 



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', required=True,
                        help='Output directory.')
    
    parser.add_argument('-s', '--s', type=float, default=0.4, required=False,
                        help='The ratio of Double-stranded Templates (DT), 0.4 by default.')
    parser.add_argument('-r', '--R_bias', type=float, default=1.0, required=False,
                        help='Reads-level strand bias, 1.0 by default.')
    parser.add_argument('-t', '--T_bias', type=float, default=1.0, required=False,   
                        help='Template-level strand bias, 1.0 by default.')
    
    parser.add_argument('-M', '--MASS_array', nargs='+', type=float, default=[30], required=False,
                        help='The 1d list/array of DNA input mass, default is [30], one-to-one correspond to "T_array".')
    parser.add_argument('-T', '--T_array', nargs='+', type=int,  default=[4000], required=False,   
                        help='The 1d list/array of the primary templates number in the saturated state, default is [4000], one-to-one correspond to "MASS_array".')
    parser.add_argument('-V', '--VAF_array', nargs='+', type=float,  default=[0.0002], required=False,
                        help='The 1d list/array of VAF for the mutation sites, default is [0.0002], one-to-one correspond to "VAFcolor_array".')
    parser.add_argument('-C', '--VAFcolor_array', nargs='+', type=str,  default=["orangered"], required=False,
                        help='The 1d list/array of colors for the VAFs, default is ["orangered"], one-to-one correspond to "VAF_array".')
    parser.add_argument('-L', '--rule_list', nargs='+', type=str,  default=["inclusion_4-1+0"], required=False,
                        help='The 1d list of detection rules for the mutation sites, default is ["inclusion_4-1+0"].')
    parser.add_argument('-D', '--targetDEP_array', nargs='+', type=int, default=[5000,10000,15000,20000], required=False,
                        help='The 1d list/array of target downsampling depth, default is [5000,10000,15000,20000].')
    
    parser.add_argument('-P', '--DEP', type=int, default=60000, required=False,
                        help='The primary sequencing depth in the saturated state, minimum 60000 by default, should be greater than the maximum of "targetDEP_array".')
    
    parser.add_argument('-x', '--n_whole_repeat', type=int, default=10, required=False,
                        help='Repetition number for the whole process, 10 by default.')
    parser.add_argument('-y', '--n_TmSamp_repeat', type=int, default=10, required=False,
                        help='Repetition number for the mutated templates sampling, 10 by default.')
    parser.add_argument('-z', '--n_DSrepeat', type=int, default=10, required=False,
                        help='Repetition number for the downsampling procedure, 10 by default.')
    
    parser.add_argument('-n', '--n_cpu', type=int, default=1, required=False, 
                        help='Parallel number for the repetition of the whole process, should be <= n_whole_repeat, 1 by default.')
    
    parser.add_argument('-c', '--heatmap_colormap', type=str, default="hot_r", required=False,
                        help='The colormap used for the heatmap plotting, "hot_r" by default.')

    parser.add_argument('-d', '--debug', action='store_true')

    args = parser.parse_args()
    
    logging.basicConfig(format="[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)s] %(message)s",
                        level=logging.DEBUG if args.debug else logging.INFO)
    
    start = default_timer()
    
    MASS_array = asarray(args.MASS_array)
    T_array = asarray(args.T_array)
    VAF_array = asarray(args.VAF_array)
    VAFcolor_array = asarray(args.VAFcolor_array)
    targetDEP_array = asarray(args.targetDEP_array)

    DEP = max(args.DEP, 60000, max(targetDEP_array) + 10000)
    
    main(args.output_path, 
         args.s, args.R_bias, args.T_bias,  
         MASS_array, T_array, 
         VAF_array, VAFcolor_array,
         args.rule_list, 
         targetDEP_array, 
         DEP,
         args.n_whole_repeat, args.n_TmSamp_repeat, args.n_DSrepeat, 
         args.n_cpu,
         args.heatmap_colormap)
    
    end = default_timer()
    print('\nRunning time: %s Second'%(end-start))
    
    
