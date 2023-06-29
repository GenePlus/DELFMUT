# User Guide of DELFMUT

## Introduction
DELFMUT is a sequencing Depth Estimation model designed for the stable detection of Low-Frequency MUTations in duplex sequencing.



## Install
The following python packages need to be installed before running DELFMUT:
* argparse
* logging
* numpy
* pandas
* os
* re
* scipy
* functools
* timeit
* multiprocessing
* matplotlib
* seaborn



## Usage
```
Usage: DELFMUT.py [-h] -o OUTPUT_PATH [-s S] [-r R_BIAS] [-t T_BIAS] [-M MASS_ARRAY [MASS_ARRAY ...]]
                  [-T T_ARRAY [T_ARRAY ...]] [-V VAF_ARRAY [VAF_ARRAY ...]] [-C VAFCOLOR_ARRAY [VAFCOLOR_ARRAY ...]]
                  [-L RULE_LIST [RULE_LIST ...]] [-D TARGETDEP_ARRAY [TARGETDEP_ARRAY ...]] [-P DEP]
                  [-x N_WHOLE_REPEAT] [-y N_TMSAMP_REPEAT] [-z N_DSREPEAT] [-n N_CPU] [-c HEATMAP_COLORMAP] [-d]

options:
  -h, --help            Show this help message and exit
  -o, --output_path     Output directory (output_path).
  -s, --s               The ratio of Double-stranded Templates (s), 0.55 by default.
  -r, --R_bias          Reads-level strand bias (R_bias), 1.0 by default.
  -t, --T_bias          Template-level strand bias (T_bias), 1.0 by default.
  -M, --MASS_array [MASS_ARRAY ...]
                        The 1d list/array of DNA input (MASS_array), default is [30], one-to-one corresponding to "T_array".
  -T, --T_array [T_ARRAY ...]
                        The 1d list/array of the number of templates in the saturated state (T_array), default is [3892], one-to-one corresponding to "MASS_array".
  -V, --VAF_array [VAF_ARRAY ...]
                        The 1d list/array of VAF of the mutations (VAF_array), default is [0.0002], one-to-one corresponding to "VAFcolor_array".
  -C, --VAFcolor_array [VAFCOLOR_ARRAY ...]
                        The 1d list/array of colors for the VAFs (VAFcolor_array), default is ["orangered"], one-to-one corresponding to "VAF_array".
  -L, --rule_list [RULE_LIST ...]
                        The 1d list of mutation detection rules (rule_list), default is ["inclusion_4-1+0"].
  -D, --targetDEP_array [TARGETDEP_ARRAY ...]
                        The 1d list/array of target sequencing depth for down-sampling (targetDEP_array), default is [5000,10000,15000,20000].
  -P, --DEP             The raw sequencing depth in the saturated state (DEP), minimum 60000 by default, should be greater than the maximum of "targetDEP_array".
  -x, --n_whole_repeat  Repetition number for the whole process from the templates initialization to the down-sampling procedure (n_whole_repeat), 10 by default.
  -y, --n_TmSamp_repeat
                        Repetition number for the generation of mutated templates and reads (n_TmSamp_repeat), 10 by default.
  -z, --n_DSrepeat      Repetition number for the downsampling procedure (n_DSrepeat), 10 by default.
  -n, --n_cpu           Parallel number for the repetition of the whole process (n_cpu), should be <= n_whole_repeat, 1 by default.
  -c, --heatmap_colormap
                        The colormap used for the heatmap plotting (heatmap_colormap), "hot_r" by default.
  -d, --debug
```




## Example
An example of running DELFMUT on a linux system:
```
outputPath_prefix=../DELFMUT/maxDep45000/output && mkdir -p ${outputPath_prefix}
s=0.55
R_bias=1.0
T_bias=1.0
output_path=${outputPath_prefix}/s${s}/Rbias${R_bias} && mkdir -p ${output_path}

echo -e "\n%%---- DELFMUT model running on current parameter combination ----%%"
echo "(Parameter 1) output_path: ${output_path}"
echo "(Parameter 2) s: ${s}"
echo "(Parameter 3) R_bias: ${R_bias}"
echo "(Parameter 4) T_bias: ${T_bias}"

MASS_array=(30 50 80)
T_array=(3892 5772 9268)

#VAF_array=(0.0002 0.0005 0.001 0.002 0.005 0.008) 
VAF_array=(0.0002 0.0005 0.001) 

#VAFcolor_array=(orangered goldenrod limegreen hotpink deepskyblue darkviolet)
VAFcolor_array=(orangered goldenrod limegreen)

rule_list=(inclusion_2-1+0  \
           inclusion_4-1+0  \
           inclusion_2-2+0  \
           inclusion_1-1+1  \
           inclusion_2-1+1  \
           inclusion_1-2+2  \
           inclusion_1-3+3) 

#targetDEP_array=(2000 5000 8000 10000 15000 20000 25000 30000 35000 40000 45000)
targetDEP_array=(10000 20000 30000 40000)

echo "(Parameter 5) MASS_array: ${MASS_array[@]}"
echo "(Parameter 6) T_array: ${T_array[@]}"
echo "(Parameter 7) VAF_array: ${VAF_array[@]}"
echo "(Parameter 8) VAFcolor_array: ${VAFcolor_array[@]}"
echo "(Parameter 9) rule_list: ${rule_list[@]}"
echo "(Parameter 10) targetDEP_array: ${targetDEP_array[@]}"

DEP=60000    #not required: DEP = max(args.DEP, 60000, max(targetDEP_array) + 10000)
echo "(Parameter 11) DEP: ${DEP}"

n_whole_repeat=100
n_TmSamp_repeat=100
n_DSrepeat=100

n_cpu=50

heatmap_colormap=hot_r

echo "(Parameter 12) n_whole_repeat: ${n_whole_repeat}"
echo "(Parameter 13) n_TmSamp_repeat: ${n_TmSamp_repeat}"
echo "(Parameter 14) n_DSrepeat: ${n_DSrepeat}"
echo "(Parameter 15) n_cpu: ${n_cpu}"
echo "(Parameter 16) heatmap_colormap: ${heatmap_colormap}"


#DELFMUT model, 16 input parameters

python3 DELFMUT.py \
  --output_path ${output_path} \
  --s ${s} \
  --R_bias ${R_bias} \
  --T_bias ${T_bias} \
  --MASS_array ${MASS_array[@]} \
  --T_array ${T_array[@]} \
  --VAF_array ${VAF_array[@]} \
  --VAFcolor_array ${VAFcolor_array[@]} \
  --rule_list ${rule_list[@]} \
  --targetDEP_array ${targetDEP_array[@]} \
  --DEP ${DEP} \
  --n_whole_repeat ${n_whole_repeat} \
  --n_TmSamp_repeat ${n_TmSamp_repeat} \
  --n_DSrepeat ${n_DSrepeat} \
  --n_cpu ${n_cpu} \
  --heatmap_colormap ${heatmap_colormap}
```



## output
- "resultPLOT_*": The line plots of DELFMUT's results under different combinations of input parameters.
- "decisionPLOT_*": The heatmaps of DELFMUT's results under different combinations of input parameters.
- "resultDF_*": The dataframes of DELFMUT's results under different combinations of input parameters.
- "detectFreq_D4array_*": The multiple result arrays and input parameters of DELFMUT saved as npz data.

The filenames contain "*Mean*", "*SD*", "*meanSD*", "*Median*" and "*CV*" repesent the mean, SD(standard deviation), mean+/-SD and CV(coefficient of variation) of the detection frequencies, respectively. 



## Contact
```
Guiying Wu (email: wuguiying_start@163.com)
Ke Wang (email: wangke@geneplus.org.cn)
Huan Fang (email: fanghuan@geneplus.org.cn)
```


