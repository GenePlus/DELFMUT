
##Example
An example of running DELFMUT on a linux system:
```
outputPath_prefix=../DELFMUT/maxDep45000/output && mkdir -p ${outputPath_prefix}
s=0.6
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


