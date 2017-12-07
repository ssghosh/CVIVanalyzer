import linecache
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import os
import argparse

parser = argparse.ArgumentParser(description='Plot CVIV data')
parser.add_argument('filenames', metavar='F', type=str, nargs='+',
                            help='files to be plotted')
parser.add_argument('--same', action='store_true', help="Plot all files \
on single canvas (otherwise they will all be separate pngs).\
It is assumed you will do this with only CV or IV files rather\
than a mix.")
parser.add_argument('--outputname', action='store', help="Name \
of the single output file. If this is not set the file will \
be named after the input file. If used without --same this\
will be used as the base name for all output files, with \
numbers to distinguish them.")

args = parser.parse_args()
args_dict = vars(args)

#filename = sys.argv[1]
#outfilename = sys.argv[2]

#yaxis_range = [0,3.5E21]
yaxis_range = None
cvfreqs = [10000.]
#cvfreqs = [1000.,10000.]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
def extract_iv(filename):
    with open(filename,"r") as f:
        full_lines = [line for line in f]
        idx = [i for i, line in enumerate(full_lines) if "BiasVoltage" in line][0]
        headers = full_lines[idx].split('\t')
        iv_idx = headers.index("Current_Avg")
        gr_iv_idx = headers.index("GR Current_Avg")
        temp_idx = headers.index("Temperature")

        iv = []
        gr_iv = []
        bv = []
        temp = []

        data = full_lines[idx+1:]
        for line in data:
            words = line.split()
            if words[0][0] == "$":
                continue
            bv.append(abs(float(words[0])))
            temp.append(float(words[temp_idx]))
            iv.append(abs(float(words[iv_idx])))
            gr_iv.append(abs(float(words[gr_iv_idx])))

        return bv, iv, gr_iv, temp 

def extract_ivs(filepaths):
    biasring_iv_data = {}
    guardring_iv_data = {}
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        bv, iv, gr_iv, temp = extract_iv(filepath)
        if "GRswitched" in filename:
            guardring_iv_data[filename] = {'bv' : bv, 'gr_iv' : gr_iv, 'temp' : temp}
        else:
            biasring_iv_data[filename] = {'bv' : bv, 'iv' : iv, 'temp' : temp}
    return biasring_iv_data, guardring_iv_data
    
def extract_cv(filename):
    with open(filename,"r") as f:
        full_lines = [line for line in f]
        idx = [i for i, line in enumerate(full_lines) if "BiasVoltage" in line][0]
        headers = full_lines[idx].split('\t')
        f_idxs = [(i,header) for i, header in enumerate(headers) if "LCR_Cp" in header]
        #f1_idx = headers.index("LCR_Cp_freq1")
        #f2_idx = headers.index("LCR_Cp_freq2")
        temp_idx = headers.index("Temperature")

        cv = {header : [] for i, header in f_idxs}
        bv = []
        temp = []

        data = full_lines[idx+1:]
        for line in data:
            words = line.split()
            if words[0][0] == "$":
                continue
            bv.append(abs(float(words[0])))
            temp.append(abs(float(words[temp_idx])))
            for item in f_idxs:
                cap = float(words[item[0]])
                cv[item[1]].append(cap)
            #f1.append(abs(float(words[f1_idx])))
            #f2.append(abs(float(words[f2_idx])))

        return bv, cv, temp 

def extract_cvs(filepaths):
    cv_data = {}
    for filepath in filepaths:
        bv, cv, temp = extract_cv(filepath)
        cv_data[os.path.basename(filepath)] = {'bv' : bv, 'cv' : cv, 'temp' : temp}

    return cv_data

def main(filepaths, outputname, cvfreqs=None, same=False):
    biasring_iv_data = {}
    guardring_iv_data = {}
    cv_data = {}
    if all(["iv" in os.path.basename(filepath).lower() for filepath in filepaths]):
        biasring_iv_data, guardring_iv_data = extract_ivs(filepaths)
    elif all(["cv" in os.path.basename(filepath).lower() for filepath in filepaths]):
        cv_data = extract_cvs(filepaths)
    else:
        print ">>> Measurement type not recognized, or measurement types have been mixed. Please use files named with the scheme {CV,IV}_*.txt"
        print filepaths
        sys.exit()
        
    
    if same:
        fig = plt.figure(1)
        fig.suptitle(outputname[:-4])
        ax = fig.add_subplot(111)
        # here we're assuming that the "else" above caught anything that's not {CV,IV}
        for file_i, filepath in enumerate(guardring_iv_data.keys()):
            filename = os.path.basename(filepath)
            ax.plot(guardring_iv_data[filepath]['bv'], 
                    guardring_iv_data[filepath]['gr_iv'], 
                    colors[file_i]+'-o', label = filename)
            ax.set_ylabel('Current (A)')
            ax.set_xlabel("Bias Voltage (V)")
            ax.legend(loc='upper left', prop={'size':6})
            if yaxis_range:
                ax.set_ylim(yaxis_range)
        for file_i, filepath in enumerate(cv_data.keys()):
            filename = os.path.basename(filepath)
            # transform the capacitances to 1/C^2 - should make this an option
            for i, key in enumerate(cv_data[filepath]['cv'].keys()):
                if cvfreqs:
                    if float(key[11:]) in cvfreqs:
                        cv_data[filepath]['cv'][key] = [1/(c**2) if c != 0 else cv_data[filepath]['cv'][key][idx-1] for idx, c in enumerate(cv_data[filepath]['cv'][key]) ]
                        ax.plot(cv_data[filepath]['bv'], cv_data[filepath]['cv'][key], colors[file_i]+'o', label = filename)
                else:
                    cv_data[filepath]['cv'][key] = [1/(c**2) for c in cv_data[filepath]['cv'][key]]
                    ax.plot(cv_data[filepath]['bv'], cv_data[filepath]['cv'][key], colors[file_i]+'o', label = filename)
            ax.set_xlabel("Bias Voltage (V)")
            ax.set_ylabel('1/C^2 ($F^{-2}$)')
            ax.legend(loc='upper left', prop={'size':6})
        if not outputname:
            print ">>> No output name given with '--same' option. Please give an output name."
            sys.exit()
        plt.savefig(outputname)
        plt.show()
    else:
        for file_i, filepath in enumerate(biasring_iv_data.keys()):
            fig = plt.figure(1)
            fig.suptitle(filename[:-4])
            ax = fig.add_subplot(111)
            filename = os.path.basename(filepath)
            ax.plot(bv, gr_iv, colors[i], label = 'Bias ring leakage (A)')
            ax.set_ylabel('Current (A)')
            ax.set_xlabel("Bias Voltage (V)")
            ax.legend(loc='upper left', prop={'size':6})
            if yaxis_range:
                ax.set_ylim(yaxis_range)
            if outputname:
                plt.savefig(outputname[:-4]+'_'+file_i+outputname[-4:])
            else:
                plt.savefig(filename[:-4]+'.png')
            plt.show()
        for file_i, filepath in enumerate(cv_data.keys()):
            filename = os.path.basename(filepath)
            fig = plt.figure(1)
            fig.suptitle(filename[:-4])
            ax = fig.add_subplot(111)
            filename = os.path.basename(filepath)
            # transform the capacitances to 1/C^2 - should make this an option
            for i, key in enumerate(cv_data[filepath]['cv'].keys()):
                if cvfreqs:
                    if float(key[11:]) in cvfreqs:
                        cv_data[filepath]['cv'][key] = [1/(c**2) for c in cv_data[filepath]['cv'][key]]
                        ax.plot(cv_data[filepath]['bv'], cv_data[filepath]['cv'][key], colors[file_i]+'o', label = filename)
                else:
                    cv_data[filepath]['cv'][key] = [1/(c**2) for c in cv_data[filepath]['cv'][key]]
                    ax.plot(cv_data[filepath]['bv'], cv_data[filepath]['cv'][key], colors[file_i]+'o', label = filename)
            ax.set_xlabel("Bias Voltage (V)")
            ax.set_ylabel('1/C^2 ($F^{-2}$)')
            ax.legend(loc='upper left', prop={'size':6})
            if outputname:
                plt.savefig(outputname[:-4]+'_'+str(file_i)+outputname[-4:])
            else:
                plt.savefig(filename[:-4]+'.png')
            plt.show()

main(args_dict['filenames'], args_dict['outputname'], cvfreqs, args_dict['same'])


Eg0 = 1.166 # eV # 1.206 eV in http://indico.cern.ch/event/129737/session/3/contribution/24/ma
alpha = 4.74 * 1e-4 # eV/K
beta = 636.0 # K
k_B = 8.61734E-5 # eV/K
def ScaleCurrent(ScaleTemp, Temp):
    Eg = 1.21#Eg0 - alpha * Temp**2 / (Temp + beta)
    return pow((ScaleTemp+273.15)/(Temp+273.15),2)*np.exp((-Eg/(2.*k_B))*(1./(ScaleTemp+273.15) - 1./(Temp+273.15)))

def ScaleCurrent2(ScaleTemp, Temp):
    Eg = 1.21#Eg0 - alpha * Temp**2 / (Temp + beta)
    return pow((ScaleTemp+273.15)/(Temp+273.15),2)*np.exp((-Eg/(2.*k_B))*(1./(ScaleTemp+273.15) - 1./(Temp+273.15)))
