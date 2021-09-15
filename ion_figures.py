########################################################################
# Author(s):    D. Knowles
# Date:         19 Aug 2021
# Desc:         creates ION figures and tables for presentation/paper
########################################################################

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import prep_logs

def main():
    """Main script that creates all tables and graphs.

    """

    # read current file location
    file_dir = os.path.dirname(os.path.realpath(__file__))

    # create figures log directory
    prep_logs(["figures"])

    # parameter sensitivity plot
    param_sensitivity_log = os.path.join(file_dir,"log",
                                         "chemnitz_fde_detailed_-1_50")
    plot_param_sensitivity(param_sensitivity_log)

    # compute chemnitz table of results
    chemnitz_traces = [
                       "chemnitz_fde_-1_10",
                       "chemnitz_fde_-1_20",
                       "chemnitz_fde_-1_50",
                       "chemnitz_fde_-1_100",
                       "chemnitz_fde_-1_200",
                      ]
    chemnitz_table(file_dir, chemnitz_traces)

    # compute google table of results
    google_table(file_dir, "google_fde_-1_100")

    # create fault hypothesis timing graph
    chemnitz_timing = [
                       "chemnitz_timing_1_-1_50",
                       "chemnitz_timing_2_-1_50",
                       "chemnitz_timing_3_-1_50",
                      ]
    plot_chemnitz_timing(file_dir,chemnitz_timing)

    # create measurement count timing graph
    google_timing = "google_timing_-1_100"
    plot_google_timing(file_dir,google_timing)

    plt.show()

def plot_accuracy(fde_data, method, label, color, style):
    """Plot balanced accuracy.

    Parameters
    ----------
    fde_data : dict
        dictionary that contains values to plot
    method : string
        FDE method to plot
    label : string
        label to attach to the graph
    color : string
        hex for plot line color
    style : string
        linestyle for graph

    """
    xmarks = []
    true_negative_rate = []
    false_positive_rate = []
    false_negative_rate = []
    true_positive_rate = []
    population = []
    pop_negative = []
    pop_positive = []
    accuracy = []
    balanced_accuracy = []
    for param in fde_data[method].keys():
        xmarks.append(param)
        tp = sum(fde_data[method][param]["tp"])
        tn = sum(fde_data[method][param]["tn"])
        fn = sum(fde_data[method][param]["fn"])
        fp = sum(fde_data[method][param]["fp"])

        true_negative_rate.append(tn / (tn + fp))
        false_positive_rate.append(fp / (fp + tn))
        if (tp + fn) == 0:
            false_negative_rate.append(0.)
            true_positive_rate.append(1.)
        else:
            false_negative_rate.append(fn / (fn + tp))
            true_positive_rate.append(tp / (tp + fn))

        balanced_accuracy.append(100*(true_positive_rate[-1]
                            + true_negative_rate[-1])/2.)

    plt.plot(xmarks, balanced_accuracy, label=label,
             color=color, linewidth = 3.0, linestyle=style)



def plot_param_sensitivity(log_dir):
    """Plot parameter sensitivity on top of each other.

    Parameters
    ----------
    log_dir : string
        log directory of the data to plot

    """

    # create FDE data
    fde_data = {}

    for file in sorted(os.listdir(log_dir)):
        if file[-4:] == ".csv":
            file_list = file.split("-")
            method = file_list[-2]
            if method not in fde_data:
                fde_data[method] = {}
            df = pd.read_csv(os.path.join(log_dir,file))
            for index, row in df.iterrows():
                if row["parameter"] not in fde_data[method]:
                    fde_data[method][row["parameter"]] = {}
                for col in df.columns:
                    if col == "parameter":
                        continue
                    elif col not in fde_data[method][row["parameter"]]:
                        fde_data[method][row["parameter"]][col] = [row[col]]
                    else:
                        fde_data[method][row["parameter"]][col].append(row[col])

    methods = ["edm","residual","solution"]
    labels = ["EDM","Residual","Solution"]
    colors = ["#b1040e","#006cb8","#008566"]
    style = ["-","--",":"]
    fig = plt.figure()
    for cc, method in enumerate(methods):
        plot_accuracy(fde_data, method, labels[cc], colors[cc], style[cc])

    plt.xscale("log")
    plt.ylim((40.,105.))
    plt.legend()
    plt.xlabel("Respective Threshold Parameter Value")
    plt.ylabel("Balanced Accuracy [%]")
    fig.set_size_inches(4,4)
    fig.tight_layout()
    fig.savefig(os.path.join("log","figures","param_sensitivity.png"),
            format="png",
            bbox_inches="tight",
            dpi=300)


def plot_metric(xdata, data, ylabel):
    """Plot metrics in a pleasant way.

    Parameters
    ----------
    xdata : list
        list of corresponding x vales to plot
    data : dict
        dictionary that contains values to plot
    ylabel : string
        ylabel for the graph
    """

    methods = ["edm","residual","solution"]
    labels = ["EDM","Residual","Solution"]
    colors = ["#b1040e","#006cb8","#008566"]
    style = ["-","--",":"]
    fig = plt.figure()
    for cc, method in enumerate(methods):
        plt.plot(xdata, data[method], label=labels[cc],
                 color=colors[cc], linewidth = 3.0, linestyle=style[cc])

    plt.legend()
    plt.xlabel("Bias Added to Measurements [m]")
    plt.ylabel(ylabel)
    fig.set_size_inches(4,4)
    fig.tight_layout()
    fig.savefig(os.path.join("log","figures",ylabel + ".png"),
            format="png",
            bbox_inches="tight",
            dpi=300)

def chemnitz_table(file_dir, chemnitz_traces):
    """Organize Chemnitz Dataset Results

    Parameters
    ----------
    file_dir : string
        directory of this file
    chemnitz_traces : list
        list of the log directories to plot

    """
    column_order = ["base","truth","residual","solution","edm"]
    errors_added = []
    balanced_accuracy = {}
    fnr = {}
    fpr = {}
    best_param = {}

    for trace in chemnitz_traces:
        # add error to list for table
        errors_added.append(trace.split("_")[-1])

        log_dir = os.path.join(file_dir,"log",trace)
        for file in sorted(os.listdir(log_dir)):
            if file[-4:] == ".csv":
                file_list = file.split("-")
                method = file_list[-2]

                # read in as df
                df = pd.read_csv(os.path.join(log_dir,file))

                df["tnr"] = df["tn"] / (df["tn"] + df["fp"])
                df["tpr"] = df["tp"] / (df["tp"] + df["fn"])
                df["tpr"] = df["tpr"].fillna(1.)
                df["fnr"] = 100.*df["fn"] / (df["fn"] + df["tp"])
                df["fnr"] = df["fnr"].fillna(0.)
                df["fpr"] = 100.*df["fp"] / (df["fp"] + df["tn"])
                df["balanced_accuracy"] = 100*(df["tpr"] + df["tnr"])/2.
                best_idx = int(df[["balanced_accuracy"]].idxmax())

                if method not in balanced_accuracy:
                    balanced_accuracy[method] = [df.iloc[best_idx]["balanced_accuracy"]]
                    fnr[method] = [df.iloc[best_idx]["fnr"]]
                    fpr[method] = [df.iloc[best_idx]["fpr"]]
                    best_param[method] = [df.iloc[best_idx]["parameter"]]
                else:
                    balanced_accuracy[method].append(df.iloc[best_idx]["balanced_accuracy"])
                    fnr[method].append(df.iloc[best_idx]["fnr"])
                    fpr[method].append(df.iloc[best_idx]["fpr"])
                    best_param[method].append(df.iloc[best_idx]["parameter"])

    print("% balanced accuracy")
    accuracy_df = pd.DataFrame.from_dict(balanced_accuracy).round(2)
    accuracy_df["errors_added"] = errors_added
    accuracy_df.set_index(keys="errors_added", inplace=True)
    accuracy_df = accuracy_df.reindex(columns=column_order)
    plot_metric(errors_added,balanced_accuracy,"Balanced Accuracy [%]")
    print(accuracy_df,"\n\n")

    print("% missed detection")
    fnr_df = pd.DataFrame.from_dict(fnr).round(2)
    fnr_df["errors_added"] = errors_added
    fnr_df.set_index(keys="errors_added", inplace=True)
    fnr_df = fnr_df.reindex(columns=column_order)
    plot_metric(errors_added,fnr,"Missed Detection Rate [%]")
    print(fnr_df,"\n\n")

    print("% false alarm")
    fpr_df = pd.DataFrame.from_dict(fpr).round(2)
    fpr_df["errors_added"] = errors_added
    fpr_df.set_index(keys="errors_added", inplace=True)
    fpr_df = fpr_df.reindex(columns=column_order)
    plot_metric(errors_added,fpr,"False Alarm Rate [%]")
    print(fpr_df,"\n\n")

    print("best parameter")
    param_df = pd.DataFrame.from_dict(best_param)
    param_df["errors_added"] = errors_added
    param_df.set_index(keys="errors_added", inplace=True)
    param_df = param_df.reindex(columns=column_order)
    print(param_df,"\n\n")

def google_table(file_dir, google_dir):
    """Organize Google dataset results.

    Parameters
    ----------
    file_dir : string
        directory of this file
    google_dir : string
        name of the log directory where data is located

    """
    column_order = ["base","truth","residual","solution","edm"]
    balanced_accuracy = {}
    fnr = {}
    fpr = {}
    best_param = {}

    df_data = {}

    log_dir = os.path.join(file_dir,"log",google_dir)
    for file in sorted(os.listdir(log_dir)):
        if file[-4:] == ".csv":
            file_list = file.split("-")
            method = file_list[-2]

            # read in as df
            df = pd.read_csv(os.path.join(log_dir,file))

            df["tnr"] = df["tn"] / (df["tn"] + df["fp"])
            df["tpr"] = df["tp"] / (df["tp"] + df["fn"])
            df["tpr"] = df["tpr"].fillna(1.)
            df["fnr"] = 100.*df["fn"] / (df["fn"] + df["tp"])
            df["fnr"] = df["fnr"].fillna(0.)
            df["fpr"] = 100.*df["fp"] / (df["fp"] + df["tn"])
            df["balanced_accuracy"] = 100*(df["tpr"] + df["tnr"])/2.

            if method not in df_data:
                df_data[method] = {}
            for index, row in df.iterrows():
                if row["parameter"] not in df_data[method]:
                    df_data[method][row["parameter"]] = row.to_frame().transpose()
                else:
                    df_data[method][row["parameter"]] = pd.concat((df_data[method][row["parameter"]],
                                                                   row.to_frame().transpose()))

    for key, parameter_dict in df_data.items():
        avg_ba = []
        method_params = []
        for param, df_param in parameter_dict.items():
            method_params.append(param)
            avg_ba.append(df_param["balanced_accuracy"].mean())
        best_idx = np.argmax(np.array(avg_ba))

        balanced_accuracy[key] = parameter_dict[method_params[best_idx]]["balanced_accuracy"].tolist()
        fnr[key] = parameter_dict[method_params[best_idx]]["fnr"].tolist()
        fpr[key] = parameter_dict[method_params[best_idx]]["fpr"].tolist()
        best_param[key] = parameter_dict[method_params[best_idx]]["parameter"].tolist()


    print("\n\n\n\nGoogle")
    print("% balanced accuracy")
    accuracy_df = pd.DataFrame.from_dict(balanced_accuracy).round(2)
    accuracy_df = accuracy_df.reindex(columns=column_order)
    print(accuracy_df.mean(),"\n\n")

    print("% missed detection")
    fnr_df = pd.DataFrame.from_dict(fnr).round(2)
    fnr_df = fnr_df.reindex(columns=column_order)
    print(fnr_df.mean(),"\n\n")

    print("% false alarm")
    fpr_df = pd.DataFrame.from_dict(fpr).round(2)
    fpr_df = fpr_df.reindex(columns=column_order)
    print(fpr_df.mean(),"\n\n")

    print("best parameter")
    param_df = pd.DataFrame.from_dict(best_param)
    param_df = param_df.reindex(columns=column_order)
    print(param_df.mean(),"\n\n")

def plot_chemnitz_timing(file_dir, trace_list):
    """Calculate fault hypothesis timing metrics

    Parameters
    ----------
    file_dir : string
        directory of this file
    trace_list : list
        list of the log directories to plot

    """


    timing_data = {}

    for trace in trace_list:
        num_errors = int(trace.split("_")[-3])

        log_dir = os.path.join(file_dir,"log",trace)
        for file in sorted(os.listdir(log_dir)):
            if file[-4:] == ".csv":
                file_list = file.split("-")
                method = file_list[-2]

                if method not in timing_data:
                    timing_data[method] = {}

                csv_filename = os.path.join(log_dir,file)
                with open(csv_filename) as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter=',')
                    line_count = 0
                    for row in csv_reader:
                        if line_count == 0:
                            measurement_counts = row
                            line_count += 1
                        else:
                            row = [float(v) for v in row]
                            ii = line_count - 1

                            # measurement_count = int(measurement_counts[ii])
                            if num_errors not in timing_data[method]:
                                timing_data[method][num_errors] = row
                            else:
                                timing_data[method][num_errors].extend(row)
                            line_count += 1

    methods = ["edm","residual","solution"]
    labels = ["EDM","Residual","Solution"]
    colors = ["#b1040e","#006cb8","#008566"]
    styles = ["-","--",":"]

    fig = plt.figure()
    fig.set_size_inches(4,4)
    for ii, method in enumerate(methods):
        # print("method: ",method)
        # if method not in ["residual","edm"]:
        #     continue
        fault_counts = []
        avg_times = []
        for key, value in timing_data[method].items():
            # print(key)
            fault_counts.append(key)
            avg_times.append(np.mean(np.array(value)*1000))
            # print(np.mean(np.array(value)))
            if method == "residual" and key == 1:
                plt.scatter(key,np.mean(np.array(value)*1000),
                color=colors[ii], label="Residual", s=75, zorder=2,
                marker="X")
        if method != "residual":
            plt.plot(fault_counts,avg_times, linewidth=3.0, zorder=1,
                     color=colors[ii],linestyle=styles[ii], label=labels[ii])
    plt.legend()
    plt.yscale("log")
    plt.ylabel("Average Computation Time [ms]")
    plt.xlabel("Number of Faults")
    fig.tight_layout()
    fig.savefig(os.path.join("log","figures","chemnitz_timing.png"),
            format="png",
            bbox_inches="tight",
            dpi=300)

def plot_google_timing(file_dir, directory):
    """Calculate measurement count timing metrics

    Parameters
    ----------
    file_dir : string
        directory of this file
    directory : list
        name of the log directory where data is located

    """

    timing_data = {}

    log_dir = os.path.join(file_dir,"log",directory)
    for file in sorted(os.listdir(log_dir)):
        if file[-4:] == ".csv":
            file_list = file.split("-")
            method = file_list[-2]

            if method not in timing_data:
                timing_data[method] = {}

            csv_filename = os.path.join(log_dir,file)
            with open(csv_filename) as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        measurement_counts = row
                        line_count += 1
                    else:
                        row = [float(v) for v in row]
                        ii = line_count - 1

                        measurement_count = int(measurement_counts[ii])
                        if measurement_count not in timing_data[method]:
                            timing_data[method][measurement_count] = row
                        else:
                            timing_data[method][measurement_count].extend(row)
                        line_count += 1

    fig = plt.figure()
    methods = ["edm","residual","solution"]
    labels = ["EDM","Residual","Solution"]
    colors = ["#b1040e","#006cb8","#008566"]
    styles = ["-","--",":"]
    for method, timing_dict in timing_data.items():
        keys = sorted(list(timing_dict.keys()))
        avg_times = []
        key_list = []
        for key in keys:
            if key > 40:
                continue
            plot_values = np.array(timing_dict[key])
            key_list.append(key)
            avg_times.append(1000*np.mean(np.array(plot_values)))
        if method in ["residual","edm","solution"]:
            if method == "edm":
                color = colors[0]
                style = styles[0]
                label = labels[0]
            elif method == "residual":
                color = colors[1]
                style = styles[1]
                label = labels[1]
            elif method == "solution":
                color = colors[2]
                style = styles[2]
                label = labels[2]
            plt.plot(key_list,avg_times,label=label, color=color,
                    linewidth=3.0, linestyle = style)
    plt.legend()
    plt.yscale("log")
    plt.ylabel("Average Computation Time [ms]")
    plt.xlabel("Number of Measurements")
    fig.set_size_inches(4,4)
    fig.tight_layout()
    fig.savefig(os.path.join("log","figures","google_timing.png"),
            format="png",
            bbox_inches="tight",
            dpi=300)

if __name__ == "__main__":
    main()
