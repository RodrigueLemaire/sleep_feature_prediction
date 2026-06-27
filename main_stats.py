from tabulate import tabulate
from datetime import datetime
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from datahelper import *

TESTING = 7
if __name__ == "__main__":
    logging.debug("start_main\n")

    data_sleep_all = data_sleep_all()
    data_partinfo_all = data_partinfo_all()
    data_selfreport_all = data_selfreport_all()

    #logging.info("\n" + tabulate(data_sleep_all, headers='keys', tablefmt='psql'))

    # View data for participant
    if TESTING == 1:
        id = 2  # id is between 1 and 29

        dataframe1 = read_id(id, data_partinfo_all)
        dataframe2 = read_id(id, data_sleep_all)
        dataframe3 = read_id(id, data_selfreport_all)
        logging.info("\n" + tabulate(dataframe1, headers='keys', tablefmt='psql'))
        logging.info("\n" + tabulate(dataframe2, headers='keys', tablefmt='psql'))
        logging.info("\n" + tabulate(dataframe3, headers='keys', tablefmt='psql'))

    # Calculate time delta between start and end of dataset for each participant
    if TESTING == 2:

        list_delta = []

        for id_loop in range(1, 30):
            df_loop = read_id(id_loop, data_sleep_all)

            start = datetime.fromisoformat(df_loop['day'].iloc[0])
            end = datetime.fromisoformat(df_loop['day'].iloc[-1])

            days = (end.date() - start.date()).days

            list_delta.append(days)

        dataframe_delta = pd.DataFrame({"id": range(1, 30), "delta days": list_delta})
        dataframe_delta["delta days"] = dataframe_delta["delta days"].astype(int)

        logging.info("Table of time difference in days between first night and "
                     "last night recorded in the dataset per participant")
        logging.info("\n" + tabulate(dataframe_delta, headers='keys', tablefmt='psql'))

        total = dataframe_delta["delta days"].sum()
        logging.info("Total days in dataset : " + str(total))

    # Calculate valid completed days in the dataset for each participant
    if TESTING == 3:

        list_days = []

        for id_loop in range(1, 30):
            df_loop = read_id(id_loop, data_sleep_all)

            days = pd.to_datetime(df_loop["day"], utc=True).dt.date.nunique()

            list_days.append(days)

        dataframe_days = pd.DataFrame({"id": range(1, 30), "total days": list_days})
        dataframe_days = dataframe_days.sort_values('total days', ascending=False)
        dataframe_days["total days"] = dataframe_days["total days"].astype(int)

        logging.info("Table of unique days recorded excluding missing days, per participant")
        logging.info("\n" + tabulate(dataframe_days, headers='keys', tablefmt='psql'))

        total = dataframe_days["total days"].sum()
        logging.info("Total days in dataset : " + str(total))

    # Combine dataframes for a participant and run Spearman correlations
    if TESTING == 4:
        id = 6  # id is between 1 and 29

        dataframe1 = read_id(id, data_partinfo_all)
        dataframe2 = read_id(id, data_sleep_all).copy()
        dataframe3 = read_id(id, data_selfreport_all).copy()

        #dataframe2["day"] = pd.to_datetime(dataframe2["day"], utc=True).dt.date.nunique()
        #dataframe3["date"] = pd.to_datetime(dataframe3["date"], utc=True).dt.date.nunique()

        dataframe2.rename(columns={'day': 'date'}, inplace=True)

        out = dataframe2.merge(dataframe3, on="date", how="left")

        logging.info("\n" + tabulate(out, headers='keys', tablefmt='psql'))
        #logging.info("\n" + tabulate(dataframe2, headers='keys', tablefmt='psql'))

        # remove date, time, and other values which interfere with the correlation
        out = out.drop(columns=['date', 'time', 'id_x', 'id_y', 'selfassessment_available', 'nback_available', 'nback_valid'])

        # run spearman correlation on the entire table (pandas)
        out_spear = out.corr(method="spearman")

        logging.info("\n" + tabulate(out_spear, headers='keys', tablefmt='psql'))

        # run spearman and p-value calculations on the entire table (scipy)
        out = out.apply(pd.to_numeric, errors='coerce')
        out_spear_r = spearmanr(out, axis=0, nan_policy='omit')
        out_spear_r = pd.DataFrame(out_spear_r.statistic)
        out_spear_r.index = out_spear_r.columns = out.columns

        logging.info("\n" + tabulate(out_spear_r, headers='keys', tablefmt='psql'))

        assert(
                tabulate(out_spear, headers='keys', tablefmt='psql')
                ==
                tabulate(out_spear_r, headers='keys', tablefmt='psql')
        )

    # Refer to testing 4; for all participants now
    if TESTING == 5:

        out_spear_all_corr: [pd.DataFrame] = []

        for id_loop in range(1, 30):
            dataframe1 = read_id(id_loop, data_sleep_all).copy()
            dataframe2 = read_id(id_loop, data_selfreport_all).copy()

            dataframe1.rename(columns={'day': 'date'}, inplace=True)

            out = dataframe1.merge(dataframe2, on="date", how="left")

            #logging.info("\n" + tabulate(out, headers='keys', tablefmt='psql'))

            # remove date, time, and other values which interfere with the correlation
            out = out.drop(columns=['date', 'time', 'id_x', 'id_y', 'selfassessment_available', 'nback_available', 'nback_valid'])

            # run spearman correlation on the entire table
            out = out.apply(pd.to_numeric, errors='coerce')
            out_spear_r = spearmanr(out, axis=0, nan_policy='omit')

            out_spear_r_corr = pd.DataFrame(out_spear_r.statistic)
            out_spear_r_corr.index = out_spear_r_corr.columns = out.columns
            out_spear_all_corr.append(out_spear_r_corr)

        column_names = list(out_spear_all_corr[0].columns)
        column_names_comb = list(combinations(column_names, 2))
        column_data = {value: [] for value in column_names_comb}

        logging.info("\nColumns: " + str(column_names))
        logging.info("\nPossible pairs: " + str(column_names_comb))

        # For each possible correlation pairing, get the result for each participant
        for name_pair in column_names_comb:
            for id_loop in range(0, 29):
                column_data[name_pair].append(
                    out_spear_all_corr[id_loop].loc[name_pair[0], name_pair[1]],
                )

        df_column_data = pd.DataFrame.from_dict(column_data)

        logging.info("\nAll correlations: \n" + tabulate(df_column_data, headers='keys', tablefmt='psql'))

        # Only consider strong correlations for the sake of readability
        threshold = 0.5
        df_column_data = df_column_data.loc[:, (df_column_data.max() > threshold) | (df_column_data.min() < -threshold)]

        logging.info("\nStrong correlations only: \n" + tabulate(df_column_data, headers='keys', tablefmt='psql'))

        # Attempt to clear out NaN values (thank you second participant)
        df_clean = [df_column_data[c].dropna().to_numpy() for c in df_column_data.columns if df_column_data[c].notna().any()]

        plt.figure(figsize=(12, 8), dpi=100)
        plt.boxplot(df_clean, tick_labels=list(df_column_data.columns))
        plt.title("Strong correlations")
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.7)
        plt.tight_layout()
        plt.grid()
        plt.show()

    # Find missing data in initial dataset
    if TESTING == 6:

        list_partially_empty = []
        list_fully_empty = []

        for id_loop in range(1, 30):
            dataframe1 = read_id(id_loop, data_sleep_all).copy()
            dataframe2 = read_id(id_loop, data_selfreport_all).copy()

            dataframe1.rename(columns={'day': 'date'}, inplace=True)

            out = dataframe1.merge(dataframe2, on="date", how="left")
            out = out.drop(
                columns=['date', 'time', 'id_x', 'id_y', 'selfassessment_available', 'nback_available', 'nback_valid'])

            list_partially_empty.append("ID: " + str(id_loop) + " -> " + str(out.columns[out.isnull().any()]))
            list_fully_empty.append("ID: " + str(id_loop) + " -> " + str(out.columns[out.isna().all()].tolist()))

        logging.info("Partially empty columns :")
        logging.info(str(list_partially_empty))
        logging.info("\nFully empty columns :")
        logging.info(str(list_fully_empty))

    # Testing 5; but with p-values
    if TESTING == 7:

        out_spear_all_corr: [pd.DataFrame] = []
        out_spear_all_pval: [pd.DataFrame] = []

        for id_loop in range(1, 30):
            dataframe1 = read_id(id_loop, data_sleep_all).copy()
            dataframe2 = read_id(id_loop, data_selfreport_all).copy()

            dataframe1.rename(columns={'day': 'date'}, inplace=True)

            out = dataframe1.merge(dataframe2, on="date", how="left")

            # remove date, time, and other values which interfere with the correlation
            out = out.drop(columns=['date', 'time', 'id_x', 'id_y', 'selfassessment_available', 'nback_available', 'nback_valid'])

            # run spearman correlation on the entire table
            out = out.apply(pd.to_numeric, errors='coerce')
            out_spear_r = spearmanr(out, axis=0, nan_policy='omit')

            out_spear_r_corr = pd.DataFrame(out_spear_r.statistic)
            out_spear_r_pval = pd.DataFrame(out_spear_r.pvalue)

            out_spear_r_corr.index = out_spear_r_corr.columns = out.columns
            out_spear_r_pval.index = out_spear_r_pval.columns = out.columns

            out_spear_all_corr.append(out_spear_r_corr)
            out_spear_all_pval.append(out_spear_r_pval)

        column_names = list(out_spear_all_corr[0].columns)
        column_names_comb = list(combinations(column_names, 2))
        column_corr = {value: [] for value in column_names_comb}
        column_pval = {value: [] for value in column_names_comb}

        logging.info("\nColumns: " + str(column_names))
        logging.info("\nPossible pairs: " + str(column_names_comb))

        # For each possible correlation pairing, get the result for each participant
        for name_pair in column_names_comb:
            for id_loop in range(0, 29):
                column_corr[name_pair].append(
                    out_spear_all_corr[id_loop].loc[name_pair[0], name_pair[1]],
                )
                column_pval[name_pair].append(
                    out_spear_all_pval[id_loop].loc[name_pair[0], name_pair[1]],
                )

        df_column_corr = pd.DataFrame.from_dict(column_corr)
        df_column_pval = pd.DataFrame.from_dict(column_pval)

        logging.info("\nAll correlations: \n" + tabulate(df_column_corr, headers='keys', tablefmt='psql'))

        # Only consider strong correlations for the sake of readability
        threshold = 0.5
        df_column_corr = df_column_corr.loc[:, (df_column_corr.max() > threshold) | (df_column_corr.min() < -threshold)]

        common_cols = df_column_corr.columns.intersection(df_column_pval.columns)

        df_column_pval = df_column_pval[common_cols]

        logging.info("\nStrong correlations only: \n" + tabulate(df_column_corr, headers='keys', tablefmt='psql'))

        # Attempt to clear out NaN values (thank you second participant)
        df_clean_corr = [df_column_corr[c].dropna().to_numpy() for c in df_column_corr.columns if
                    df_column_corr[c].notna().any()]
        df_clean_pval = [df_column_pval[c].dropna().to_numpy() for c in df_column_pval.columns if
                    df_column_pval[c].notna().any()]


        fig, axs = plt.subplots(2, 1, figsize=(12, 12))

        # Correlation graph
        axs[0].boxplot(df_clean_corr)
        axs[0].set_title("Strong correlations (and their p-values distribution)")
        axs[0].set_xticklabels(list(df_column_corr.columns), rotation=90)
        #axs[0].subplots_adjust(bottom=0.7)


        # P-values graph
        axs[1].boxplot(df_clean_pval)
        #axs[1].set_title("P-values")
        axs[1].set_xticklabels(list(df_column_pval.columns), rotation=90)
        #axs[1].subplots_adjust(bottom=0.7)
        plt.setp(axs[0].get_xticklabels(), visible=False)

        plt.tight_layout()
        axs[0].grid()
        axs[1].grid()
        plt.show()



