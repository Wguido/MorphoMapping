# Author: Amelie Bauerdick
# WabnitzLab
# 2024


"""
Create interactive umap and densmap visualizations.
Plot feature importance and apply several clustering algorithms after dimensionality reduction.
Class MM is based on pandas DataFrames.

Class:

    MM

Functions:

    convert_to_CSV(fcs_path: str, csv_path: str)
    read_CSV(path: str, add_index: bool, index_name: str) -> df
    read_files(self,path: str, add_index: bool =False, index_name: str ='Index')->df
    get_features -> list
    get_df -> df
    add_metadata(label: str, value) -> df
    rename_variables(label_mapping)
    select_condition(condition: str, value)
    select_events(event_size: int)
    drop_variables(*labels)
    drop_events(first_row: int, last_row: int)
    save_feature(*features) -> df
    concat_variables(*dataframes) -> df
    save_xlsx(path: str)
    save_csv(path: str)
    concat_df(self, *new_df, join='inner',add_index: bool =False, index_name: str ='Index')
    update_column_values(column_name: str, rename_values)
    weighted_features (self,features:list[str],weight: int)
    stand_scaler (self, first_column: Optional[str] = None, last_column: str =None)
    minmax_norm(first_column: str =None, last_column: str =None)
    quant_scaler(first_column: str =None, last_column: str =None)
    umap(nn: int, mdist: float, met: str)
    dmap(dlambda: float, nn: int, mdist: float, met: str)
    feature_importance(dep: str, indep: str) -> df
    plot_feature_importance(features, path: str, base_width: int, base_height: int)
    cluster_kmeans(n_cluster: int)
    cluster_gmm(number_component: int, random_s: int)
    cluster_hdbscan(cluster_size: int)
    check_dataframe()
    prepare_data_source()
    configure_hover_tooltips(feature: str, hover_tooltips: list[tuple[str, str]] = None)
    create_base_plot(fig_width: int, fig_height: int, fig_title: str, label_x: str, label_y: str, range_x: list[float], range_y: list[float], tools_emb, title_align: str)
    configure_axes_and_legend(plot, show_axes: bool, show_legend: bool)
    cat_plot(feature: str, subs: list[str], colors: list[str], outputf: str, fig_width: int, fig_height: int, fig_title: str, label_x: str,label_y: str, range_x: list[float], range_y: list[float],
             hover_tooltips: list[tuple[str, str]] = None, show_legend: bool = False, point_size: int, point_alpha: float, show_axes: bool = False, title_align: str = 'center') -> None:
    lin_plot(outputf: str, feature: str, colors: list[str], fig_width: int, fig_height: int, fig_title: str, label_x: str, label_y: str, range_x: list[float], range_y: list[float],
             hover_tooltips: list[tuple[str, str]] = None, show_legend: bool = False, point_size: int, point_alpha: float, show_axes: bool = True, title_align: str = 'center') -> None:
    cell_plot(self, cluster_column, top_clusters: int, palette, ID, x_label, y_label, png: bool, svg: bool,name: str):

variables:

    self.df

"""

# Import Packages

import flowkit as fk
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from sklearn.metrics import r2_score
import sklearn.cluster as cluster
from sklearn.mixture import GaussianMixture
import umap
import hdbscan
from bokeh.models import (HoverTool,
                          ColumnDataSource,
                          Range1d,
                          LinearColorMapper,
                          CategoricalColorMapper)
from bokeh.plotting import figure, show, output_file
import bokeh
from adjustText import adjust_text
from typing import Optional

bokeh_version = bokeh.__version__
bokeh_above_three = int(bokeh_version.split(".")[0]) >= 3


class MM:
    """\
    A class to create interactive dimensionality reduction plots.
    Based on pandas DataFrame.

    Parameters
    ----------
    df
        DataFrame containing all Imaging Flow Cytometry data

    Returns
    -------
    None

    """

    def __init__(self):
        self.df = pd.DataFrame()

    def convert_to_CSV(self, fcs_path: str, csv_path: str):
        """\
        Converts fcs-file to .csv file and saves it to csv_path.

        Parameters
        ----------
        fcs_path
           path to fcs file
        csv_path
           path to csv file

        Returns
        -------
        None

        """
        try:
            if not csv_path:
                raise ValueError("The CSV path is empty or invalid.")

            sample = fk.Sample(fcs_path)
            sample.export(filename=csv_path, source='raw')

            print(f"File successfully converted to {csv_path}.")

        except FileNotFoundError:
            print(f"FCS file not found: {fcs_path}")
            raise
        except Exception as e:
            print(f"Conversion failed: {e}")
            raise

    def read_CSV(self, path: str, add_index: bool = False, index_name: str = 'Index'):
        """\
        Load csv-file and save it as self.df

        Parameters
        ----------
        path
            path to csv file
        add_index
            add index to df
        index_name
            set name of index

        Returns
        ----------
        DataFrame

        """

        self.df = pd.read_csv(path)

        # rename columns
        self.df.columns = (
            self.df.columns.str.strip()
            .str.replace(' ', '_')
            .str.replace('&', 'and')
            .str.replace('+', 'plus')
            .str.replace('-', 'minus')
        )

        if add_index:
            self.df[index_name] = range(len(self.df))
            self.df.set_index(index_name, inplace=True)

        return self.df

    def read_files(self, path: str, add_index: bool = False, index_name: str = 'Index', join: str = 'outer'):
        """\
        read all csv-files in the given folder

        Parameters
        ----------
        path
            path to csv file
        add_index
            add index to df
        index_name
            set name of index
        join
            join df ('inner,'outer')

        Returns
        ----------
        DataFrame
        """

        dfs = []

        for file in os.listdir(path):
            if file.lower().endswith('.csv'):
                file_path = os.path.join(path, file)
                df = self.read_CSV(file_path, add_index=False)
                df['file'] = file
                dfs.append(df)

        if not dfs:
            print(f"No CSV files found in: {path}")
            self.df = pd.DataFrame()
            return self.df

        combined_df = pd.concat(dfs, ignore_index=True, join=join)

        if add_index:
            combined_df[index_name] = range(len(combined_df))
            combined_df.set_index(index_name, inplace=True)

        columns = [col for col in combined_df.columns if col != 'file'] + ['file']
        combined_df = combined_df[columns]

        self.df = combined_df
        return self.df

    def get_features(self):
        """\
        return list of self.df columns
        """
        return list(self.df.columns)

    def get_df(self):
        """\
        return self.df
        """
        return self.df

    def add_metadata(self, label: str, value):
        """\
        add column with specific value to self.df

        Parameters
        ----------
        label
            name of column
        value
            value of column

        Returns
        ----------
        DataFrame

        """
        empty = self.df.empty
        if empty:
            raise ValueError("Dataframe is empty.")

        self.df[label] = value
        return self.df

    def rename_variables(self, label_mapping):
        """\
        rename column(s) with new column labels

        Parameters
        ----------
        label_mapping
            old and new assigned column labels

        Returns
        ----------
        None

        """

        missing_columns = [col for col in label_mapping.keys() if col not in self.df.columns]

        if missing_columns:
            print(f"The following columns do not exist: {missing_columns}")
        else:
            self.df.rename(columns=label_mapping, inplace=True)

    def select_condition(self, condition: str, value):
        """\
        select specific rows by condition and save new df as self.df

        Parameters
        ----------
        condition
            name of column
        value
            value of column

        Returns
        ----------
        None
        """

        if condition not in self.df.columns:
            raise ValueError(f"Column '{condition}' does not exist.")
        if value not in self.df[condition].values:
            raise ValueError(f"Value '{value}' does not exist in column '{condition}'.")
        # select rows and save new df
        self.df = self.df.loc[self.df[condition] == value]

    def select_events(self, event_size: int):
        """\
        randomly select events and save as self.df

        Parameters
        ----------
        event_size
            number of events

        Returns
        ----------
        None
        """

        if event_size > self.df.shape[0]:
            raise ValueError(f"Number of events '{event_size}' is larger than the number of rows ({self.df.shape[0]}).")
        else:
            self.df = self.df.sample(event_size, random_state=1).copy()
            self.df.sort_index(inplace=True)

    def drop_variables(self, *labels):
        """\
        drop certain columns from self.df

        Parameters
        ----------
        labels
            name of column(s)

        Returns
        ----------
        None
        """
        missing_labels = [label for label in labels if label not in self.df.columns]
        if missing_labels:
            raise ValueError(f"Column(s) {missing_labels} do not exist.")
        else:
            self.df = self.df.drop(columns=list(labels))

    def drop_events(self, first_row: int, last_row: int):
        """\
        drop specific rows from self.df

        Parameters
        ----------
        first_row
            number of first row that should be dropped
        last_row
            number of last row that should be dropped

        Returns
        ----------
        None
        """
        if first_row not in self.df.index:
            raise ValueError(f"First row '{first_row}' does not exist.")
        if last_row not in self.df.index:
            raise ValueError(f"Last row '{last_row}' does not exist.")
        if first_row > last_row:
            raise ValueError(f"First row number '{first_row}' is greater than last row number '{last_row}'.")
        self.df = self.df.drop(index=self.df.index[first_row:last_row + 1])

    def save_feature(self, *features):
        """\
        save specific columns of self.df in new DataFrame and return new DataFrame

        Parameters
        ----------
        features
            name of features which should be returned as df

        Returns
        ----------
        DataFrame
        """
        labels = [feature for feature in features if feature not in self.df.columns]
        if labels:
            raise ValueError(f"Column(s) {labels} do not exist.")

        df = self.df[list(features)].copy()

        return df

    def concat_variables(self, *dataframes):
        """\
        attach new columns as df to self.df and return resulting df

        Parameters
        ----------
        dataframes
            new dataframes which should be attached to self.df

        Returns
        ----------
        DataFrame
        """
        result_df = pd.concat([self.df] + list(dataframes), axis=1)
        self.df = result_df
        return result_df

    def save_xlsx(self, path: str):
        """\
        save self.df as xlsx file to chosen path

        Parameters
        ----------
        path
            chosen path

        Returns
        ----------
        None
        """
        self.df.to_excel(path, index=False)
        print(f"DataFrame successfully saved to {path}")

    def save_csv(self, path: str):
        """\
        save self.df as csv to chosen path

        Parameters
        ----------
        path
            chosen path

        Returns
        ----------
        None
        """
        self.df.to_csv(path, index=False)
        print(f"DataFrame successfully saved to {path}")

    def concat_df(self, *new_df, join='inner', add_index: bool = False, index_name: str = 'Index'):
        """\
        concatenate self.df and new dfs (joining inner by default)

        Parameters
        ----------
        new_df
            df that should be concatenated
        join
            type of joining
        add_index
            add index to df
        index_name
            set name of index

        Returns
        ----------
        None
        """
        self.df = pd.concat([self.df, *new_df], join=join,ignore_index=True)

        if add_index:
            self.df[index_name] = range(len(self.df))
            self.df.set_index(index_name, inplace=True)

    def update_column_values(self, column_name: str, rename_values):
        """\
        replace values in a specific column with a specific new value

        Parameters
        ----------
        column_name
            name of column
        rename_values
            new value of column

        Returns
        ----------
        None
        """

        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' does  not exist.")

        self.df[column_name] = self.df[column_name].astype(str)

        for original_value, new_value in rename_values.items():
            self.df[column_name] = self.df[column_name].str.replace(str(original_value), new_value)

    def weighted_features(self, features: list[str], weight: int):
        """\

        Parameters
        ----------
        features
            list of features
        weight
            factor for weighted features

        Returns
        ---------
        df

        """
        for feature in features:
            self.df[feature] = self.df[feature] * weight

        return self.df

    def stand_scaler(self, first_column: Optional[str] = None, last_column: Optional[str] = None):
        """
        Apply Standard Scaler to specified columns of self.df.

        Parameters
        ----------
        first_column : str or None
            Name of the first column to scale.
        last_column : str or None
            Name of the last column to scale.

        Returns
        -------
        None
        """
        if self.df is None or self.df.empty:
            raise ValueError("DataFrame is empty or not set.")

        if first_column is None or last_column is None:
            df1 = self.df.select_dtypes(include="number")
            columns = df1.columns
        else:
            if first_column not in self.df.columns:
                raise ValueError(f"First column '{first_column}' does not exist.")
            if last_column not in self.df.columns:
                raise ValueError(f"Last column '{last_column}' does not exist.")

            first_id = self.df.columns.get_loc(first_column)
            last_id = self.df.columns.get_loc(last_column)

            if first_id > last_id:
                raise ValueError(f"First column '{first_column}' is after last column '{last_column}'.")

            df1 = self.df.iloc[:, first_id:last_id + 1]
            columns = df1.columns

        if df1.empty:
            raise ValueError("No numeric data found to scale.")

        scaler = StandardScaler()
        df2 = pd.DataFrame(scaler.fit_transform(df1), columns=columns, index=df1.index)

        self.df.loc[:, columns] = df2.astype(float)

    def minmax_norm(self, first_column: Optional[str] = None, last_column: str = None):
        """\
        Apply MinMax normalization to self.df.
        Specify whether all columns should be normalized by setting parameters.

        Parameters
        ----------
        first_column
            name of first column. Start of normalization.
        last_column
            name of last column. End of normalization.

        Returns
        ----------
        None

        """

        # check df
        if first_column is None or last_column is None:
            df1 = self.df
        else:
            if first_column not in self.df.columns:
                raise ValueError(f"First column '{first_column}' does not exist.")
            if last_column not in self.df.columns:
                raise ValueError(f"Last column '{last_column}' does not exist.")

            first_id = self.df.columns.get_loc(first_column)
            last_id = self.df.columns.get_loc(last_column)
            if first_id > last_id:
                raise ValueError(f"First column '{first_column}' is after last column '{last_column}'.")

            df1 = self.df.iloc[:, first_id:last_id + 1]

        # Apply min-max normalization
        df = (df1 - df1.min()) / (df1.max() - df1.min())

        # Replace the subset
        if first_column is not None and last_column is not None:
            self.df.iloc[:, first_id:last_id + 1] = df
        else:
            self.df = df

    def quant_scaler(self, first_column: Optional[str] = None, last_column: str = None):
        """\
        Apply  QuantileTransformer to self.df.
        Specify whether all columns should be normalized by setting parameters.

        Parameters
        ----------
        first_column
            name of first column. Start of normalization.
        last_column
            name of last column. End of normalization.

        Returns
        ----------
        None
        """

        if first_column is None or last_column is None:
            df1 = self.df
        else:
            if first_column not in self.df.columns:
                raise ValueError(f"First column '{first_column}' does not exist.")
            if last_column not in self.df.columns:
                raise ValueError(f"Last column '{last_column}' does not exist.")

            first_id = self.df.columns.get_loc(first_column)
            last_id = self.df.columns.get_loc(last_column)
            if first_id > last_id:
                raise ValueError(f"First column '{first_column}' is after last column '{last_column}'.")

            df1 = self.df.iloc[:, first_id:last_id + 1]

        # Apply Quantile Transformation
        scaler = QuantileTransformer(output_distribution='uniform')
        df2 = pd.DataFrame(scaler.fit_transform(df1),
                           columns=df1.columns,
                           index=df1.index)

        if first_column is not None and last_column is not None:
            self.df.iloc[:, first_id:last_id + 1] = df2
        else:
            self.df = df2

    def umap(self, nn: int, mdist: float, met: str):
        """\
        Run umap with self.df. Adds x and y values to self.df as extra columns.

        Parameters
        ----------
        nn
            nearest neighbours for umap settings
        mdist
            minimum distance for umap settings
        met
            metric for umap settings

        Returns
        ----------
        None

        """
        reducer = umap.UMAP(
            n_neighbors=nn,
            min_dist=mdist,
            metric=met
        )

        embedding = reducer.fit_transform(self.df)

        x = embedding[:, 0]
        y = embedding[:, 1]

        self.df['x'] = x
        self.df['y'] = y

    def dmap(self, dlambda: float, nn: int, mdist: float, met: str):
        """\
        Run dmap with self.df. Adds x and y values to self.df as extra columns.

        Parameters
        ----------
        dlambda
            denslambda for densmap settings
        nn
            nearest neighbours for densmap settings
        mdist
            minimum distance for densmap settings
        met
            metric for densmap settings

        Returns
        ----------
        None
        """

        reducer = umap.UMAP(
            densmap=True,
            dens_lambda=dlambda,
            n_neighbors=nn,
            min_dist=mdist,
            metric=met
        )

        embedding = reducer.fit_transform(self.df)
        x = embedding[:, 0]
        y = embedding[:, 1]

        self.df['x'] = x
        self.df['y'] = y

    def feature_importance(self, dep: str, indep: str):
        """\
        Calculates feature importance of columns in self.df (especially for x and y after dmap/umap were run).
        Returns DataFrame with the 10 most important features and their according importance values.

        Parameters
        ----------
        dep
            name of column which should represent the dependent variable
        indep
            name of column which should represent the independent variable

        Returns
        ----------
        DataFrame

        """

        data = self.df.copy()
        data = data.drop(indep, axis=1)

        # split data
        train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

        X_train = train_df.drop(dep, axis=1)
        y_train = train_df[dep]
        X_test = test_df.drop(dep, axis=1)
        y_test = test_df[dep]

        print("length of data for training:", len(X_train))
        print("length of data for testing:", len(X_test))

        # run RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # predict dependent variable
        y_pred = model.predict(X_test)

        # r²-value calculation
        r2 = r2_score(y_test, y_pred)
        print("r² Score:", r2)

        # Feature Importance
        importance = model.feature_importances_

        # sort features according to importance
        s_id = np.argsort(importance)
        pos = np.arange(s_id.shape[0])

        # MinMax scaling
        scaler = MinMaxScaler()
        importance_scaled = scaler.fit_transform(importance.reshape(-1, 1)).flatten()

        # importance
        total_importance = np.sum(importance_scaled)
        percentage_importance = (importance_scaled / total_importance) * 100

        # show top ten
        top_n = 10
        s_id = s_id[-top_n:]
        features = pd.DataFrame(
            {'index1': np.array(X_train.columns)[s_id], 'importance_normalized': importance_scaled[s_id],
             'percentage_importance': percentage_importance[s_id]})
        return features

    def plot_feature_importance(self, features, path: str, base_width: int = 10, base_height: int = 6):
        """\
        Plots the ten most important features and returns a pyplot.
        Needs the ten top features and their importances as parameters.

        Parameters
        ----------
        features
            DataFrame returned by function feature_importance
        path
            path to dict where plot should be saved
        base_width
            width of plot
        base_height
            height of plot

        Returns
        ----------
        Show plot

        """

        num_features = len(features)

        plot_width = base_width
        plot_height = base_height + 0.2 * num_features

        ax = features.plot.bar(x='index1',
                               y='importance_normalized',
                               color='darkgray',
                               legend=False,
                               figsize=(plot_width, plot_height),
                               width=0.8, fontsize=20)

        plt.xlabel('')
        plt.ylabel('Importance', fontsize=20)

        # adjust text height
        for i, v in enumerate(features['percentage_importance']):
            if features['importance_normalized'][i] + 0.01 > 1.1:
                text_height = 1
            else:
                text_height = features['importance_normalized'][i] + 0.01
            ax.text(i, text_height, f'{v:.1f}%', ha='center', va='bottom', fontsize=16, color='black')

        plt.title('Top 10 Features', fontsize=30, loc='left')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')

        if path is not None:
            try:
                plt.savefig(path, dpi=300, bbox_inches='tight')
                print(f"Plot successfully saved to {path}")
            except Exception as e:
                print(f"An error occurred while saving to png: {e}")

        return plt.show()

    def cluster_kmeans(self, n_cluster: int, label_x: str, label_y: str):
        """\
        Cluster self.df by kmeans clustering and show result as plt.show().

        Parameters
        ----------
        number_cluster
            number of clusters
        label_x
            x-axis label
        label_y
            y-axis label

        Returns
        ----------
        Show plot
        """

        kmeans_labels = cluster.KMeans(n_clusters=n_cluster).fit_predict(self.df)

        # plot
        plt.style.use('seaborn-v0_8-poster')
        plt.figure(figsize=(6, 6))

        plt.scatter(self.df[['x']],
                    self.df[['y']],
                    c=kmeans_labels,
                    s=1,
                    cmap='Set1');

        plt.title('K-Means Clustering')
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.xticks([])
        plt.yticks([])

        self.df['kmeans_cluster'] = kmeans_labels.tolist()
        return plt.show()

    def cluster_gmm(self, number_component: int, random_s: int, label_x: str, label_y: str):
        """\
        Cluster self.df by Gaussian Mixture Modeles and plot result.

        Parameters
        ----------
        number_component
            number of components of gmm clustering
        random_s
            random state of gmm clustering
        label_x
            x-axis label
        label_y
            y-axis label

        Returns
        ----------
        Show plot

        """

        gmm = GaussianMixture(n_components=number_component, random_state=random_s)

        gmm.fit(self.df)
        gaussian_labels = gmm.predict(self.df)

        # plot
        plt.style.use('seaborn-v0_8-poster')
        plt.figure(figsize=(6, 6))

        plt.scatter(self.df[['x']],
                    self.df[['y']],
                    c=gaussian_labels,
                    s=1,
                    cmap='Set1');

        plt.title('Gaussian Mixture Clustering')
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.xticks([])
        plt.yticks([])

        self.df['GMM_cluster'] = gaussian_labels.tolist()
        return plt.show()

    def cluster_hdbscan(self, cluster_size: int, label_x: str, label_y: str):
        """\
        Cluster self.df by hdbscan and plot result.

        Parameters
        ----------
        cluster_size
            size of clusters
        label_x
            x-axis label
        label_y
            y-axis label

        Returns
        ----------
        Show plot

        """

        clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size, gen_min_span_tree=True)

        clusterer.fit(self.df)
        hdbscan_labels = clusterer.labels_

        # Plot
        outliers_mask = hdbscan_labels == -1
        plt.style.use('seaborn-v0_8-poster')
        plt.figure(figsize=(6, 6))

        plt.scatter(self.df[['x']],
                    self.df[['y']],
                    c=hdbscan_labels,
                    cmap='Spectral',
                    s=5)

        plt.scatter(self.df.loc[outliers_mask, 'x'],
                    self.df.loc[outliers_mask, 'y'],
                    s=4,
                    c='gray',
                    marker='v',
                    label='Outliers',
                    alpha=0.5)

        plt.title('HDBSCAN Clustering')
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.legend(markerscale=6)

        plt.xticks([])
        plt.yticks([])

        self.df['hdbscan_cluster'] = hdbscan_labels.tolist()
        return plt.show()



    def check_dataframe(self):
        """\
        check if df is empty
        """
        if self.df.empty:
            raise ValueError("Dataframe is empty.")

    def prepare_data_source(self):
        """\
        set ColumnDataSource as self.df and return ColumnDataSource
        """
        return ColumnDataSource(self.df)

    def configure_hover_tooltips(self, feature: str, hover_tooltips: list[tuple[str, str]] = None):
        """\
        Set hover tooltips. Either standard or personalized.

        Parameters
        ----------
        feature
            column of dataset
        hover_tooltips

        Returns
        ----------
        HoverTool

        """
        if hover_tooltips is None:
            hover_tooltips = [
                ("Feature", f"{feature}"),
                ("Index", "ID"),
                ("X-Value", "x"),
                ("Y-Value", "y")
            ]

        hover_tooltips_formatted = "".join([
            f"<div><span style='font-size: 12px; font-weight: bold;'>{label}:</span> <span style='font-size: 12px;'>@{field}</span></div>"
            for label, field in hover_tooltips
        ])

        hovertool_kwargs = {
            "name": "data",
            "tooltips": hover_tooltips_formatted
        }
        return HoverTool(**hovertool_kwargs)

    def create_base_plot(self, fig_width: int,
                         fig_height: int,
                         fig_title: str,
                         label_x: str,
                         label_y: str,
                         range_x: list[float],
                         range_y: list[float],
                         tools_emb,
                         title_align: str):
        """\
        Settings for plotting umap/densmap.

        Parameters
        ----------
        fig_width
            width of figure
        fig_height
            height of figure
        fig_title
            title of figure
        label_x
            x-axis label
        label_y
            y-axis label
        range_x
            range of x-axis
        range_y
            range of y-axis
        tools_emb
            tools that should be embedded
        title_align
            position of title

        Returns
        ----------
        plot

        """
        if not bokeh_above_three:
            plot_kwargs = dict(
                plot_width=fig_width,
                plot_height=fig_height,
                title=fig_title,
                tools=tools_emb,
                x_axis_label=label_x,
                y_axis_label=label_y,
                x_range=Range1d(start=range_x[0], end=range_x[1]),
                y_range=Range1d(start=range_y[0], end=range_y[1])
            )
        else:
            plot_kwargs = dict(
                width=fig_width,
                height=fig_height,
                title=fig_title,
                tools=tools_emb,
                x_axis_label=label_x,
                y_axis_label=label_y,
                x_range=Range1d(start=range_x[0], end=range_x[1]),
                y_range=Range1d(start=range_y[0], end=range_y[1])
            )

        plot = figure(**plot_kwargs)

        # configure title
        plot.title.align = title_align
        plot.title.text_font_size = '30pt'

        return plot

    def configure_axes_and_legend(self, plot, show_axes: bool, show_legend: bool):
        """\
        Settings for axes and legend.

        Parameters
        ----------
        plot
        show_axes
            add axes
        show_legend
            add legend

        Returns
        ----------
        None

        """

        # axes
        if not show_axes:
            plot.xaxis.visible = False
            plot.yaxis.visible = False

        plot.grid.visible = False
        plot.outline_line_color = None

        plot.xaxis.axis_label_text_font_size = "20pt"
        plot.yaxis.axis_label_text_font_size = "20pt"

        plot.xaxis.ticker = []
        plot.yaxis.ticker = []

        # legend
        if show_legend:
            plot.legend.title_text_font_style = "bold"
            plot.legend.background_fill_alpha = 0.0
            plot.legend.border_line_alpha = 0
            plot.legend.label_text_font_size = '20pt'
            plot.legend.title_text_font_size = '20pt'
            plot.legend.glyph_height = 30
            plot.legend.glyph_width = 30
            plot.add_layout(plot.legend[0], 'center')
        else:
            plot.legend.visible = False

    def cat_plot(self, feature: str,
                 subs: list[str],
                 colors: list[str],
                 outputf: str,
                 fig_width: int,
                 fig_height: int,
                 fig_title: str,
                 label_x: str,
                 label_y: str,
                 range_x: list[float],
                 range_y: list[float],
                 hover_tooltips: list[tuple[str, str]] = None,
                 show_legend: bool = False,
                 point_size: int = 10,
                 point_alpha: float = 0.6,
                 show_axes: bool = False,
                 title_align: str = 'center') -> None:

        """\
        Create plot with categorical color mapper.Choose feature, colors and more.
        Loads html file.

        Parameters
        ----------
        feature
            feature for categorical mapper
        subs
            values of feature
        colors
            list of colors
        outputf
            path to outputfile
        fig_width
            width of figure
        fig_height
            height of figure
        fig_title
            title of figure
        label_x
            x-axis label
        label_y
            y-axis label
        range_x
            x-axis range
        range_y
            y-axis range
        hover_tooltips
            hover tooltips
        show_legend
            possibility to add legend
        point_size
            size of points
        point_alpha
            alpha of points
        show_axes
            add axes
        title_align
            position of title

        Returns
        ----------
        show plot

        """

        self.check_dataframe()
        output_file(outputf)

        source = self.prepare_data_source()
        hover_emb = self.configure_hover_tooltips(feature, hover_tooltips)
        cm = CategoricalColorMapper(palette=colors, factors=subs)

        tools_emb = ['save', 'lasso_select', 'pan', 'wheel_zoom', 'reset', hover_emb]
        plot = self.create_base_plot(fig_width,
                                     fig_height,
                                     fig_title,
                                     label_x,
                                     label_y,
                                     range_x,
                                     range_y,
                                     tools_emb,
                                     title_align)

        plot.circle('x', 'y',
                    size=point_size,
                    color={'field': feature, 'transform': cm},
                    alpha=point_alpha,
                    source=source,
                    name="data",
                    legend_group=feature)

        self.configure_axes_and_legend(plot, show_axes, show_legend)

        show(plot)

    def lin_plot(self,
                 outputf: str,
                 feature: str,
                 colors: list[str],
                 fig_width: int,
                 fig_height: int,
                 fig_title: str,
                 label_x: str,
                 label_y: str,
                 range_x: list[float],
                 range_y: list[float],
                 hover_tooltips: list[tuple[str, str]] = None,
                 show_legend: bool = False,
                 point_size: int = 3,
                 point_alpha: float = 0.7,
                 show_axes: bool = True,
                 title_align: str = 'center') -> None:

        """\
        Create plot with Linear color mapper.Choose feature, colors and more.
        Loads html file.

        Parameters
        ----------
        outputf
            path to output file
        feature
            feature for linear mapper
        colors
            list of colors
        fig_width
            width of figure
        fig_height
            height of figure
        fig_title
            title of figure
        label_x
            x-axis label
        label_y
            y-axis label
        range_x
            range of x-axis
        range_y
            range of y-axis
        hover_tooltips
            hover tooltips
        show_legend
            add legend
        point_size
            size of points
        point_alpha
            alpha of points
        show_axes
            add axes
        title_align
            title position

        Returns
        ----------
        show plot

        """

        self.check_dataframe()
        output_file(outputf)

        source = self.prepare_data_source()
        hover_emb = self.configure_hover_tooltips(feature, hover_tooltips)
        lm = LinearColorMapper(palette=colors, low=min(self.df[feature]), high=max(self.df[feature]))

        tools_emb = ['save', 'lasso_select', 'pan', 'wheel_zoom', 'reset', hover_emb]
        plot = self.create_base_plot(fig_width,
                                     fig_height,
                                     fig_title,
                                     label_x,
                                     label_y,
                                     range_x,
                                     range_y,
                                     tools_emb,
                                     title_align)

        plot.circle('x', 'y',
                    size=point_size,
                    fill_color={'field': feature, 'transform': lm},
                    alpha=point_alpha,
                    line_alpha=0,
                    line_width=0.03,
                    source=source,
                    name="data",
                    legend_group=feature)

        self.configure_axes_and_legend(plot, show_axes, show_legend)

        show(plot)

    def cell_plot(self,
                  cluster_column,
                  top_clusters: int,
                  palette,
                  ID,
                  x_label,
                  y_label,
                  png: bool,
                  svg: bool,
                  name: str):
        """\
        Create plot with highlighted IDs.Save as png or svg.

        Parameters
        ----------
        cluster_column
            column with cluster numbers
        top_clusters
            number of top clusters that should be highlighted
        palette
            colours for top clusters
        ID
            IDs of cells of interest
        x_label
            label for x-axis
        y_label
            label for y-axis
        png
            saved as .png if True
        svg
            saved as.svg if True
        name
            name of file

        Returns
        --------
        None

        """
        if cluster_column not in self.df.columns:
            raise ValueError(f"Cluster column '{cluster_column}' not found in DataFrame.")

        # identify most frequent clusters
        top_clusters = self.df[cluster_column].value_counts().index[:top_clusters]

        cluster_dtype = self.df[cluster_column].dtype

        if np.issubdtype(cluster_dtype, np.number):
                other_label = -1
        else:
                other_label = 'Other'

        self.df["cluster_limited"] = self.df[cluster_column].apply(lambda x: x if x in top_clusters else other_label)
        

        unique_clusters = np.sort(self.df["cluster_limited"].unique())
        cluster_colors = {cluster: palette[i] for i, cluster in enumerate(unique_clusters)}
        colors = self.df["cluster_limited"].map(cluster_colors)

        # Check which IDs are in the DataFrame
        existing_ids = [idx for idx in ID if idx in self.df.index]
        missing_ids = [idx for idx in ID if idx not in self.df.index]

        # Select Highlighted Cells
        highlight_cells = self.df.loc[existing_ids] if existing_ids else None

        # Plot
        plt.figure(figsize=(10, 8))

        # Scatter Plot for All Cells
        plt.scatter(
            self.df["x"],
            self.df["y"],
            c=colors,
            alpha=0.5,
            s=50
        )

        # Highlight Specific IDs (Larger, Red, Black Border)
        if highlight_cells is not None:
            plt.scatter(
                highlight_cells["x"],
                highlight_cells["y"],
                c="red",
                edgecolors="black",
                s=200
            )

            # Adjust Label Positions to Prevent Overlap
            texts = []

            for idx, row in highlight_cells.iterrows():
                texts.append(plt.text(
                    row["x"], row["y"], str(idx),
                    fontsize=10, fontweight="bold",
                    color="black", ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
                ))

            # Automatically Adjust Text Labels to Avoid Overlaps
            adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

        # Axis Labels
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)

        # Cluster Legend (Without Highlighted Dots)
        legend_labels = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors[cl], markersize=10,
                       label=f'Cluster {cl}' if cl != -1 else 'Other')
            for cl in unique_clusters
        ]

        plt.legend(handles=legend_labels, loc="upper right", fontsize=10)

        if svg is True:
            svg_filename = name + ".svg"
            plt.savefig(svg_filename, format="svg", bbox_inches="tight", dpi=300)
            print(f" plot saved as: {svg_filename}")
            plt.show()

        if png is True:
            png_filename = name + ".png"
            plt.savefig(png_filename, format="png", bbox_inches="tight", dpi=300)
            print(f" plot saved as: {png_filename}")
            plt.show()

        if png is False and svg is False:
            print("unable to save plot")

        if missing_ids:
            print(f" The following highlight IDs were not found in the dataset: {missing_ids}")
