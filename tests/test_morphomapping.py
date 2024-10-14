from morphomapping import MM
import unittest
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TestMorphoMapping(unittest.TestCase):

    def setUp(self):
        self.test_fcs_path="./files/FCS_test_file.fcs"
        self.test_csv_path = "./files/CSV_test_file_small.csv"
        self.test_xlsx_path = "./files/xlsx_test_file_small.xlsx"

        self.df = pd.DataFrame({
            "Area_Ch01": [5.7, 14, 9.6],
            "Donor": ["JBZ", "IKH", "LME"],
            "File_Number": [1, 2, 3]
        })
        self.df.to_csv(self.test_csv_path, index=False)

        self.extra_df1= pd.DataFrame({
            "Height": [7, 9, 0]
        })

        self.extra_df2 = pd.DataFrame({
            "Perimeter": [8, 20, 900]
        })


    def test_convert_to_CSV(self):
        test_MM=MM()

        test_MM.convert_to_CSV(self.test_fcs_path, self.test_csv_path)
        assert os.path.exists(self.test_csv_path) \

        #invalid fcs path
        test_fcs_path = "./files/FCS_test_file_not_found.fcs"

        with self.assertRaises(FileNotFoundError):
            test_MM.convert_to_CSV(test_fcs_path, self.test_csv_path)

        #invalid csv path
        test_csv_path = ""

        with self.assertRaises(ValueError) as a:
            test_MM.convert_to_CSV(self.test_fcs_path, test_csv_path)

        self.assertEqual(str(a.exception), "The CSV path is empty or invalid.")


    def test_read_CSV(self):
        test_MM = MM()

        test_csv_path = "./files/CSV_test_file2.csv"

        df1 = pd.DataFrame({
            "Area + Length": [5, 10.4, 0],
            "Leukocytes & Erys": ['1', 'yes', '0']
        })

        df1.to_csv(test_csv_path, index=False)

        df2 = test_MM.read_CSV(test_csv_path, add_index=True, index_name='Index')
        renamed_columns = ["Area_plus_Length", "Leukocytes_and_Erys"]
        self.assertListEqual(list(df2.columns), renamed_columns)
        self.assertIn('Index', df2.index.names)

        if os.path.exists(test_csv_path):
            os.remove(test_csv_path)

    def test_get_features(self):
        test_MM=MM()

        test_MM.read_CSV(self.test_csv_path)
        features1=test_MM.get_features()
        features2=["Area_Ch01","Donor","File_Number"]

        self.assertEqual(features1,features2)

    def test_get_df(self):
        test_MM=MM()

        test_MM.read_CSV(self.test_csv_path)
        df1=test_MM.get_df()
        df2=pd.DataFrame({
            "Area_Ch01": [5.7, 14, 9.6],
            "Donor": ["JBZ", "IKH", "LME"],
            "File_Number": [1, 2, 3]
        })

        pd.testing.assert_frame_equal(df1, df2)


    def test_add_metadata(self):
        test_MM = MM()
        test_MM.df = pd.DataFrame()

        with self.assertRaises(ValueError) as a:
            test_MM.add_metadata("donor", "Donor 1")

        self.assertEqual(str(a.exception), "Dataframe is empty.")

    def test_rename_variables(self):
        test_MM = MM()

        test_MM.read_CSV(self.test_csv_path)

        new_label = {
            "Area_Ch01": "Area CD4"
        }

        test_MM.rename_variables(new_label)

        df1 = pd.DataFrame({
            "Area CD4": [5.7, 14, 9.6],
            "Donor": ["JBZ", "IKH", "LME"],
            "File_Number": [1, 2, 3]
        })

        self.assertEqual(list(test_MM.df.columns), list(df1.columns))

        pd.testing.assert_frame_equal(test_MM.df, df1)

    def test_select_condition(self):
        test_MM = MM()

        test_MM.read_CSV(self.test_csv_path)

        with self.assertRaises(ValueError) as context:
            test_MM.select_condition('Non_Existing_Column', 0)
        self.assertEqual(str(context.exception), "Column 'Non_Existing_Column' does not exist.")

        with self.assertRaises(ValueError) as context:
            test_MM.select_condition('Donor', 'Non_Existing_Value')
        self.assertEqual(str(context.exception), "Value 'Non_Existing_Value' does not exist in column 'Donor'.")

        expected_df = pd.DataFrame({
            "Area_Ch01":5.7,
            "Donor": ["JBZ"],
            "File_Number": 1
        })

        test_MM.select_condition('Donor', 'JBZ')

        pd.testing.assert_frame_equal(test_MM.df.reset_index(drop=True), expected_df.reset_index(drop=True))

    def test_select_events(self):
        test_MM = MM()
        test_MM.read_CSV(self.test_csv_path)

        with self.assertRaises(ValueError) as v:
            test_MM.select_events(100)
        self.assertEqual(str(v.exception), f"Number of events '100' is larger than the number of rows (3).")

        test_MM.select_events(1)
        self.assertEqual(test_MM.df.shape[0], 1)
        self.assertTrue(test_MM.df.index.is_monotonic_increasing)

    def test_drop_variables(self):
        test_MM = MM()
        test_MM.read_CSV(self.test_csv_path)

        with self.assertRaises(ValueError) as v:
            test_MM.drop_variables('Non_Existing_Column')
        self.assertEqual(str(v.exception), "Column(s) ['Non_Existing_Column'] do not exist.")

        test_MM.drop_variables('File_Number')
        columns = ['Area_Ch01', 'Donor']
        self.assertListEqual(list(test_MM.df.columns), columns)

        test_MM.drop_variables('Area_Ch01', 'Donor')
        self.assertTrue(test_MM.df.empty)

    def test_save_feature(self):
        test_MM = MM()
        test_MM.read_CSV(self.test_csv_path)

        with self.assertRaises(ValueError) as context:
            test_MM.save_feature('Non_Existing_Column')
        self.assertEqual(str(context.exception), "Column(s) ['Non_Existing_Column'] do not exist.")

        df1 = test_MM.save_feature('Area_Ch01', 'Donor')
        df2 = self.df[['Area_Ch01', 'Donor']].copy()
        pd.testing.assert_frame_equal(df1, df2)

        df3 = test_MM.save_feature('File_Number')
        df4 = self.df[['File_Number']].copy()
        pd.testing.assert_frame_equal(df3,df4)

    def test_concat_variables(self):
        test_MM = MM()
        test_MM.read_CSV(self.test_csv_path)

        df1 = test_MM.concat_variables(self.extra_df1, self.extra_df2)
        df2 = pd.concat([self.df, self.extra_df1, self.extra_df2], axis=1)

        pd.testing.assert_frame_equal(df1, df2)
        pd.testing.assert_frame_equal(test_MM.df, df1)

    def test_save_xlsx(self):
        test_MM = MM()
        test_MM.read_CSV(self.test_csv_path)

        test_MM.save_xlsx("./files/xlsx_test_file_small_2.xlsx")
        self.assertTrue(os.path.exists("./files/xlsx_test_file_small_2.xlsx"))

        df1 = pd.read_excel("./files/xlsx_test_file_small_2.xlsx", index_col=0)
        df2 = pd.read_csv(self.test_csv_path, index_col=0)
        pd.testing.assert_frame_equal(df1,df2)



    def test_save_csv(self):
        test_MM=MM()

        test_MM.read_CSV(self.test_csv_path)
        test_MM.save_csv("./files/CSV_test_file_small_2.csv")

        self.assertTrue(os.path.exists("./files/CSV_test_file_small_2.csv"))

        df1 = pd.read_csv("./files/CSV_test_file_small_2.csv", index_col=0)
        df2 = pd.read_csv(self.test_csv_path, index_col=0)
        pd.testing.assert_frame_equal(df1,df2)

    def test_concat_df(self):
        test_MM = MM()
        test_MM.read_CSV(self.test_csv_path)

        df = pd.DataFrame({
            "test_number": [1, 2, 3],
        })

        test_MM.concat_df(df)

        dfc1 = test_MM.get_df()
        dfc2 = pd.concat([self.df,df], join='inner')

        pd.testing.assert_frame_equal(dfc1, dfc2)

    def test_update_column_values(self):
        test_MM = MM()
        test_MM.read_CSV(self.test_csv_path)

        updated_values = {
            "JBZ": "1",
            "IKH": "2",
            "LME": "3"
        }

        test_MM.update_column_values("Donor", updated_values)

        df2 = pd.DataFrame({
            "Area_Ch01": [5.7, 14, 9.6],
            "Donor": ["1", "2", "3"],
            "File_Number": [1, 2, 3]
        })

        pd.testing.assert_frame_equal(test_MM.df, df2)

        with self.assertRaises(ValueError):
            test_MM.update_column_values("Non_Existent_Column", updated_values)

    def test_minmax_norm(self):
        test_MM = MM()
        test_MM.df = pd.DataFrame({
            "c1": [1, 2, 3, 4, 5],
            "c2": [10, 20, 30, 40, 50],
            "c3": [1000, 2000, 3000, 4000,5000]
        })

        test_MM.minmax_norm()

        df1 = pd.DataFrame({
            "c1": [0.00, 0.25, 0.50, 0.75, 1.00],
            "c2": [0.00, 0.25, 0.50, 0.75, 1.00],
            "c3": [0.00, 0.25, 0.50, 0.75, 1.00]
        })
        pd.testing.assert_frame_equal(test_MM.df, df1)

        test_MM.df = pd.DataFrame({
            "c1": [1, 2, 3, 4, 5],
            "c2": [10, 20, 30, 40, 50],
            "c3": [1000, 2000, 3000, 4000, 5000]
        })

        test_MM.minmax_norm("c1", "c2")

        df2 = pd.DataFrame({
            "c1": [0.00, 0.25, 0.50, 0.75, 1.00],
            "c2": [0.00, 0.25, 0.50, 0.75, 1.00],
            "c3": [1000, 2000, 3000, 4000, 5000]
        })
        pd.testing.assert_frame_equal(test_MM.df, df2)

        with self.assertRaises(ValueError):
            test_MM.minmax_norm("Non_Existent_Column", "B")

        with self.assertRaises(ValueError):
            test_MM.minmax_norm("c3", "c1")

    def test_quant_scaler(self):
        test_MM = MM()
        test_MM.df = pd.DataFrame({
            "c1": [1, 2, 3, 4, 5],
            "c2": [10, 20, 30, 40, 50],
            "c3": [1000, 2000, 3000, 4000, 5000]
        })

        test_MM.quant_scaler()

        expected_df1 = pd.DataFrame({
            "c1": [0.0, 0.25, 0.5, 0.75, 1.0],
            "c2": [0.0, 0.25, 0.5, 0.75, 1.0],
            "c3": [0.0, 0.25, 0.5, 0.75, 1.0]
        })
        pd.testing.assert_frame_equal(test_MM.df, expected_df1)

        test_MM.df = pd.DataFrame({
            "c1": [1, 2, 3, 4, 5],
            "c2": [10, 20, 30, 40, 50],
            "c3": [1000, 2000, 3000, 4000, 5000]
        })
        test_MM.quant_scaler("c1", "c2")

        expected_df2 = pd.DataFrame({
            "c1": [0.0, 0.25, 0.5, 0.75, 1.0],
            "c2": [0.0, 0.25, 0.5, 0.75, 1.0],
            "c3": [1000, 2000, 3000, 4000, 5000]
        })
        pd.testing.assert_frame_equal(test_MM.df, expected_df2)

        with self.assertRaises(ValueError):
            test_MM.quant_scaler("Non_Existent_Column", "c2")

        with self.assertRaises(ValueError):
            test_MM.quant_scaler("c3", "c1")

    def test_umap(self):
        test_MM = MM()
        test_MM.df = pd.DataFrame({
            "c1": np.random.rand(10),
            "c2": np.random.rand(10),
            "c3": np.random.rand(10)
        })

        test_MM.umap(nn=2, mdist=0.5, met='euclidean')

        self.assertIn('x', test_MM.df.columns)
        self.assertIn('y', test_MM.df.columns)

        self.assertEqual(test_MM.df.shape[1], 5)

        self.assertTrue(np.issubdtype(test_MM.df['x'].dtype, np.number))
        self.assertTrue(np.issubdtype(test_MM.df['y'].dtype, np.number))

    def test_dmap(self):
        test_MM = MM()
        test_MM.df = pd.DataFrame({
            "c1": np.random.rand(10),
            "c2": np.random.rand(10),
            "c3": np.random.rand(10)
        })

        test_MM.dmap(nn=2, dlambda=0.7, mdist=0.5, met='euclidean')

        self.assertIn('x', test_MM.df.columns)
        self.assertIn('y', test_MM.df.columns)

        self.assertEqual(test_MM.df.shape[1], 5)

        self.assertTrue(np.issubdtype(test_MM.df['x'].dtype, np.number))
        self.assertTrue(np.issubdtype(test_MM.df['y'].dtype, np.number))

    def test_feature_importance(self):
        test_MM = MM()
        test_MM.df = pd.DataFrame({
            "c1": np.random.rand(500),
            "c2": np.random.rand(500),
            "c3": np.random.rand(500),
            "c4": np.random.rand(500),
            "c5": np.random.rand(500),
            "c6": np.random.rand(500),
            "c7": np.random.rand(500),
            "c8": np.random.rand(500),
            "c9": np.random.rand(500),
            "c10": np.random.rand(500),
            "independent_variable": np.random.rand(500),
            "dependent_variable": np.random.rand(500)
        })

        features = test_MM.feature_importance(dep='dependent_variable', indep='independent_variable')

        self.assertTrue(set(['index1', 'importance_normalized', 'percentage_importance']).issubset(features.columns))

        self.assertEqual(features.shape[0], 10)

        self.assertTrue(np.issubdtype(features['importance_normalized'].dtype, np.number))
        self.assertTrue(np.issubdtype(features['percentage_importance'].dtype, np.number))

        sorted = features.sort_values(by='percentage_importance', ascending=True).reset_index(drop=True)
        self.assertTrue(sorted['percentage_importance'].equals(features['percentage_importance']))

    def test_plot_feature_importance(self):
        test_MM = MM()
        test_MM.df = pd.DataFrame({
            "c1": np.random.rand(500),
            "c2": np.random.rand(500),
            "c3": np.random.rand(500),
            "c4": np.random.rand(500),
            "c5": np.random.rand(500),
            "c6": np.random.rand(500),
            "c7": np.random.rand(500),
            "c8": np.random.rand(500),
            "c9": np.random.rand(500),
            "c10": np.random.rand(500),
            "independent_variable": np.random.rand(500),
            "dependent_variable": np.random.rand(500)
        })

        features = test_MM.feature_importance(dep='dependent_variable', indep='independent_variable')

        plot_path = "./files/test_plot.png"
        test_MM.plot_feature_importance(features, path=plot_path)

        self.assertTrue(os.path.exists(plot_path))

        os.remove(plot_path)

    def test_cluster_kmeans(self):
        test_MM = MM()
        test_MM.df = pd.DataFrame({
            "x": np.random.rand(1000),
            "y": np.random.rand(1000)
        })

        test_MM.cluster_kmeans(n_cluster=5,label_x='UMAP 1',label_y='UMAP 2')

        self.assertIn('kmeans_cluster', test_MM.df.columns)

        clusters = test_MM.df['kmeans_cluster'].unique()
        self.assertEqual(len(clusters), 5)

        plot_path = "./test_kmeans_plot.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')

        self.assertTrue(os.path.exists(plot_path))

        os.remove(plot_path)

    def test_cluster_gmm(self):
        test_MM = MM()
        test_MM.df = pd.DataFrame({
            "x": np.random.rand(1000),
            "y": np.random.rand(1000)
        })

        test_MM.cluster_gmm(number_component=5, random_s=42, label_x='UMAP 1',label_y='UMAP 2')

        self.assertIn('GMM_cluster', test_MM.df.columns)

        clusters = test_MM.df['GMM_cluster'].unique()
        self.assertEqual(len(clusters), 5)

        plot_path = "./test_gmm_plot.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')

        self.assertTrue(os.path.exists(plot_path))

        os.remove(plot_path)

    def test_cluster_hdbscan(self):
        test_MM = MM()
        test_MM.df = pd.DataFrame({
            "x": np.random.rand(1000),
            "y": np.random.rand(1000)
        })

        test_MM.cluster_hdbscan(cluster_size=5, label_x='UMAP 1',label_y='UMAP 2')

        self.assertIn('hdbscan_cluster', test_MM.df.columns)

        clusters = test_MM.df['hdbscan_cluster'].unique()
        self.assertTrue(len(set(clusters) - {-1}) >= 1)

        plot_path = "./test_hdbscan_plot.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')

        self.assertTrue(os.path.exists(plot_path))

        os.remove(plot_path)

    def test_cat_plot(self):
        test_MM = MM()
        test_MM.df = pd.DataFrame({
            "x": np.random.rand(1000),
            "y": np.random.rand(1000),
            "feature": np.random.choice(['sub1', 'sub2', 'sub3','sub4'], 1000)
        })

        plot_path = "./test_cat_plot.html"

        test_MM.cat_plot(
            feature='feature',
            subs=['sub1', 'sub2', 'sub3','sub4'],
            colors=['blue', 'red', 'yellow','orange'],
            outputf=plot_path,
            fig_width=1000,
            fig_height=1000,
            fig_title='Cat Plot',
            label_x='X',
            label_y='Y',
            range_x=[-10, 10],
            range_y=[-15, 15],
            hover_tooltips=[("Feature", "@feature"), ("X-Value", "@x"), ("Y-Value", "@y")],
            show_legend=True,
            point_size=5,
            point_alpha=0.3,
            show_axes=True,
            title_align='center'
        )

        self.assertTrue(os.path.exists(plot_path))

        os.remove(plot_path)

    def test_lin_plot(self):
        test_MM = MM()
        test_MM.df = pd.DataFrame({
            "x": np.random.rand(1000),
            "y": np.random.rand(1000),
            "feature": np.random.rand(1000)
        })

        plot_path = "./test_lin_plot.html"


        test_MM.lin_plot(
            feature='feature',
            colors='Plasma256',
            outputf=plot_path,
            fig_width=1000,
            fig_height=1000,
            fig_title='Lin Plot',
            label_x='X',
            label_y='Y',
            range_x=[-10, 10],
            range_y=[-15, 15],
            hover_tooltips=[("Feature", "@feature"), ("X-Value", "@x"), ("Y-Value", "@y")],
            show_legend=True,
            point_size=5,
            point_alpha=0.3,
            show_axes=True,
            title_align='center'
        )

        self.assertTrue(os.path.exists(plot_path))

        os.remove(plot_path)


if __name__ =='__main__':
    unittest.main()