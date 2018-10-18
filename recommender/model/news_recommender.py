import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix


class NewsRecommender:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.user_group_tag_distr = None
        self._clustering = None
        # need to think about open this attribute to be set
        self.item_buffer_factor = 1.2
        self.max_item_id = -1
        self.max_user_id = -1
        self.shape_y = -1
        self.tag_id_df = None

    # @classmethod
    # def from_file(self, filepath):
    #     obj = joblib.load(filepath)
    #     return obj

    # def to_file(self, filepath):
    #     joblib.dump(self, filepath, compress=3)
    #     print("dump model to {}".format(filepath))

    def _split_user_group(self, user_item_rating_df, clustering_labels):
        print("clustering labels size: {}".format(clustering_labels.size))
        clustering_label_series = pd.Series(clustering_labels)
        clustering_label_df_data = {
            'user_id': clustering_label_series.index,
            'user_group': clustering_label_series.values
        }
        clustering_label_df = pd.DataFrame(
            data=clustering_label_df_data)
        print('user_item_rating_df shape {}'.format(user_item_rating_df.shape))
        user_group_item_rating_df = pd.merge(
            left=user_item_rating_df,
            right=clustering_label_df,
            how='left',
            on='user_id'
        )
        print("user_group_item_rating_df shape {}".format(
            user_group_item_rating_df.shape))
        user_group_item_rating_df_list = [
            user_group_item_rating_df[user_group_item_rating_df.user_group == label]
            for label in range(self.n_clusters)
        ]
        return user_group_item_rating_df_list

    def _get_tag_distr_df(self, user_item_rating_df, item_tag_df):
        tag_rating_df = pd.merge(
            left=user_item_rating_df,
            right=item_tag_df,
            on='item_id'
        )
        tag_rating_df.drop(['user_id', 'item_id', 'tag'], axis=1, inplace=True)
        tag_rating_distr_df = tag_rating_df.groupby('tag_id').sum()
        rating_sum = tag_rating_df['rating'].sum()
        tag_rating_distr_df['probability'] = (
            tag_rating_distr_df.rating / rating_sum
        )
        tag_distr_df = tag_rating_distr_df[['probability']]
        tag_distr_df = tag_distr_df.reset_index()
        tag_distr_df = pd.merge(
            left=self.tag_id_df,
            right=tag_distr_df,
            how='left'
        )
        tag_distr_df.fillna(0, inplace=True)
        del tag_rating_distr_df
        return tag_distr_df

    def _cluster_users(self, user_readed_coo_matrix):
        """AgglomerativeClustering in sklearn requires dense array as input which
           will cause memory error when input is large dataset. At this moment, use
           KMean clustering instead though it only supports euclidean similarity.
           clustering = AgglomerativeClustering(
               n_clusters=4,
               affinity='cosine',
               linkage='average'
           )
        Args:
        user_readed_coo_matrix (scipy.sparse.coo_matrix): user-item-score matrix 
                                                          with coordinate foramt
                                                          as sparse matrix to
                                                          save space
        """
        n_clusters = self.n_clusters
        print("try to create {} clusters model".format(n_clusters))
        clustering = KMeans(
            n_clusters=n_clusters,
            random_state=0
        )
        clustering.fit(user_readed_coo_matrix)
        print(clustering.labels_)
        self._clustering = clustering
        return clustering.labels_

    def _convert_2_coo_matrix(self, coo_row, coo_col, coo_data, shape):
        return coo_matrix(
            (coo_data.astype(np.double), (coo_row, coo_col)),
            shape=shape
        )

    # def _convert_2_user_readed_coo_matrix(self, user_item_rating_df):
    #     coo_row = user_item_rating_df.user_id.values
    #     coo_col = user_item_rating_df.item_id.values
    #     coo_data = user_item_rating_df.rating.values
    #     max_user_id = coo_row.max()
    #     shape_x = max_user_id + 1
    #     if self.shape_y < 0:
    #         raise Exception('shape_y can not be zero')
    #     max_item_id = coo_col.max()

    #     user_readed_coo_matrix = coo_matrix(
    #         (coo_data.astype(np.double), (coo_row, coo_col)),
    #         shape=(shape_x, self.shape_y)
    #     )
    #     return user_readed_coo_matrix

    def fit(self, user_item_rating_df, item_tag_df):
        print('start training')
        fit_coo_row = user_item_rating_df.user_id.values
        fit_coo_col = user_item_rating_df.item_id.values
        fit_coo_data = user_item_rating_df.rating.values
        self.max_user_id = fit_coo_row.max()
        self.max_item_id = fit_coo_col.max()
        fit_shape_x = self.max_user_id + 1
        fit_shape_y = int((self.max_item_id + 1) * self.item_buffer_factor)
        self.shape_y = fit_shape_y
        fit_shape = (fit_shape_x, fit_shape_y)
        user_readed_coo_matrix = self._convert_2_coo_matrix(
            fit_coo_row,
            fit_coo_col,
            fit_coo_data,
            fit_shape
        )
        print('user_readed_coo_matrix shape: {}'.format(user_readed_coo_matrix.shape))
        clustering_labels = self._cluster_users(user_readed_coo_matrix)
        user_group_item_rating_df_list = self._split_user_group(
            user_item_rating_df,
            clustering_labels
        )
        del user_item_rating_df
        user_group_tag_distr_list = []
        tag_id_list = item_tag_df.tag_id.value_counts().index.tolist()
        self.tag_id_df = pd.DataFrame({'tag_id': tag_id_list})
        print('tag_id_list: {}'.format(tag_id_list))
        for user_group_item_rating_df in user_group_item_rating_df_list:
            tag_distr_df = self._get_tag_distr_df(
                user_group_item_rating_df,
                item_tag_df
            )
            tag_distr_df.sort_values(by=['tag_id'], inplace=True)
            tag_distr_df.set_index('tag_id', inplace=True)
            print(tag_distr_df)
            user_group_tag_distr_list.append(tag_distr_df.probability.values)
        print("user_group_tag_distr_list")
        print(user_group_tag_distr_list)
        self.user_group_tag_distr = np.array(user_group_tag_distr_list)
        print(self.user_group_tag_distr)
        print('user_group_tag_distr shape: {}'.format(self.user_group_tag_distr.shape))
        print('training done')

    def _cal_sim_with_tag_distr_items(self, tag_distr_items):
        if self.user_group_tag_distr is not None:
            _Y = np.array(tag_distr_items)
            print("_Y shape: {}".format(_Y.shape))
            _X = self.user_group_tag_distr
            print("_X shape: {}".format(_X.shape))
            return euclidean_distances(_X, _Y)
        else:
            raise Exception('Model not trained yet. Please fit data to train.')

    def _get_sorted_top_n_items(self, sim_matrix, top_n=10):
        sorted_index_matrix = np.argsort(sim_matrix, axis=1)
        return sorted_index_matrix[:, :top_n]

    def _sorted_index_2_sorted_item_id(
        self,
        tag_distr_items,
        sorted_index_matrix
    ):
        item_id_array = tag_distr_items.index.values
        print("item_id_array: {}".format(item_id_array))

        def vectorize_func(sorted_index_matrix):
            return item_id_array[sorted_index_matrix]

        sorted_item_id_matrix = vectorize_func(sorted_index_matrix)
        print("sorted_item_id_matrix: {}".format(sorted_item_id_matrix))
        return sorted_item_id_matrix

    def _predict_for_user_groups(self, user_groups, sorted_item_id_matrix):
        user_group_list = user_groups.tolist()
        prediction_list = [sorted_item_id_matrix[user_group, :].tolist()
                           for user_group in user_group_list]
        predictions = np.array(prediction_list)
        return predictions

    def _attach_user_id_2_predictions(self, predictions, user_id_series):
        pred_shape = predictions.shape
        user_id_df = user_id_series.to_frame('user_id')
        print("user_id_df: {}".format(user_id_df))
        headers = ['top_{}'.format(i) for i in range(pred_shape[1])]
        print("headers: {}".format(headers))
        prediction_df = pd.DataFrame(predictions, columns=headers)
        prediction_df.reset_index(inplace=True)
        prediction_df.rename({"index": "user_id"}, axis='columns', inplace=True)
        prediction_df = pd.merge(
            left=prediction_df,
            right=user_id_df,
            how='inner'
        )
        print('prediction_df: {}'.format(prediction_df))
        # print('prediction_df: {}'.format(prediction_df))
        return prediction_df

    def predict(self, user_item_rating_df, tag_distr_items, top_n=10):
        try:
            print(user_item_rating_df)
            predict_coo_row = user_item_rating_df.user_id.values
            predict_coo_col = user_item_rating_df.item_id.values
            predict_coo_data = user_item_rating_df.rating.values
            predict_shape_x = predict_coo_row.max() + 1
            predict_shape_y = predict_coo_col.max() + 1
            if self.shape_y < 0:
                raise ValueError('need to run fit to train data first')
            if predict_shape_y > self.shape_y:
                raise ValueError('invalid shape_y: {}, item_id exceed range'.format(predcit_shape_y))
            predict_shape = (predict_shape_x, self.shape_y)
            user_readed_coo_matrix = self._convert_2_coo_matrix(
                predict_coo_row,
                predict_coo_col,
                predict_coo_data,
                predict_shape
            )
            if self._clustering is None:
                raise Exception('_clustering can not be None')
            print("user_readed_coo_matrix:")
            print(user_readed_coo_matrix)
            user_groups = self._clustering.predict(user_readed_coo_matrix)
            print('user groups: ')
            print(user_groups)
            print('tag_distr_items index: ')
            print(tag_distr_items.index)
            print('tag_distr_items values: ')
            print(tag_distr_items.values)
            sim_matrix = self._cal_sim_with_tag_distr_items(
                tag_distr_items.values)
            print("sim matrix shape: {}".format(sim_matrix.shape))
            sorted_index_matrix = self._get_sorted_top_n_items(
                sim_matrix,
                top_n
            )
            print('sorted index matrix: ')
            print(sorted_index_matrix)
            sorted_item_id_matrix = self._sorted_index_2_sorted_item_id(
                tag_distr_items,
                sorted_index_matrix
            )
            predictions = self._predict_for_user_groups(
                user_groups,
                sorted_item_id_matrix
            )
            user_id_series = user_item_rating_df.user_id.drop_duplicates()
            prediction_df = self._attach_user_id_2_predictions(
                predictions,
                user_id_series
            )
            return prediction_df
            # return most_sim_matrix
        except Exception as ex:
            print('Caught this error: ' + repr(ex))
