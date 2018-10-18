# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class RecommendService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            with open(os.path.join(model_path, 'news-recommender.pkl'), 'rb') as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, user_item_rating_df, item_tag_distr_df, top_n=10):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""

        clf = cls.get_model()
        # convert item_tag_df to tag_item_distr
        if clf.tag_id_df is None:
            raise Exception('tag_id_df is None. need to train model first')

        if 'tag' in item_tag_distr_df.columns:
            item_tag_distr_df.drop(columns=['tag'])

        item_tag_distr_df = item_tag_distr_df.pivot(
            index='item_id',
            columns='tag_id',
            values='distr'
        )
        # fill up lost tag_id column
        for tag_id in clf.tag_id_df.tag_id.values:
            if tag_id not in item_tag_distr_df.columns:
                item_tag_distr_df[tag_id] = 0
        item_tag_distr_df.fillna(0, inplace=True)
        tag_item_distr = item_tag_distr_df[clf.tag_id_df.tag_id.values]
        print('tag_item_distr')
        print(tag_item_distr)
        return clf.predict(user_item_rating_df, tag_item_distr, top_n)