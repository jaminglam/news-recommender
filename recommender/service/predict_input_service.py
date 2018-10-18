class PredictInputService(object):
    REQUEST_TYPE_ENDPOINT = 1
    REQUEST_TYPE_BATCH_TRANSFORM = 2
    REQUEST_CONTENT_TYPE_JSON = 'application/json'
    REQUEST_CONTENT_TYPE_CSV = 'text/csv'


    @staticmethod
    def get_input_data(request_type, request_content_type):
        
