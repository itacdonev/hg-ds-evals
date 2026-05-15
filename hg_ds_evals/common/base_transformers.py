from pyspark.ml import Transformer

class BaseTransformer(Transformer):
    """
    Base transformer with common functionality.
    """

    def __init__(self):
        super(BaseTransformer, self).__init__()
