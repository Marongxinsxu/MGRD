from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MetricAbstract:
    def __init__(self):
        self.bigger= True

    def __str__(self):
        return self.__class__.__name__

    def __call__(self,groundtruth,pred ) ->float:
        raise Exception("Not callable for an abstract function")

class acc(MetricAbstract):
    def __init__(self):
        self.bigger = True

    def __call__(self, pred, test_data):
        labels = test_data.data[:, -1]
        acc = accuracy_score(labels, pred)

        return acc
class pre(MetricAbstract):
    def __init__(self):
        self.bigger = True

    def __call__(self, pred, test_data):
        labels = test_data.data[:, -1]
        pre = precision_score(labels, pred)

        return pre
class rec(MetricAbstract):
    def __init__(self):
        self.bigger = True

    def __call__(self, pred, test_data):
        labels = test_data.data[:, -1]
        rec = recall_score(labels, pred)

        return rec
class f1(MetricAbstract):
    def __init__(self):
        self.bigger = True

    def __call__(self, pred, test_data):
        labels = test_data.data[:, -1]
        f1 = f1_score(labels, pred)

        return f1
