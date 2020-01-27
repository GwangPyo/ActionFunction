import ray
from multiprocessing import Lock


@ray.remote
class SingletonModel(object):

    def __init__(self, model):
        """
        Singletone model for remote evaluation or prediction. Wrapped Keras Model API for work
        """
        self.__instant = model

    def predict_on_batch(self, x, y, sample_weight=None, class_weight=None, *args, **kwargs):
        return self.__instance.predict_on_batch(x, y, sample_weight, class_weight, *args, **kwargs)

    def test_on_batch(self, x, y, sample_weight=None, *args, **kwargs):
        return self.__instance.test_on_batch(x, y, sample_weight, *args, **kwargs)

    def train_on_batch(self,x, y, sample_weight=None, class_weight=None, *args, **kwargs):
        return self.__instance.train_on_batch(x, y, sample_weight, class_weight, *args, **kwargs)

    def predict(self, x, batch_size=None, verbose=0, steps=None, callbacks=None, *args, **kwargs):
        return self.__instance.predict(x, batch_size, verbose, steps, callbacks, *args, **kwargs)

    def fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None,
                      validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False,
                      shuffle=True, initial_epoch=0, *args, **kwargs):
        return self.__instance.fit_generator(generator, steps_per_epoch, epochs, verbose,
                                            callbacks, validation_data, validation_steps, class_weight,
                                            max_queue_size, workers, use_multiprocessing,
                                            shuffle, initial_epoch, *args, **kwargs)

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None, *args, **kwargs):
        return self.__instance.fit(x, y, batch_size, epochs, verbose, callbacks, validation_split,
                                  validation_data, shuffle, class_weight, sample_weight,
                                  initial_epoch, steps_per_epoch, validation_steps, *args, **kwargs)

    def evaluate(self,x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, *args,
                 **kwargs):
        return self.__instance.evaluate(x, y, batch_size, verbose, sample_weight, steps, callbacks, *args, **kwargs)

    def evaluate_generator(self, generator, steps=None, callbacks=None,
                           max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0):
        return self.__instance.evaluate_generator(generator, steps, callbacks,
                                                 max_queue_size, workers, use_multiprocessing, verbose)

    def predict_generator(self, generator, steps=None, callbacks=None, max_queue_size=10,
                          workers=1, use_multiprocessing=False, verbose=0):
        return self.__instance.predict_on_batch(generator, steps, callbacks, max_queue_size,
                                               workers, use_multiprocessing, verbose)

    def get_layer(self, name=None, index=None):
        return self.__instance.get_layer(name, index)

    def save_weights(self, filepath):
        self.__instance.save_weights(filepath)

    def load_weights(self, filepath, by_name=False):
        self.__instance.load_weights(filepath, by_name)


InstanceDictLock = Lock()
SingletonModelInstances = {}


def register(key, model):
    with InstanceDictLock:
        if key in SingletonModelInstances.keys():
            raise AttributeError("The given key {} is already registered".format(key))
        SingletonModelInstances[key] = SingletonModel.remote(model)
    return key


def get_model(key):
    return SingletonModelInstances[key]


