"""Tensorflow dataset API."""
from collections import namedtuple

import tensorflow as tf


LoaderDataset = namedtuple("LoaderDataset", ["feed_dict", "initializer", "dataset", "other"])


class BaseDataLoader(object):
    """Interface between tensorflow model and tensorflow dataset."""

    def __init__(self, graph, session, params):
        """Initialize."""
        self.graph = graph
        self.session = session
        self.params = params

        self.datasets = {}
        self.type = None
        self.shape = None
        self.handle = None
        self.input_tensors = None

    def add_datasets(self, dict_in, batch_size=None, prefetch_multi=1, other=None):
        """
        Add tensorflow dataset into loader.

        Parameters
        ----------
        dict_in : dict[str, tf.data.Dataset]
            dictionary with name as key, tf.dataset as value.
        batch_size : Option[int]
            setting batch for input datasets. If None use params.batch_size.
        other : dict[str, other_obj]
            store other object with the same key name in dict_in.

        Returns
        -------
        None

        """
        with self.graph.as_default():
            for name, dataset in dict_in.items():
                # setting dataset batch
                if batch_size:
                    dataset = dataset.prefetch(batch_size * prefetch_multi)
                    dataset = dataset.batch(batch_size, drop_remainder=True)
                else:
                    dataset = dataset.prefetch(self.params.batch_size * prefetch_multi)
                    dataset = dataset.batch(self.params.batch_size, drop_remainder=True)
                # check input datasets type and shape.
                if self.type and self.shape:
                    asrmsg = "Wrong type or shape with exist dataset. {} {}"
                    assert len(dataset.output_shapes) == len(self.shape), asrmsg.format(
                            len(dataset.output_shapes), len(self.shape))
                    for d_shape, s_shape in zip(dataset.output_shapes, self.shape):
                        assert d_shape.as_list()[1:] == s_shape.as_list()[1:], asrmsg.format(
                            d_shape.as_list()[1:], s_shape.as_list()[1:])
                    assert dataset.output_types == self.type, asrmsg
                else:
                    self.type = dataset.output_types
                    self.shape = dataset.output_shapes
                    # Change batch dim to None
                    for shape in self.shape:
                        shape.dims[0] = tf.Dimension(None)
                    self.handle = tf.placeholder(tf.string, shape=[])
                    data_iter = tf.data.Iterator.from_string_handle(
                        self.handle, self.type, self.shape)
                    self.input_tensors = data_iter.get_next()

                data_iter = dataset.make_initializable_iterator()
                string_handle = self.session.run(data_iter.string_handle())
                data_init = data_iter.initializer
                feed_dict = {self.handle: string_handle}
                # setting LoaderDataset
                other_stored = None
                if other is not None:
                    if name in other:
                        other_stored = other[name]
                loader_dataset = LoaderDataset(
                    feed_dict=feed_dict, initializer=data_init, dataset=dataset,
                    other=other_stored)

                self.datasets[name] = loader_dataset

    def get_dataset(self, name):
        """
        Get tensorflow dataset by name.

        Same as BaseDataLoader.datasets[name].dataset

        Parameters
        ----------
        name : str

        Returns
        -------
        tf.data.Dataset

        """
        return self.datasets[name].dataset

    def get_feed_dict(self, name):
        """
        Get dataset feed_dict by name.

        Same as BaseDataLoader.datasets[name].feed_dict

        Parameters
        ----------
        name : str

        Returns
        -------
        dict

        """
        return self.datasets[name].feed_dict

    def get_initializer(self, name):
        """
        Get dataset initializer by name.

        Same as BaseDataLoader.datasets[name].initializer

        Parameters
        ----------
        name : str

        Returns
        -------
        tf.Tensor

        """
        return self.datasets[name].initializer
