"""Configure native CPU execution before model/tensor creation."""
import importlib.metadata
import os
import platform
import sys


def configure(seed=42, threads=4):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return tf


def environment():
    import tensorflow as tf
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("tensorflow", "keras", "numpy", "Pillow", "matplotlib")},
            "devices": [str(d) for d in tf.config.get_visible_devices()],
            "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
            "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
            "oneDNN": os.environ.get("TF_ENABLE_ONEDNN_OPTS")}
