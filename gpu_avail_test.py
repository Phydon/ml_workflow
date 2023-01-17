import tensorflow as tf


def gpu_check():
    gpu = tf.config.list_physical_devices('GPU')

    if gpu:
        print("Num GPUs Available: ", len(gpu))
        print(f"GPUs available: {gpu}")
    else:
        print("No GPUs available")


if __name__ == "__main__":
    gpu_check()
