import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from configparser import ConfigParser


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _parse_feature(image_string, label=None):
    if label is not None:
        feature = {
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string)
        }
    else:
        feature = {
        'image_raw': _bytes_feature(image_string)
        }
    return feature

def image_example(image_string, label=None):
    image_shape = tf.image.decode_jpeg(image_string).shape
    if label is not None:
        feature = _parse_feature(image_string, label)
    else:
        feature = _parse_feature(image_string)
    return tf.train.Example(features=tf.train.Features(feature=feature))

def annotations_dict(df):
    _dict = {}
    for index, row in df.iterrows():
        label = row[1:].to_numpy()
        image = row[0]
        _dict[str(image)] = label
    return _dict

def gen_image_list(df):
    return df.iloc[:,0].to_list()


def sharding_data(img_per_shard, image_list, annotations, tfrecord_dir):
    """Split data into n shards of tfrecord"""
    if len(image_list) < img_per_shard:
        n_shards = 1
    else:
        print(f'????????image_list length:{len(image_list)}?????????')
        n_shards = int(len(image_list)/img_per_shard) + (1 if len(image_list) % img_per_shard != 0 else 0)
        print(f"%%%%%%number of shards: {n_shards}%%%%%%%%")
    
    tfrecords_path = "{}_{}.records"
    name = tfrecord_dir
    index = 0
    for shard in tqdm(range(n_shards)):
        tfrecords_shard_path = tfrecords_path.format(name, '%.5d-of-%.5d' % (shard, n_shards - 1))
        end = index + img_per_shard
        images_shard_list = image_list[index: end]
        # print(f'image shard list:{images_shard_list}')
        write_tfrecord(tfrecords_shard_path, images_shard_list, annotations=annotations)
        index = end

def fold_df(df, output_dir, n_fold=10):
    fold_size = int(np.ceil(len(df)/n_fold))
    index = 0

    for fold in tqdm(range(n_fold)):
        if fold == n_fold:
            end = len(df) + 1
        else:
            end = index+fold_size
        csv_path = os.path.join(output_dir, 'fold'+str(fold+1))
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        df_valid = df.iloc[index:end, :].to_csv(os.path.join(csv_path, 'valid.csv'), index=False)
        df_train = pd.concat([df.iloc[0:fold_size*fold,:], df.iloc[end:,:]]).to_csv(os.path.join(csv_path, 'train.csv'), index=False)
        index = end       

def write_tfrecord(tfrecord_dir, image_list, annotations=None):
    with tf.io.TFRecordWriter(tfrecord_dir) as writer:
        for filename in tqdm(image_list):
            image_string = open(filename, 'rb').read()
            if annotations:
                label = annotations[filename]
                tf_example = image_example(image_string, label)
            else:
                tf_example = image_example(image_string)
            writer.write(tf_example.SerializeToString())

def gen_tfrecord(df, tf_record_dir, data_is_labeled, sharding=True, img_per_shard=None):
    image_list = gen_image_list(df)
    print(f'!!!!!!!image_list_length:{len(image_list)}!!!!!!!')
    if data_is_labeled:
        annotations = annotations_dict(df)
        print(f'length of annotations: {len(annotations)}')
        if sharding == True:
            if img_per_shard is not None:
                sharding_data(img_per_shard, image_list, annotations, tf_record_dir)
            else:
                print("ValueError: Please enter a value for img_per_shard \n")
                sys.exit()
        else:
            write_tfrecord(tf_record_dir +'.records', image_list, annotations=annotations)
    else:
        write_tfrecord(os.path.join(tf_record_dir+'.records'), image_list, annotations=None)   

               
def main():
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    output_dir = cp["DATA"].get("output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset_name = cp["DATA"].get("dataset_name")
    csv_file = cp["DATA"].get("dataset_csv_file")
    csv_val_file = cp["DATA"].get("dataset_csv_val_file")
    csv_test_file = cp["DATA"].get("dataset_test_csv_file")
    print(f'%%%%%%{csv_test_file}%%%%%%')
    img_per_shard = cp["DATA"].getint("img_per_shard")
    sharding = cp["DATA"].get("sharding")
    data_is_labeled = cp["DATA"].get("data_is_labeled")
    test_data_is_labeled = cp["DATA"].get("test_data_is_labeled")
    n_fold = cp["DATA"].getint("n_fold")
    dataset_name = cp["DATA"].get("dataset_name")

    if csv_file != '':
        df = pd.read_csv(csv_file).sample(frac=1)
        print(f'********generating training and validation data records********')
        image_list = gen_image_list(df)
        if n_fold > 1:
            train_data_name = dataset_name+'_train'
            val_data_name = dataset_name+'_val'
            fold_df(df, output_dir, n_fold=n_fold)
            print("*****write data into the folder of each fold***** \n")
            for fold in tqdm(range(n_fold)):
                tf_record_dir_train = os.path.join(output_dir, 'fold'+str(fold+1), dataset_name+'_train')
                gen_tfrecord(df, tf_record_dir_train, data_is_labeled, sharding=sharding, img_per_shard=img_per_shard)
                tf_record_dir_val = os.path.join(output_dir, 'fold'+str(fold+1), dataset_name+'_val')
                gen_tfrecord(df, tf_record_dir_val, data_is_labeled, sharding=False)
        else:
            tf_record_dir = os.path.join(output_dir, dataset_name+'_train')
            gen_tfrecord(df, tf_record_dir, data_is_labeled, sharding=sharding, img_per_shard=img_per_shard)
            df_val = pd.read_csv(csv_val_file)
            tf_record_val_dir = os.path.join(output_dir, dataset_name + '_val')
            gen_tfrecord(df_val, tf_record_val_dir, data_is_labeled, sharding=False)
    else:
        print("*******You have no training and validation data to generate*******")

    if csv_test_file != '':
        print(f'********generating test data records********')
        df_test = pd.read_csv(csv_val_file)
        tf_record_dir = os.path.join(output_dir, dataset_name+'_test')
        gen_tfrecord(df_test, tf_record_dir, test_data_is_labeled, sharding=False)
    else:
        print("*******You have no test data to generate*******")

    
        


if __name__ == "__main__":
    main()

