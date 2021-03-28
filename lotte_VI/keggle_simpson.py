import math, re, os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
# from kaggle_datasets import KaggleDatasets
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE

try: # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError: # no TPU found, detect GPUs
    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
    #strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # for clusters of multi-GPU machines

# print("Number of accelerators: ", strategy.num_replicas_in_sync)

# GCS_DS_PATH = KaggleDatasets().get_gcs_path("the-simpsons-characters-data-tfrecords")

IMAGE_SIZE = [224, 224]
EPOCHS = 25
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

# GCS_PATH_SELECT = GCS_DS_PATH
# GCS_PATH = GCS_PATH_SELECT
TRAINING_FILENAMES = tf.io.gfile.glob('C:/data/LPD_competition/train')
VALIDATION_FILENAMES = tf.io.gfile.glob('C:/data/LPD_competition/train')
TEST_FILENAMES = tf.io.gfile.glob('C:/data/LPD_competition/test')

CLASSES = ['0', '1', '10', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '11', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '12', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '13', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '14', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '15', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '16', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '17', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '18', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '19', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '2', '20', '200', '201', '202', '203', '204', 
'205', '206', '207', '208', '209', '21', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '22', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '23', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '24', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '25', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '26', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '27', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '28', '280', '281', '282', '283', '284', '285', '286', '287', '288', '289', '29', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299', '3', '30', '300', '301', '302', '303', '304', '305', '306', '307', '308', '309', '31', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '32', '320', '321', '322', '323', '324', '325', '326', '327', '328', '329', '33', '330', '331', '332', '333', '334', '335', '336', '337', '338', '339', '34', '340', '341', '342', '343', 
'344', '345', '346', '347', '348', '349', '35', '350', '351', '352', '353', '354', '355', '356', '357', '358', '359', 
'36', '360', '361', '362', '363', '364', '365', '366', '367', '368', '369', '37', '370', '371', '372', '373', '374', '375', '376', '377', '378', '379', '38', '380', '381', '382', '383', '384', '385', '386', '387', '388', '389', '39', '390', '391', '392', '393', '394', '395', '396', '397', '398', '399', '4', '40', '400', '401', '402', '403', '404', '405', '406', '407', '408', '409', '41', '410', '411', '412', '413', '414', '415', '416', '417', '418', '419', '42', '420', '421', '422', '423', '424', '425', '426', '427', '428', '429', '43', '430', '431', '432', '433', '434', '435', '436', '437', '438', '439', '44', '440', '441', '442', '443', '444', '445', '446', '447', '448', '449', '45', '450', '451', '452', '453', '454', '455', '456', '457', '458', '459', '46', '460', '461', '462', '463', '464', '465', '466', '467', '468', '469', '47', '470', '471', '472', '473', '474', '475', '476', '477', '478', '479', '48', '480', '481', '482', 
'483', '484', '485', '486', '487', '488', '489', '49', '490', '491', '492', '493', '494', '495', '496', '497', '498', 
'499', '5', '50', '500', '501', '502', '503', '504', '505', '506', '507', '508', '509', '51', '510', '511', '512', '513', '514', '515', '516', '517', '518', '519', '52', '520', '521', '522', '523', '524', '525', '526', '527', '528', '529', '53', '530', '531', '532', '533', '534', '535', '536', '537', '538', '539', '54', '540', '541', '542', '543', '544', '545', '546', '547', '548', '549', '55', '550', '551', '552', '553', '554', '555', '556', '557', '558', '559', '56', '560', '561', '562', '563', '564', '565', '566', '567', '568', '569', '57', '570', '571', '572', '573', '574', '575', '576', '577', '578', '579', '58', '580', '581', '582', '583', '584', '585', '586', '587', '588', '589', '59', '590', '591', '592', '593', '594', '595', '596', '597', '598', '599', '6', '60', '600', '601', '602', '603', '604', '605', '606', '607', '608', '609', '61', '610', '611', '612', '613', '614', '615', '616', '617', '618', '619', '62', '620', '621', '622', '623', '624', '625', '626', '627', '628', '629', '63', '630', '631', '632', '633', '634', '635', '636', '637', '638', '639', '64', '640', '641', '642', '643', '644', '645', '646', '647', '648', '649', '65', '650', '651', '652', '653', '654', '655', '656', '657', '658', '659', '66', '660', '661', '662', '663', '664', '665', '666', '667', '668', '669', '67', '670', '671', '672', '673', '674', '675', '676', '677', '678', '679', '68', '680', '681', '682', '683', '684', '685', '686', '687', '688', '689', '69', '690', '691', '692', '693', '694', '695', '696', '697', '698', '699', '7', '70', '700', '701', '702', '703', '704', '705', '706', '707', '708', '709', '71', '710', '711', '712', '713', 
'714', '715', '716', '717', '718', '719', '72', '720', '721', '722', '723', '724', '725', '726', '727', '728', '729', 
'73', '730', '731', '732', '733', '734', '735', '736', '737', '738', '739', '74', '740', '741', '742', '743', '744', '745', '746', '747', '748', '749', '75', '750', '751', '752', '753', '754', '755', '756', '757', '758', '759', '76', '760', '761', '762', '763', '764', '765', '766', '767', '768', '769', '77', '770', '771', '772', '773', '774', '775', '776', '777', '778', '779', '78', '780', '781', '782', '783', '784', '785', '786', '787', '788', '789', '79', '790', '791', '792', '793', '794', '795', '796', '797', '798', '799', '8', '80', '800', '801', '802', '803', '804', '805', '806', '807', '808', '809', '81', '810', '811', '812', '813', '814', '815', '816', '817', '818', '819', '82', '820', '821', '838', '839', '84', '840', '841', '842', '843', '844', '845', '846', '847', '848', '849', '85', '850', '851', '852', 
'853', '854', '855', '856', '857', '858', '859', '86', '860', '861', '862', '863', '864', '865', '866', '867', '868', 
'869', '87', '870', '871', '872', '873', '874', '875', '876', '877', '878', '879', '88', '880', '881', '882', '883', '884', '885', '886', '887', '888', '889', '89', '890', '891', '892', '893', '894', '895', '896', '897', '898', '899', '9', '90', '900', '901', '902', '903', '904', '905', '906', '907', '908', '909', '91', '910', '911', '912', '913', '914', '915', '916', '917', '918', '919', '92', '920', '921', '922', '923', '924', '925', '926', '927', '928', '929', '93', '930', '931', '932', '933', '934', '935', '936', '937', '938', '939', '94', '940', '941', '942', '943', '944', '945', '946', '947', '948', '949', '95', '950', '951', '952', '953', '954', '955', '956', '957', '958', '959', '96', '960', '961', '962', '963', '964', '965', '966', '967', '968', '969', '97', '970', '971', '972', '973', '974', '975', '976', '977', '978', '979', '98', '980', '981', '982', '983', '984', '985', '986', '987', '988', '989', '99', '990', '991', 
'992', '993', '994', '995', '996', '997', '998', '999']

# ========================= Visualization utilities ======================
# numpy and matplotlib defaults
np.set_printoptions(threshold=15, linewidth=80)

def batch_to_numpy_images_and_labels(data):
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object: # binary string in this case, these are image ID strings
        numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is the case for test data)
    return numpy_images, numpy_labels

def title_from_label_and_target(label, correct_label):
    if correct_label is None:
        return CLASSES[label], True
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',
                                CLASSES[correct_label] if not correct else ''), correct

def display_one_flower(image, title, subplot, red=False, titlesize=16):
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)
    
def display_batch_of_images(databatch, predictions=None):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)]
        
    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images)//rows
        
    # size and spacing
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols,1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))
    
    # display
    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):
        title = '' if label is None else CLASSES[label]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)
    
    #layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()

def display_confusion_matrix(cmat, score, precision, recall):
    plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.matshow(cmat, cmap='Reds')
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 12})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 12})
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.show()
    
def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])

# ================== Datasets ====================================
def to_float32(image, label):
    return tf.cast(image, tf.float32), label

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum # returns a dataset of image(s)

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def data_augment(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
#     image = tf.image.random_flip_left_right(image)
#     image = tf.image.random_flip_up_down(image)
#     image = tf.image.rot90(image)

#     image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.dtypes.int32))
    
#     minval = IMAGE_SIZE[0] // 2
#     maxval = IMAGE_SIZE[0]
#     random_shape = tf.random.uniform(shape=[2], minval=minval, maxval=maxval, dtype=tf.dtypes.int32)
#     image = tf.image.random_crop(image, size=[random_shape[0], random_shape[1], 3])
#     image = tf.image.resize(image, size=[IMAGE_SIZE[0], IMAGE_SIZE[1]])
    
#     if random_shape[0] < IMAGE_SIZE[0] or random_shape[1] < IMAGE_SIZE[0]:
#         image = tf.image.resize(image, size=[IMAGE_SIZE[0], IMAGE_SIZE[1]])
#     else:
#         image = tf.image.random_crop(image, size=[IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    
#     image = tf.image.random_saturation(image, 0.7, 1.3)
#     image = tf.image.random_brightness(image, 0.05)

    image = tf.clip_by_value(image, 0, 1)
    
    return image, label   

def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))




