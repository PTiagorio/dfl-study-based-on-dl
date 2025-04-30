from configuration_and_functions import *

run_information_folder = '/content/drive/MyDrive/runs/' + RUN_NAME + '/run_information'
cmd = 'mkdir -p ' + run_information_folder
os.system(cmd)

# Commented out IPython magic to ensure Python compatibility.
# %%capture cap_input_variables --no-stderr

print("RUN_NAME: " + str(RUN_NAME))
print("use_custom_dataset: " + str(use_custom_dataset))
print("commands_amount: " + str(commands_amount))
print("nodes: " + str(nodes))
print("awgn_backgorund_noise_percentage: " + str(awgn_backgorund_noise_percentage))
print("poisoned_nodes: " + str(poisoned_nodes))
print("poisoned_nodes_noise_percentage: " + str(poisoned_nodes_noise_percentage))
print("use_real_backgorund_noise: " + str(use_real_backgorund_noise))
print("use_quantization: " + str(use_quantization))
print("use_prunning: " + str(use_prunning))
print("test_amount: " + str(test_amount))
print("val_amount: " + str(val_amount))
print("batch_size: " + str(batch_size))
print("epochs: " + str(epochs))
print("unbalance: " + str(unbalance))
print("evaluate_tflite_training: " + str(evaluate_tflite_training))
print("transfer_learning_on_last_layer_only: " + str(transfer_learning_on_last_layer_only))
print("load_weights: " + str(load_weights))
#print("do_global_model: " + str(do_global_model))
#print("use_knowledge_destillation: " + str(use_knowledge_destillation))

f = open(run_information_folder + "/input_variables.txt", "w")
f.write("RUN_NAME: " + str(RUN_NAME) + "\n")
f.write("use_custom_dataset: " + str(use_custom_dataset) + "\n")
f.write("commands_amount: " + str(commands_amount) + "\n")
f.write("nodes: " + str(nodes) + "\n")
f.write("awgn_backgorund_noise_percentage: " + str(awgn_backgorund_noise_percentage) + "\n")
f.write("poisoned_nodes: " + str(poisoned_nodes) + "\n")
f.write("poisoned_nodes_noise_percentage: " + str(poisoned_nodes_noise_percentage) + "\n")
f.write("use_real_backgorund_noise: " + str(use_real_backgorund_noise) + "\n")
f.write("use_quantization: " + str(use_quantization) + "\n")
f.write("use_prunning: " + str(use_prunning) + "\n")
f.write("test_amount: " + str(test_amount) + "\n")
f.write("val_amount: " + str(val_amount) + "\n")
f.write("batch_size: " + str(batch_size) + "\n")
f.write("epochs: " + str(epochs) + "\n")
f.write("unbalance: " + str(unbalance) + "\n")
f.write("evaluate_tflite_training: " + str(evaluate_tflite_training) + "\n")
f.write("transfer_learning_on_last_layer_only: " + str(transfer_learning_on_last_layer_only) + "\n")
f.write("load_weights: " + str(load_weights) + "\n")
f.close()

"""### Generate a background noise dataset

Whether you're using the default speech dataset or a custom dataset, you should have a good set of background noises so your model can distinguish speech from other noises (including silence).

Because the following background samples are provided in WAV files that are a minute long or longer, we need to split them up into smaller one-second samples so we can reserve some for our test dataset. We'll also combine a couple different sample sources to build a comprehensive set of background noises and silence:
"""

tf.keras.utils.get_file('speech_commands_v0.2.tar.gz',
                        'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
                        cache_dir='./',
                        cache_subdir='dataset-speech',
                        extract=True)
tf.keras.utils.get_file('background_audio.zip',
                        'https://storage.googleapis.com/download.tensorflow.org/models/tflite/sound_classification/background_audio.zip',
                        cache_dir='./',
                        cache_subdir='dataset-background',
                        extract=True)

"""**Note:** Although there is a newer version available, we're using v0.01 of the speech commands dataset because it's a smaller download. v0.01 includes 30 commands, while v0.02 adds five more ("backward", "forward", "follow", "learn", and "visual")."""

# Unlike word examples, the noise samples in the Speech Commands v0.02 dataset
# are not divided into 1-second snippets. Instead, they are stored as longer
# recordings. Therefore we need to cut them up in to 1-second snippet .wav
# files.

noise_wav_paths = glob.glob(os.path.join('./dataset-speech/_background_noise_', '*.wav'))
# snippets_dir = './dataset-speech/background'
os.makedirs(snippets_dir, exist_ok=True)

for noise_wav_path in noise_wav_paths:
  print("Extracting snippets from %s..." % noise_wav_path)
  extract_snippets(noise_wav_path, snippet_duration_sec=1.0)


#Old Code:
'''
if use_real_backgorund_noise:
  # Create a list of all the background wav files
  files = glob.glob(os.path.join('./dataset-speech/_background_noise_', '*.wav'))
  files = files + glob.glob(os.path.join('./dataset-background', '*.wav'))

  background_dir = './background'
  os.makedirs(background_dir, exist_ok=True)

  # Loop through all files and split each into several one-second wav files
  for file in files:
    filename = os.path.basename(os.path.normpath(file))
    print('Splitting', filename)
    name = os.path.splitext(filename)[0]
    rate = librosa.get_samplerate(file)
    length = round(librosa.get_duration(filename=file))
    for i in range(length - 1):
      start = i * rate
      stop = (i * rate) + rate
      data, _ = sf.read(file, start=start, stop=stop)
      sf.write(os.path.join(background_dir, name + str(i) + '.wav'), data, rate)
'''

"""### Prepare the speech commands dataset

We already downloaded the speech commands dataset, so now we just need to prune the number of classes for our model.

This dataset includes over 30 speech command classifications, and most of them have over 2,000 samples. But because we're using transfer learning, we don't need that many samples. So the following code does a few things:

+ Specify which classifications we want to use, and delete the rest.
+ Keep only 150 samples of each class for training (to prove that transfer learning works well with smaller datasets and simply to reduce the training time).
+ Create a separate directory for a test dataset so we can easily run inference with them later.
"""

file_name = run_information_folder + "/splitted_data.txt"

f = open(file_name, "w")
f.write('')
f.close()

if not use_custom_dataset:

  dataset_dir = './dataset-speech'
  train_dir = './dataset-train'
  val_dir = './dataset-val'
  test_dir = './dataset-test'

  sample_count_total = 0
  test_count_total = 0
  val_count_total = 0
  train_count_total = 0

  # Delete all directories that are not in our commands list
  dirs = glob.glob(os.path.join(dataset_dir, '*/'))
  for dir in dirs:
    name = os.path.basename(os.path.normpath(dir))
    if name not in commands:
      shutil.rmtree(dir)

  TARGET_SAMPLE_RATE = 44100

  dirs = glob.glob(os.path.join(dataset_dir, '*/'))
  for dir in dirs:
    files = glob.glob(os.path.join(dir, '*.wav'))

    test_count = 0
    val_count = 0
    train_count = 0

    sample_count_total = sample_count_total + len(files)

    class_dir = os.path.basename(os.path.normpath(dir))
    for file in files:
      if which_set(file, val_amount, test_amount) == 'testing':
        os.makedirs(os.path.join(test_dir, class_dir), exist_ok=True)
        #os.rename(file, os.path.join(test_dir, class_dir, os.path.basename(file)))
        #shutil.copy(file, os.path.join(test_dir, class_dir, os.path.basename(file)))

        sample_rate, xs = wavfile.read(file)
        xs = xs.astype(np.float32)
        xs = librosa.resample(xs, sample_rate, TARGET_SAMPLE_RATE).astype(np.int16)
        wavfile.write(os.path.join(test_dir, class_dir, os.path.basename(file)), TARGET_SAMPLE_RATE, xs)

        test_count = test_count + 1
      elif which_set(file, val_amount, test_amount) == 'validation':
        os.makedirs(os.path.join(val_dir, class_dir), exist_ok=True)
        #os.rename(file, os.path.join(val_dir, class_dir, os.path.basename(file)))
        #shutil.copy(file, os.path.join(val_dir, class_dir, os.path.basename(file)))

        sample_rate, xs = wavfile.read(file)
        xs = xs.astype(np.float32)
        xs = librosa.resample(xs, sample_rate, TARGET_SAMPLE_RATE).astype(np.int16)
        wavfile.write(os.path.join(val_dir, class_dir, os.path.basename(file)), TARGET_SAMPLE_RATE, xs)

        val_count = val_count + 1
      else:
        os.makedirs(os.path.join(train_dir, class_dir), exist_ok=True)
        #os.rename(file, os.path.join(train_dir, class_dir, os.path.basename(file)))
        #shutil.copy(file, os.path.join(train_dir, class_dir, os.path.basename(file)))

        sample_rate, xs = wavfile.read(file)
        xs = xs.astype(np.float32)
        xs = librosa.resample(xs, sample_rate, TARGET_SAMPLE_RATE).astype(np.int16)
        wavfile.write(os.path.join(train_dir, class_dir, os.path.basename(file)), TARGET_SAMPLE_RATE, xs)

        train_count = train_count + 1

    test_count_total = test_count_total + test_count
    val_count_total = val_count_total + val_count
    train_count_total = train_count_total + train_count

    f = open(file_name, "a")
    print(dir + ' files: ' + str(len(files)))
    f.write(dir + ' files: ' + str(len(files)) + "\n")
    print('Train Files: '  + str(train_count))
    f.write('Train Files: '  + str(train_count) + "\n")
    print('Test Files: '  + str(test_count))
    f.write('Test Files: '  + str(test_count) + "\n")
    print('Validation Files: '  + str(val_count))
    f.write('Validation Files: '  + str(val_count) + "\n")
    f.close()

  f = open(file_name, "a")
  print("Total files = " + str(sample_count_total))
  f.write("Total files = " + str(sample_count_total) + "\n")
  print("Total train files = " + str(train_count_total))
  f.write("Total train files = " + str(train_count_total) + "\n")
  print("Total test files = " + str(test_count_total))
  f.write("Total test files = " + str(test_count_total) + "\n")
  print("Total validation files = " + str(val_count_total))
  f.write("Total validation files = " + str(val_count_total) + "\n")
  f.close()

# Code to generate a real random seed:
'''
seed_value = random.randrange(sys.maxsize)
print('Seed value:', seed_value)
'''

# We used a specific number as seed so we could have the same outcome through different runs:
# seed_value = 42
# random.seed(seed_value)

if awgn_backgorund_noise_percentage > 0:

  '''
  def show_sample(audio_path):
    audio_data, sample_rate = sf.read(audio_path)
    class_name = os.path.basename(os.path.dirname(audio_path))
    print(f'Class: {class_name}')
    print(f'File: {audio_path}')
    print(f'Sample rate: {sample_rate}')
    print(f'Sample length: {len(audio_data)}')

    plt.title(class_name)
    plt.plot(audio_data)
    display(Audio(audio_data, rate=sample_rate))
    plt.show()
  '''

  train_noise_dir = './dataset-train-noise'
  val_noise_dir = './dataset-val-noise'

  percentage = awgn_backgorund_noise_percentage

  rs = np.random.RandomState(seed_value)


  dirs = glob.glob(os.path.join(train_dir, '*/'))
  for dir in dirs:
      files = glob.glob(os.path.join(dir, '*.wav'))
      for file in files:
          audio_data, _ = librosa.load(file, sr=TARGET_SAMPLE_RATE)
          noise = rs.normal(0, audio_data.std(), audio_data.size) * percentage
          audio_data_noise = audio_data + noise
          audio_data_noise = audio_data_noise.transpose()

          class_dir = os.path.basename(os.path.normpath(dir))
          os.makedirs(os.path.join(train_noise_dir, class_dir), exist_ok=True)
          filename = os.path.join(train_noise_dir, class_dir, os.path.basename(file))
          soundfile.write(filename, audio_data_noise, TARGET_SAMPLE_RATE)

  random_filename = get_random_audio_file(train_noise_dir)
  show_sample(random_filename)


  dirs = glob.glob(os.path.join(val_dir, '*/'))
  for dir in dirs:
      files = glob.glob(os.path.join(dir, '*.wav'))
      for file in files:
          audio_data, _ = librosa.load(file, sr=TARGET_SAMPLE_RATE)
          noise = rs.normal(0, audio_data.std(), audio_data.size) * percentage
          audio_data_noise = audio_data + noise
          audio_data_noise = audio_data_noise.transpose()

          class_dir = os.path.basename(os.path.normpath(dir))
          os.makedirs(os.path.join(val_noise_dir, class_dir), exist_ok=True)
          filename = os.path.join(val_noise_dir, class_dir, os.path.basename(file))
          soundfile.write(filename, audio_data_noise, TARGET_SAMPLE_RATE)

  random_filename = get_random_audio_file(val_noise_dir)
  show_sample(random_filename)

  train_dir = train_noise_dir
  val_dir = val_noise_dir

# Divide train data per amount of nodes:

x = 0
y = 0
z = 0

# dataset_train_nodes = '/content/drive/MyDrive/runs/' + RUN_NAME + '/dataset_nodes/train'
cmd = "mkdir -p " + dataset_train_nodes
os.system(cmd)

# Loop through child directories (each class of wav files)
dirs = glob.glob(os.path.join(train_dir, '*/'))
for dir in dirs:
    files = glob.glob(os.path.join(dir, '*.wav'))
    random.shuffle(files)
    for file in files:
          class_dir = os.path.basename(os.path.normpath(dir))
          os.makedirs(os.path.join(dataset_train_nodes, str(x), class_dir), exist_ok=True)
          shutil.copy(file, os.path.join(dataset_train_nodes, str(x), class_dir))
          # os.rename(file, os.path.join(dataset_train_nodes, str(x), class_dir, os.path.basename(file)))
          if(x == nodes - 1):
            if(y < unbalance and (((x + z) % 2) == 0)):
                y = y + 1
            else:
                y = 0
                x = 0
          else:
            if(y < unbalance and (((x + z) % 2) == 0)):
                y = y + 1
            else:
                y = 0
                x = x + 1
    z = z + 1

# Plot how the train data is divided through the nodes:

dirs = glob.glob(os.path.join(dataset_train_nodes, '*/', '*/'))

total_samples = 0

f = open(run_information_folder + "/splitted_nodes.txt", "w")

for dir in dirs:
    files = glob.glob(os.path.join(dir, '*.wav'))
    sample_count = len(files)
    total_samples = total_samples + sample_count
    print(dir + ' files: ' + str(sample_count))
    f.write(dir + ' files: ' + str(sample_count) + "\n")

print('total files: ' + str(total_samples))
f.write('total files: ' + str(total_samples) + "\n")

# Plot how the train data is divided through the words:
print("-----//-----")
f.write("-----//-----" + "\n")

dirs = glob.glob(os.path.join(train_dir, '*/'))

for dir in dirs:
    files = glob.glob(os.path.join(dir, '*.wav'))
    sample_count = len(files)
    print(dir + ' files: ' + str(sample_count))
    f.write(dir + ' files: ' + str(sample_count) + "\n")

# Divide validation data per amount of nodes:

x = 0
y = 0
z = 0

# dataset_val_nodes = '/content/drive/MyDrive/runs/' + RUN_NAME + '/dataset_nodes/val'
cmd = "mkdir -p " + dataset_val_nodes
os.system(cmd)

# Loop through child directories (each class of wav files)
dirs = glob.glob(os.path.join(val_dir, '*/'))
for dir in dirs:
    files = glob.glob(os.path.join(dir, '*.wav'))
    random.shuffle(files)
    for file in files:
          class_dir = os.path.basename(os.path.normpath(dir))
          os.makedirs(os.path.join(dataset_val_nodes, str(x), class_dir), exist_ok=True)
          shutil.copy(file, os.path.join(dataset_val_nodes, str(x), class_dir))
          # os.rename(file, os.path.join(dataset_val_nodes, str(x), class_dir, os.path.basename(file)))
          if(x == nodes - 1):
            if(y < unbalance and (((x + z) % 2) == 0)):
                y = y + 1
            else:
                y = 0
                x = 0
          else:
            if(y < unbalance and (((x + z) % 2) == 0)):
                y = y + 1
            else:
                y = 0
                x = x + 1
    z = z + 1

# Plot how the validation data is divided through the nodes:

dirs = glob.glob(os.path.join(dataset_val_nodes, '*/', '*/'))

total_samples = 0

for dir in dirs:
    files = glob.glob(os.path.join(dir, '*.wav'))
    sample_count = len(files)
    total_samples = total_samples + sample_count
    print(dir + ' files: ' + str(sample_count))
    f.write(dir + ' files: ' + str(sample_count) + "\n")

print('total files: ' + str(total_samples))
f.write('total files: ' + str(total_samples) + "\n")

# Plot how the validation data is divided through the words:
print("-----//-----")
f.write("-----//-----" + "\n")

dirs = glob.glob(os.path.join(val_dir, '*/'))

for dir in dirs:
    files = glob.glob(os.path.join(dir, '*.wav'))
    sample_count = len(files)
    print(dir + ' files: ' + str(sample_count))
    f.write(dir + ' files: ' + str(sample_count) + "\n")

f.close()


"""
To validate the resistence of the system to poisoning attacks, we can add some background noise to the test and validation dataset of some nodes.
"""

if poisoned_nodes > 0:

    counter = 0
    
    percentage = poisoned_nodes_noise_percentage
    
    rs = np.random.RandomState(seed_value)
    
    for x in range(poisoned_nodes):
        dirs = glob.glob(os.path.join(dataset_train_nodes, str(x), '*/'))
        for dir in dirs:
            files = glob.glob(os.path.join(dir, '*.wav'))
            for file in files:
                audio_data, _ = librosa.load(file, sr=TARGET_SAMPLE_RATE)
                noise = rs.normal(0, audio_data.std(), audio_data.size) * percentage
                audio_data_noise = audio_data + noise
                audio_data_noise = audio_data_noise.transpose()

                class_dir = os.path.basename(os.path.normpath(dir))
                filename = os.path.join(dataset_train_nodes, str(x), class_dir, os.path.basename(file))
                os.remove(filename)
                soundfile.write(filename, audio_data_noise, TARGET_SAMPLE_RATE)
                counter = counter + 1


        dirs = glob.glob(os.path.join(dataset_val_nodes, str(x), '*/'))
        for dir in dirs:
            print("Enteres here 4")
            files = glob.glob(os.path.join(dir, '*.wav'))
            for file in files:
                audio_data, _ = librosa.load(file, sr=TARGET_SAMPLE_RATE)
                noise = rs.normal(0, audio_data.std(), audio_data.size) * percentage
                audio_data_noise = audio_data + noise
                audio_data_noise = audio_data_noise.transpose()

                class_dir = os.path.basename(os.path.normpath(dir))
                filename = os.path.join(dataset_val_nodes, str(x), class_dir, os.path.basename(file))
                os.remove(filename)
                soundfile.write(filename, audio_data_noise, TARGET_SAMPLE_RATE)
               
    print(str(counter))


"""After changing the filename and path name above, you're ready to train the model with your custom dataset. In the Colab toolbar, select **Runtime > Run all** to run the whole notebook.

The following code integrates our new background noise samples into your dataset and then separates a portion of all samples to create a test set.
"""

plot_data_amounts(train_dir, "Train")
plot_data_amounts(val_dir, "Validation")
plot_data_amounts(test_dir, "Test")

# Remove some words:
shutil.rmtree(os.path.join(train_dir, 'background'), ignore_errors=True)
shutil.rmtree(os.path.join(val_dir, 'background'), ignore_errors=True)
shutil.rmtree(os.path.join(test_dir, 'background'), ignore_errors=True)

plot_data_amounts_with_totals(train_dir, "Train")
plot_data_amounts_with_totals(val_dir, "Validation")
plot_data_amounts_with_totals(test_dir, "Test")

"""### Play a sample

To be sure the dataset looks correct, let's play at a random sample from the test set:
"""

random_audio = get_random_audio_file(train_dir)
show_sample(random_audio)

random_audio = get_random_audio_file(val_dir)
show_sample(random_audio)

random_audio = get_random_audio_file(test_dir)
show_sample(random_audio)

"""## Define the model - Tensorflow
"""
model_node = []
model_node_raw = []

for x in range(nodes):
    # Create folder to store SavedModels results in the "normal" way:
    tmp_folder = '/content/drive/MyDrive/runs/' + RUN_NAME + '/run_information/weights/normal/model_node_' + str(x) + '/'
    cmd = 'mkdir -p ' + tmp_folder
    os.system(cmd)
    file_name_before = tmp_folder + "initial_weights_before_setup.txt"
    f = open(file_name_before, "w")
    f.write('')
    f.close()
    file_name_after = tmp_folder + "initial_weights_after_setup.txt"
    f = open(file_name_after, "w")
    f.write('')
    f.close()

# Get the initial weights and layer initial names, so each node is initialized in the same way
# Load the Speech Commands model. Weights are loaded along with the topology,
# since we train the model from scratch. Instead, we will perform transfer
# learning based on the model.
raw_model = tfjs.converters.load_keras_model(tfjs_model_json_path, load_weights=load_weights, use_unique_name_scope=True)
for layer in raw_model.layers:
    layer._name = layer.name + str("_raw")

# Also remove the top Dense layer and add a new Dense layer of which the output
# size fits the number of sound classes we care about.
orig_model = tf.keras.Sequential(name="OriginalModel")

for layer in raw_model.layers[:-1]:
    orig_model.add(layer)

orig_model.add(tf.keras.layers.Dense(units=len(WORDS), activation="softmax", name="dense_words"))

for layer in orig_model.layers:
    layer._name = layer.name + str("_orig")

saved_weights = orig_model.get_weights()

for x in range(nodes):
    # Load the Speech Commands model. Weights are loaded along with the topology,
    # since we train the model from scratch. Instead, we will perform transfer
    # learning based on the model.
    #raw_model.append(tfjs.converters.load_keras_model(tfjs_model_json_path, load_weights=load_weights))
    #print(raw_model[x].get_weights())

    tmp_folder = '/content/drive/MyDrive/runs/' + RUN_NAME + '/run_information/weights/normal/model_node_' + str(x) + '/'
    file_name_before = tmp_folder + "initial_weights_before_setup.txt"
    file_name_after = tmp_folder + "initial_weights_after_setup.txt"

    model_node_raw.append(tfjs.converters.load_keras_model(tfjs_model_json_path, load_weights=load_weights, use_unique_name_scope=True))
    for layer in orig_model.layers:
      f = open(file_name_before, "a")
      f.write(str(layer.get_weights()))
      f.close()

    model_node.append(tf.keras.Sequential(name="ModelNode" + str(x)))

    for layer in model_node_raw[x].layers[:-1]:
      model_node[x].add(layer)
    model_node[x].add(tf.keras.layers.Dense(units=len(WORDS), activation="softmax", name="dense_words"))

    for layer in model_node[x].layers:
      layer._name = layer.name + str("_node_") + str(x)

    model_node[x].set_weights(saved_weights)

    for layer in model_node[x].layers:
      f = open(file_name_after, "a")
      f.write(str(layer.get_weights()))
      f.close()

    if transfer_learning_on_last_layer_only:
        # Freeze all but the last layer of the model. The last layer will be fine-tuned
        # during transfer learning.
        for layer in model_node[x].layers[:-1]:
          layer.trainable = False

    model_node[x].compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["acc"])
    model_node[x].summary()

# Transform the validation data into numpy arrays and store correspondent metadata:
dataset_train = []
dataset_val = []

f = open(run_information_folder + "/splitted_dataset.txt", "w")

for x in range(nodes):
    # Where the Speech Commands v0.02 dataset has been downloaded.
    DATA_ROOT_TRAIN = os.path.join(dataset_train_nodes, str(x))
    DATA_ROOT_VAL = os.path.join(dataset_val_nodes, str(x))

    input_wav_paths_and_labels_train = []
    input_wav_paths_and_labels_val = []

    for i, word in enumerate(WORDS):
      wav_paths_train = glob.glob(os.path.join(DATA_ROOT_TRAIN, word, "*.wav"))
      print("Train Node " + str(x) + ": Found %d examples for class %s" % (len(wav_paths_train), word))
      f.write("Train Node " + str(x) + ": Found %d examples for class %s" % (len(wav_paths_train), word) + "\n")
      labels_train = [i] * len(wav_paths_train)
      input_wav_paths_and_labels_train.extend(zip(wav_paths_train, labels_train))

      wav_paths_val = glob.glob(os.path.join(DATA_ROOT_VAL, word, "*.wav"))
      print("Validation Node " + str(x) + ": Found %d examples for class %s" % (len(wav_paths_val), word))
      f.write("Validation Node " + str(x) + ": Found %d examples for class %s" % (len(wav_paths_val), word) + "\n")
      labels_val = [i] * len(wav_paths_val)
      input_wav_paths_and_labels_val.extend(zip(wav_paths_val, labels_val))

    random.shuffle(input_wav_paths_and_labels_train)
    random.shuffle(input_wav_paths_and_labels_val)

    input_wav_paths_train, labels_train = ([t[0] for t in input_wav_paths_and_labels_train],
                              [t[1] for t in input_wav_paths_and_labels_train])
    input_wav_paths_val, labels_val = ([t[0] for t in input_wav_paths_and_labels_val],
                              [t[1] for t in input_wav_paths_and_labels_val])

    dataset_train.append(get_dataset(input_wav_paths_train, labels_train))
    dataset_val.append(get_dataset(input_wav_paths_val, labels_val))

f.close()

# Transform the test data into numpy arrays and store correspondent metadata:
input_wav_paths_and_labels_test = []

f = open(run_information_folder + "/splitted_dataset.txt", "a")

for i, word in enumerate(WORDS):
  wav_paths_test = glob.glob(os.path.join(DATA_ROOT_TEST, word, "*.wav"))
  print("Test: Found %d examples for class %s" % (len(wav_paths_test), word))
  f.write("Test: Found %d examples for class %s" % (len(wav_paths_test), word) + "\n")
  
  labels_test = [i] * len(wav_paths_test)
  input_wav_paths_and_labels_test.extend(zip(wav_paths_test, labels_test))

f.close()

random.shuffle(input_wav_paths_and_labels_test)

input_wav_paths_test, labels_test = ([t[0] for t in input_wav_paths_and_labels_test],
                           [t[1] for t in input_wav_paths_and_labels_test])

dataset_test = get_dataset(input_wav_paths_test, labels_test)

xs_and_ys_test = list(dataset_test)
xs_test = np.stack([item[0] for item in xs_and_ys_test])
ys_test = np.stack([item[1] for item in xs_and_ys_test])

# Create folder to store SavedModels in the "normal" way:
tf_nodes_savedmodel_folder = '/content/drive/MyDrive/runs/' + RUN_NAME + '/models/savedModel/normal/'
cmd = "mkdir -p " + tf_nodes_savedmodel_folder
os.system(cmd)

xs = []
ys = []
xs_val = []
ys_val = []
for x in range(nodes):
    # The amount of data we have is relatively small. It fits into typical host RAM
    # or GPU memory. For better training performance, we preload the data and
    # put it into numpy arrays:
    # - xs: The audio features (normalized spectrograms).
    # - ys: The labels (class indices).
    print(
        "Loading dataset and converting data to numpy arrays. "
        "This may take a few minutes...")
    xs_and_ys = list(dataset_train[x])
    xs.append(np.stack([item[0] for item in xs_and_ys]))
    ys.append(np.stack([item[1] for item in xs_and_ys]))

    xs_and_ys_val = list(dataset_val[x])
    xs_val.append(np.stack([item[0] for item in xs_and_ys_val]))
    ys_val.append(np.stack([item[1] for item in xs_and_ys_val]))
    print("Done.")

    # Train the model.
    model_node[x].fit(xs[x], ys[x], batch_size=batch_size, validation_data=(xs_val[x], ys_val[x]), shuffle=False, epochs=epochs)
    
    #Predict
    hot_y_prediction = model_node[x].predict(xs_test)
    y_prediction = np.argmax(hot_y_prediction, axis = 1)

    #y_test=np.argmax(ys_test, axis=1)
    #Create confusion matrix and normalizes it over predicted (columns)
    show_confusion_matrix(ys_test, y_prediction, hot_y_prediction, WORDS, x, "savedModel", "normal")
    
    model_node[x].save(tf_nodes_savedmodel_folder + "model_node_" + str(x))

# Commented out IPython magic to ensure Python compatibility.
# %%capture cap_savedmodel_normal --no-stderr

print('--------------------//--------------------')
print("Data to check if the models are being saved with different weights:")
for x in range(nodes):
    #Predict
    hot_y_prediction = model_node[x].predict(xs_test)
    y_prediction = np.argmax(hot_y_prediction, axis = 1)

    #y_test=np.argmax(ys_test, axis=1)
    #Create confusion matrix and normalizes it over predicted (columns)
    show_confusion_matrix(ys_test, y_prediction, hot_y_prediction, WORDS, x, "savedModel", "normal")
print('--------------------//--------------------')
    
if (nodes > 1):
    tf_results_savedmodel_folder_normal = '/content/drive/MyDrive/runs/' + RUN_NAME + '/results/savedModel/normal/global_model'
    cmd = "mkdir -p " + tf_results_savedmodel_folder_normal
    os.system(cmd)

    file_name = tf_results_savedmodel_folder_normal + "/evaluation_results.txt"
    valid_nodes = eval_all_nodes_models(model_node, file_name)

    f = open(file_name, "a")
    print('RESULTS:')
    f.write('RESULTS:\n')
    for x in range(nodes):
      if (valid_nodes[x] > (nodes / 2)):
        print('Model ' + str(x) + ' passed with ' + str(valid_nodes[x]) + ' votes')
        f.write('Model ' + str(x) + ' passed with ' + str(valid_nodes[x]) + ' votes\n')
      else:
        print('Model ' + str(x) + ' failed because only ontained ' + str(valid_nodes[x]) + ' votes')
        f.write('Model ' + str(x) + ' failed because only ontained ' + str(valid_nodes[x]) + ' votes\n')
    f.close()

if (nodes > 1):
    tf_results_savedmodel_folder_normal_all = '/content/drive/MyDrive/runs/' + RUN_NAME + '/results/savedModel/normal/global_model_all'
    cmd = "mkdir -p " + tf_results_savedmodel_folder_normal_all
    os.system(cmd)

    weights = []
    all_weights = []
    at_least_one_valid_model = False

    # It can be used to reconstruct the model identically.
    for x in range(nodes):

      SAVED_MODEL_FILENAME = tf_nodes_savedmodel_folder + "model_node_" + str(x)

      reconstructed_model = keras.models.load_model(SAVED_MODEL_FILENAME)
      reconstructed_model.summary()

      all_weights.append(reconstructed_model.get_weights())

      if (valid_nodes[x] > (nodes / 2)):
        at_least_one_valid_model = True

        weights.append(reconstructed_model.get_weights())

    new_weights = list()
    
    # Load the Speech Commands model. Weights are loaded along with the topology,
    # since we train the model from scratch. Instead, we will perform transfer
    # learning based on the model.
    raw_model = tfjs.converters.load_keras_model(tfjs_model_json_path, load_weights=load_weights)
    
    for layer in raw_model.layers:
      layer._name = layer.name + str("_globalmodel_all_savedmodel")

    # Also remove the top Dense layer and add a new Dense layer of which the output
    # size fits the number of sound classes we care about.
    global_model = tf.keras.Sequential(name="Globalmodel_all_savedmodel")

    for layer in raw_model.layers[:-1]:
      global_model.add(layer)
    global_model.add(tf.keras.layers.Dense(units=len(WORDS), activation="softmax", name="dense_words"))
    
    for layer in global_model.layers:
      layer._name = layer.name + str("_all_final")
    
    global_model.get_weights()

    for weights_list_tuple in zip(*all_weights):
      new_weights.append(
        np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
      )

    global_model.set_weights(new_weights)

    global_model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["acc"])

    global_model.save(tf_nodes_savedmodel_folder + "global_model_all")

    hot_y_prediction = global_model.predict(xs_test)
    y_prediction = np.argmax (hot_y_prediction, axis = 1)
    show_confusion_matrix(ys_test, y_prediction, hot_y_prediction, WORDS, -2, "savedModel", "normal")

    if at_least_one_valid_model == True:
      new_weights = list()
      
      # Load the Speech Commands model. Weights are loaded along with the topology,
      # since we train the model from scratch. Instead, we will perform transfer
      # learning based on the model.
      raw_model = tfjs.converters.load_keras_model(tfjs_model_json_path, load_weights=load_weights)

      for layer in raw_model.layers:
        layer._name = layer.name + str("_globalmodel_savedmodel")

      # Also remove the top Dense layer and add a new Dense layer of which the output
      # size fits the number of sound classes we care about.
      global_model = tf.keras.Sequential(name="Globalmodel_savedmodel")

      for layer in raw_model.layers[:-1]:
        global_model.add(layer)
      global_model.add(tf.keras.layers.Dense(units=len(WORDS), activation="softmax", name="dense_words"))
    
      for layer in global_model.layers:
        layer._name = layer.name + str("_final")
      
      global_model.get_weights()

      for weights_list_tuple in zip(*weights):
        new_weights.append(
          np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
        )

      global_model.set_weights(new_weights)

      global_model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["acc"])

      global_model.save(tf_nodes_savedmodel_folder + "global_model")

      hot_y_prediction = global_model.predict(xs_test)
      y_prediction = np.argmax (hot_y_prediction, axis = 1)
      show_confusion_matrix(ys_test, y_prediction, hot_y_prediction, WORDS, -1, "savedModel", "normal")
    else:
      print("No valid Models to show")

if use_prunning:
    tf_nodes_savedmodel_folder_prunning = '/content/drive/MyDrive/runs/' + RUN_NAME + '/models/savedModel/prunning/'
    cmd = 'mkdir -p ' + tf_nodes_savedmodel_folder_prunning
    os.system(cmd)

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 2 epochs.
    batch_size = 256
    epochs = 2

    num_files = len(xs)
    end_step = np.ceil(num_files / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
          'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                  final_sparsity=0.80,
                                                                  begin_step=0,
                                                                  end_step=end_step)
    }

    model_for_pruning = []
    model_node_copy = model_node

    for x in range(nodes):

        model_for_pruning.append(prune_low_magnitude(model_node_copy[x], **pruning_params))

        # `prune_low_magnitude` requires a recompile.
        model_for_pruning[x].compile(optimizer='sgd',
                      loss="sparse_categorical_crossentropy",
                      metrics=['acc'])

        model_for_pruning[x].summary()

        # Train model with pruning.
        logdir = tempfile.mkdtemp()

        callbacks = [
          tfmot.sparsity.keras.UpdatePruningStep(),
          tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
        ]

        model_for_pruning[x].fit(xs[x], ys[x],
                          batch_size=batch_size, epochs=epochs, validation_data=(xs_val[x], ys_val[x]),
                          shuffle=False, callbacks=callbacks)

        model_for_pruning[x].save(tf_nodes_savedmodel_folder_prunning + "model_node_" + str(x))

        # Predict model with prunning.
        hot_y_prediction = model_for_pruning[x].predict(xs_test)
        y_prediction = np.argmax (hot_y_prediction, axis = 1)
        #y_test=np.argmax(ys_test, axis=1)

        # Create confusion matrix and normalizes it over predicted (columns).
        show_confusion_matrix(ys_test, y_prediction, hot_y_prediction, WORDS, x, "savedModel", "prunning")

# Commented out IPython magic to ensure Python compatibility.
# %%capture cap_savedmodel_prunning --no-stderr

if nodes > 1 and use_prunning:
    tf_results_savedmodel_folder_prunning = '/content/drive/MyDrive/runs/' + RUN_NAME + '/results/savedModel/prunning/global_model'
    cmd = "mkdir -p " + tf_results_savedmodel_folder_prunning
    os.system(cmd)

    file_name = tf_results_savedmodel_folder_prunning + "/evaluation_results.txt"
    valid_nodes_prunning = eval_all_nodes_models(model_for_pruning, file_name)

    f = open(file_name, "a")
    print('RESULTS:')
    f.write('RESULTS:\n')
    for x in range(nodes):
      if (valid_nodes_prunning[x] > (nodes / 2)):
        print('Model ' + str(x) + ' passed with ' + str(valid_nodes_prunning[x]) + ' votes')
        f.write('Model ' + str(x) + ' passed with ' + str(valid_nodes_prunning[x]) + ' votes\n')
      else:
        print('Model ' + str(x) + ' failed because only ontained ' + str(valid_nodes_prunning[x]) + ' votes')
        f.write('Model ' + str(x) + ' failed because only ontained ' + str(valid_nodes_prunning[x]) + ' votes\n')
    f.close()

if nodes > 1 and use_prunning:
    tf_results_savedmodel_folder_prunning_all = '/content/drive/MyDrive/runs/' + RUN_NAME + '/results/savedModel/prunning/global_model_all'
    cmd = "mkdir -p " + tf_results_savedmodel_folder_prunning_all
    os.system(cmd)

    weights = []
    all_weights = []
    at_least_one_valid_model = False

    # It can be used to reconstruct the model identically.
    for x in range(nodes):
      tmp_folder = '/content/drive/MyDrive/runs/' + RUN_NAME + '/run_information/weights/prunning/model_node_' + str(x) + '/'
      cmd = "mkdir -p " + tmp_folder
      os.system(cmd)
    
      SAVED_MODEL_FILENAME = tf_nodes_savedmodel_folder_prunning + "model_node_" + str(x)

      reconstructed_model = keras.models.load_model(SAVED_MODEL_FILENAME)
      reconstructed_model.summary()
      
      tmp_file_name = tmp_folder + 'prunning_node_' + str(x)
      f = open(tmp_file_name, "a")
      f.write(str(reconstructed_model.get_weights()))
      f.close()

      all_weights.append(reconstructed_model.get_weights())

      if (valid_nodes_prunning[x] > (nodes / 2)):
        at_least_one_valid_model = True

        weights.append(reconstructed_model.get_weights())

    new_weights = list()
      
    # Load the Speech Commands model. Weights are loaded along with the topology,
    # since we train the model from scratch. Instead, we will perform transfer
    # learning based on the model.
    #SAVED_MODEL_FILENAME = tf_nodes_savedmodel_folder_prunning + "model_node_" + str(0)
    #raw_model_prunning = keras.models.load_model(SAVED_MODEL_FILENAME)
    
    for layer in reconstructed_model.layers:
      layer._name = layer.name + str("_globalmodel_all_savedmodel_prunning")

    # Also remove the top Dense layer and add a new Dense layer of which the output
    # size fits the number of sound classes we care about.
    global_model = tf.keras.Sequential(name="Globalmodel_all_savedmodel_prunning")

    for layer in reconstructed_model.layers:
      global_model.add(layer)
    
    for layer in global_model.layers:
      layer._name = layer.name + str("_all_final")
    
    global_model.summary()

    for weights_list_tuple in zip(*all_weights):
        size = np.asarray(weights_list_tuple).shape
        if len(size) == 1:
          new_weights.append(
              np.array(weights_list_tuple).mean(axis=0)
          )
        else:
          new_weights.append(
              np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
          )

    global_model.set_weights(new_weights)

    global_model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["acc"])

    global_model.save(tf_nodes_savedmodel_folder_prunning + "global_model_all")

    hot_y_prediction = global_model.predict(xs_test)
    y_prediction = np.argmax (hot_y_prediction, axis = 1)
    show_confusion_matrix(ys_test, y_prediction, hot_y_prediction, WORDS, -2, "savedModel", "prunning")

    if at_least_one_valid_model == True:
      new_weights = list()
        
      # Load the Speech Commands model. Weights are loaded along with the topology,
      # since we train the model from scratch. Instead, we will perform transfer
      # learning based on the model.
      raw_model_prunning = tfjs.converters.load_keras_model(tfjs_model_json_path, load_weights=load_weights)

      for layer in raw_model_prunning.layers:
        layer._name = layer.name + str("_globalmodel_savedmodel_prunning")

      # Also remove the top Dense layer and add a new Dense layer of which the output
      # size fits the number of sound classes we care about.
      global_model = tf.keras.Sequential(name="Globalmodel_savedmodel_prunning")

      for layer in reconstructed_model.layers:
        global_model.add(layer)
    
      for layer in global_model.layers:
        layer._name = layer.name + str("_final")  
      
      global_model.get_weights()

      for weights_list_tuple in zip(*weights):
          size = np.asarray(weights_list_tuple).shape
          if len(size) == 1:
            new_weights.append(
                np.array(weights_list_tuple).mean(axis=0)
            )
          else:
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )

      global_model.set_weights(new_weights)

      global_model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["acc"])

      global_model.save(tf_nodes_savedmodel_folder_prunning + "global_model")

      hot_y_prediction = global_model.predict(xs_test)
      y_prediction = np.argmax (hot_y_prediction, axis = 1)
      show_confusion_matrix(ys_test, y_prediction, hot_y_prediction, WORDS, -1, "savedModel", "prunning")
    else:
      print("No valid Models to show")

if False and (use_quantization):
    tf_nodes_savedmodel_folder_quantization = '/content/drive/MyDrive/runs/' + RUN_NAME + '/models/savedModel/quantization/'
    cmd = 'mkdir -p ' + tf_nodes_savedmodel_folder_quantization
    os.system(cmd)

    quantize_model = tfmot.quantization.keras.quantize_model

    q_aware_model = []

    for x in range(nodes):

      # q_aware stands for for quantization aware.
      q_aware_model.append(quantize_model(model_node[x]))

      # `quantize_model` requires a recompile.
      q_aware_model[x].compile(optimizer='sgd',
                    loss="sparse_categorical_crossentropy",
                    metrics=['acc'])

      q_aware_model[x].summary()

      q_aware_model[x].save(tf_nodes_savedmodel_folder_quantization + "model_node_" + str(x))

      # Predict model with prunning.
      hot_y_prediction = q_aware_model[x].predict(xs_test)
      y_prediction = np.argmax (hot_y_prediction, axis = 1)
      #y_test=np.argmax(ys_test, axis=1)

      # Create confusion matrix and normalizes it over predicted (columns).
      show_confusion_matrix(ys_test, y_prediction, hot_y_prediction, WORDS, x, "savedModel", "quantization")

# Commented out IPython magic to ensure Python compatibility.
# %%capture cap_savedmodel_quantization --no-stderr

if False and (nodes > 1 and use_quantization):
    tf_results_savedmodel_folder_quantization = '/content/drive/MyDrive/runs/' + RUN_NAME + '/results/savedModel/quantization/global_model'
    cmd = "mkdir -p " + tf_results_savedmodel_folder_quantization
    os.system(cmd)

    file_name = tf_results_savedmodel_folder_quantization + "/evaluation_results.txt"
    valid_nodes_quantization = eval_all_nodes_models(q_aware_model, file_name)

    f = open(folder, "a")
    print('RESULTS:')
    f.write('RESULTS:\n')
    for x in range(nodes):
      if (valid_nodes_quantization[x] > (nodes / 2)):
        print('Model ' + str(x) + ' passed with ' + str(valid_nodes_quantization[x]) + ' votes')
        f.write('Model ' + str(x) + ' passed with ' + str(valid_nodes_quantization[x]) + ' votes\n')
      else:
        print('Model ' + str(x) + ' failed because only ontained ' + str(valid_nodes_quantization[x]) + ' votes')
        f.write('Model ' + str(x) + ' failed because only ontained ' + str(valid_nodes_quantization[x]) + ' votes\n')
    f.close()

if False and (nodes > 1 and use_quantization):
    tf_results_savedmodel_folder_quantization_all = '/content/drive/MyDrive/runs/' + RUN_NAME + '/results/savedModel/quantization/global_model_all'
    cmd = "mkdir -p " + tf_results_savedmodel_folder_quantization_all
    os.system(cmd)

    weights = []
    all_weights = []
    at_least_one_valid_model = False

    # It can be used to reconstruct the model identically.
    for x in range(nodes):

      SAVED_MODEL_FILENAME = tf_nodes_savedmodel_folder_quantization + "model_node_" + str(x)

      reconstructed_model = keras.models.load_model(SAVED_MODEL_FILENAME)
      reconstructed_model.summary()

      all_weights.append(reconstructed_model.get_weights())

      if (valid_nodes_quantization[x] > (nodes / 2)):
        at_least_one_valid_model = True

        weights.append(reconstructed_model.get_weights())


    new_weights = list()
    SAVED_MODEL_FILENAME = tf_nodes_savedmodel_folder_quantization + "model_node_" + str(0)
    global_model = keras.models.load_model(SAVED_MODEL_FILENAME)
    global_model.get_weights()

    for weights_list_tuple in zip(*all_weights):
        size = np.asarray(weights_list_tuple).shape
        if len(size) == 1:
          new_weights.append(
              np.array(weights_list_tuple).mean(axis=0)
          )
        else:
          new_weights.append(
              np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
          )

    global_model.set_weights(new_weights)

    global_model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["acc"])

    global_model.save(tf_nodes_savedmodel_folder_quantization + "global_model_all")

    hot_y_prediction = global_model.predict(xs_test)
    y_prediction = np.argmax (hot_y_prediction, axis = 1)
    show_confusion_matrix(ys_test, y_prediction, hot_y_prediction, WORDS, -2, "savedModel", "quantization")

    if at_least_one_valid_model == True:
      new_weights = list()
      SAVED_MODEL_FILENAME = tf_nodes_savedmodel_folder_quantization + "model_node_" + str(0)
      global_model = keras.models.load_model(SAVED_MODEL_FILENAME)
      global_model.get_weights()

      for weights_list_tuple in zip(*weights):
          size = np.asarray(weights_list_tuple).shape
          if len(size) == 1:
            new_weights.append(
                np.array(weights_list_tuple).mean(axis=0)
            )
          else:
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )

      global_model.set_weights(new_weights)

      global_model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["acc"])

      global_model.save(tf_nodes_savedmodel_folder_quantization + "global_model")

      hot_y_prediction = global_model.predict(xs_test)
      y_prediction = np.argmax (hot_y_prediction, axis = 1)
      show_confusion_matrix(ys_test, y_prediction, hot_y_prediction, WORDS, -1, "savedModel", "quantization")
    else:
      print("No valid Models to show")

      

"""## Define the model - Tensorflow Lite

> Indented block


"""

# Convert the models to TFLite.

# We need to combine the preprocessing models and the newly trained 3-class model
# so that the resultant models will be able to preform STFT and spectrogram
# calculation on mobile devices (i.e., without web browser's WebAudio).

"""
combined_model = []
for x in range(nodes):
    combined_model.append(tf.keras.Sequential(name='CombinedModel' + str(x)))
    #combined_model[x].add(preproc_model)
    combined_model[x].add(model_node[x])
    combined_model[x].build([None, EXPECTED_WAVEFORM_LEN])
    combined_model[x].summary()

    output_path = '/content/drive/MyDrive/runs/' + RUN_NAME + '/models/tflite_combined/normal'
    os.makedirs(output_path, exist_ok=True)
    tflite_output_path = output_path + '/combined_model_' + str(x) + '.tflite'
    converter = tf.lite.TFLiteConverter.from_keras_model(combined_model[x])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
    ]
    with open(tflite_output_path, 'wb') as f:
        f.write(converter.convert())
    print("Saved tflite file at: %s" % tflite_output_path)
"""

tflite_model = []
for x in range(nodes):
    tflite_model.append(tf.keras.Sequential(name='tfliteModel' + str(x)))
    #tflite_model[x].add(preproc_model)
    tflite_model[x].add(model_node[x])
    tflite_model[x].build([None, EXPECTED_WAVEFORM_LEN])
    tflite_model[x].summary()

    output_path = '/content/drive/MyDrive/runs/' + RUN_NAME + '/models/tflite/normal'
    os.makedirs(output_path, exist_ok=True)
    tflite_output_path = output_path + '/model_node_' + str(x) + '.tflite'
    converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model[x])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
    ]
    with open(tflite_output_path, 'wb') as f:
        f.write(converter.convert())
    print("Saved tflite file at: %s" % tflite_output_path)
    
if use_quantization:
    # Convert the models to TFLite.

    # We need to combine the preprocessing models and the newly trained 3-class model
    # so that the resultant models will be able to preform STFT and spectrogram
    # calculation on mobile devices (i.e., without web browser's WebAudio).

    combined_model_quantization = []
    for x in range(nodes):
        combined_model_quantization.append(tf.keras.Sequential(name='CombinedModelQuantization' + str(x)))
        #combined_model_quantization[x].add(preproc_model)
        combined_model_quantization[x].add(model_node[x])
        combined_model_quantization[x].build([None, EXPECTED_WAVEFORM_LEN])
        combined_model_quantization[x].summary()

        output_path = '/content/drive/MyDrive/runs/' + RUN_NAME + '/models/tflite/quantization'
        os.makedirs(output_path, exist_ok=True)
        tflite_output_path = output_path + '/model_node_' + str(x) + '.tflite'
        converter = tf.lite.TFLiteConverter.from_keras_model(combined_model_quantization[x])
        converter.target_spec.supported_ops = [
	    tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        with open(tflite_output_path, 'wb') as f:
            f.write(converter.convert())
        print("Saved tflite file at: %s" % tflite_output_path)
