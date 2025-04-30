from configuration_and_functions import *

input_wav_paths_and_labels_test = []

for i, word in enumerate(WORDS):
  wav_paths_test = glob.glob(os.path.join(DATA_ROOT_TEST, word, "*.wav"))
  print("Test: Found %d examples for class %s" % (len(wav_paths_test), word))
  labels_test = [i] * len(wav_paths_test)
  input_wav_paths_and_labels_test.extend(zip(wav_paths_test, labels_test))

random.shuffle(input_wav_paths_and_labels_test)

input_wav_paths_test, labels_test = ([t[0] for t in input_wav_paths_and_labels_test],
                           [t[1] for t in input_wav_paths_and_labels_test])

dataset_test = get_dataset(input_wav_paths_test, labels_test)

xs_and_ys_test = list(dataset_test)
xs_test = np.stack([item[0] for item in xs_and_ys_test])
ys_test = np.stack([item[1] for item in xs_and_ys_test])

#audio_data, _ = librosa.load(xs_test, sr=TARGET_SAMPLE_RATE)
#simple_prediction = model.predict(xs_test)
#print(simple_prediction)

for x in range(nodes):
    #Predict
    interpreter = tf.lite.Interpreter(
                model_path='/content/drive/MyDrive/runs/' + RUN_NAME + '/models/tflite/normal/model_node_' + str(x) + '.tflite',
                experimental_preserve_all_tensors=True
            )
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Run inference
    interpreter.allocate_tensors()

    all_output_data = []
    for y in range(len(xs_test)):
      interpreter.set_tensor(input_details[0]['index'], [xs_test[y]])
      interpreter.invoke()
      output_data = interpreter.get_tensor(output_details[0]['index'])
      all_output_data.append(output_data)

    y_prediction = np.argmax(all_output_data, axis = 2)
    #y_test=np.argmax(ys_test, axis=1)
    #Create confusion matrix and normalizes it over predicted (columns)
    show_confusion_matrix(ys_test, y_prediction, all_output_data, WORDS, x, "tflite", "normal")


# Commented out IPython magic to ensure Python compatibility.
# %%capture cap_tflite_normal --no-stderr

valid_nodes_tflite_count = 0

if (nodes > 1):
    tf_results_tflite_folder_normal = '/content/drive/MyDrive/runs/' + RUN_NAME + '/results/tflite/normal/global_model'
    cmd = "mkdir -p " + tf_results_tflite_folder_normal
    os.system(cmd)
    file_name = tf_results_tflite_folder_normal + "/evaluation_results.txt"

    valid_nodes_tflite = eval_all_nodes_models_tflite('normal', file_name)

    f = open(file_name, "a")
    print('RESULTS:')
    f.write('RESULTS:\n')
    f.close()
    
    for x in range(nodes):
      if (valid_nodes_tflite[x] > (nodes / 2)):
        f = open(file_name, "a")
        print('Model ' + str(x) + ' passed with ' + str(valid_nodes_tflite[x]) + ' votes')
        f.write('Model ' + str(x) + ' passed with ' + str(valid_nodes_tflite[x]) + ' votes\n')
        f.close()
        valid_nodes_tflite_count = valid_nodes_tflite_count + 1
      else:
        f = open(file_name, "a")
        print('Model ' + str(x) + ' failed because only ontained ' + str(valid_nodes_tflite[x]) + ' votes')
        f.write('Model ' + str(x) + ' failed because only ontained ' + str(valid_nodes_tflite[x]) + ' votes\n')
        f.close()

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# wget https://github.com/ricardodeazambuja/flatbuffers/releases/download/v2.0.1a/flatc.zip
# unzip flatc.zip
# chmod +x flatc
# sudo mv flatc /usr/local/bin/
# flatc --version

# Commented out IPython magic to ensure Python compatibility.
os.system("wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs")

# Convert tflite to json
# %env schema=schema.fbs
os.environ['schema'] = '/content/drive/MyDrive/dependencies/schema.fbs'

if nodes > 1:
  tf_results_tflite_folder_normal_all = '/content/drive/MyDrive/runs/' + RUN_NAME + '/results/tflite/normal/global_model_all'
  cmd = "mkdir -p " + tf_results_tflite_folder_normal_all
  os.system(cmd)

  #Choose if you want to limit print configurations or not (optional)
  #np.set_printoptions(threshold=1000)
  #np.set_printoptions(threshold=np.inf)

  for x in range(nodes):
      # If schema version is smaller than v3, flatc needs to use --raw-binary
      output_path = '/content/drive/MyDrive/runs/' + RUN_NAME + '/models/tflite/normal/'
      os.environ['output_path'] = output_path
      tflite_filename = output_path + 'model_node_' + str(x) + '.tflite'
      os.environ['tflite_filename'] = tflite_filename
      os.system("flatc -t -o ${output_path} --strict-json --defaults-json ${schema} -- ${tflite_filename} # type: ignore")

      # Get and open json file
      json_filename = output_path + 'model_node_' + str(x) + '.json'
      with open(json_filename) as f:
        model_json = json.load(f)

      temp_numpy_2d_arrays = []

      # If first iteration, only open data and fill the first array
      if x == 0:
          numpy_2d_arrays_final = []
          for buffer in model_json['buffers']:
              if "data" in buffer:
                  numpy_2d_arrays_final.append(buffer['data'])
              else:
                  numpy_2d_arrays_final.append(np.array([]))

          y = 0
          temp_weight_final = []
          for buffer in model_json['buffers']:
              temp_weight_final_row = []
              if "data" in buffer:
                  numpy_2d_arrays_final[y] = np.array(numpy_2d_arrays_final[y])
                  temp_numpy_2d_arrays = list(partition(4,numpy_2d_arrays_final[y]))

                  for temp_list in range(len(temp_numpy_2d_arrays)):
                      temp_numpy_2d_arrays[temp_list] = list(temp_numpy_2d_arrays[temp_list])
                      temp_numpy_2d_arrays[temp_list] = bytearray(temp_numpy_2d_arrays[temp_list])

                      if y < 26 and model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT32':
                          temp_weight = struct.unpack('<l', temp_numpy_2d_arrays[temp_list])
                          temp_weight_final_row.append(temp_weight[0])
                      elif y < 26 and model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'FLOAT32':
                          temp_weight = struct.unpack('<f', temp_numpy_2d_arrays[temp_list])
                          temp_weight_final_row.append(temp_weight[0])
                      elif y < 26 and model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'COMPLEX64':
                          input("Complex Number, needs to be implemented!")
                      else:
                          temp_weight_final_row.append(0)


              temp_weight_final.append(temp_weight_final_row)
              y = y + 1

      # If not first iteration, open data and sum information to already existing one
      # Only do this if arrays are different from the already exists
      else:
          numpy_2d_arrays = []
          for buffer in model_json['buffers']:
              if "data" in buffer:
                  numpy_2d_arrays.append(buffer['data'])
              else:
                  numpy_2d_arrays.append(np.array([]))

          y = 0
          for buffer in model_json['buffers']:
              if "data" in buffer:
                  numpy_2d_arrays[y] = np.array(numpy_2d_arrays[y])
                  if not np.array_equal(numpy_2d_arrays[y],  numpy_2d_arrays_final[y]):
                      print("CHANGEABLE BUFFER! NODE: " + str(x) + ", BUFFER NUMBER: " + str(y))
                      temp_numpy_2d_arrays = list(partition(4,numpy_2d_arrays[y]))

                      for temp_list in range(len(temp_numpy_2d_arrays)):

                          temp_numpy_2d_arrays[temp_list] = list(temp_numpy_2d_arrays[temp_list])
                          temp_numpy_2d_arrays[temp_list] = bytearray(temp_numpy_2d_arrays[temp_list])

                          if model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT32':
                              temp_weight = struct.unpack('<l', temp_numpy_2d_arrays[temp_list])
                          elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'FLOAT32':
                              temp_weight = struct.unpack('<f', temp_numpy_2d_arrays[temp_list])
                          elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'COMPLEX64':
                              input("Complex Number, needs to be implemented!")
                          else:
                              input("Subgraph type not recognized!")

                          temp_weight_final[y][temp_list] = temp_weight_final[y][temp_list] + temp_weight[0]
              y = y + 1

  # Divide each
  y = 0
  j = 0
  for buffer in model_json['buffers']:
      if "data" in buffer:
          print("Y: " + str(y))
          temp_array = []
          if not np.array_equal(numpy_2d_arrays[y],  numpy_2d_arrays_final[y]):
              print("BUFFER NUMBER: " + str(y))
              for j in range(len(temp_weight_final[y])):
                  temp_weight_final[y][j] = temp_weight_final[y][j] / valid_nodes_tflite_count

                  if model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT32':
                      temp_weight_final[y][j] = np.int32(temp_weight_final[y][j])
                      temp_bytearray = bytearray(struct.pack("<l", temp_weight_final[y][j]))
                      for i in temp_bytearray:
                          temp_array.append(i)
                  elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'FLOAT32':
                      temp_weight_final[y][j] = np.float32(temp_weight_final[y][j])
                      temp_bytearray = bytearray(struct.pack("<f", temp_weight_final[y][j]))
                      for i in temp_bytearray:
                          temp_array.append(i)
                  elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'COMPLEX64':
                      input("Complex Number, needs to be implemented!")
                  else:
                      input("Subgraph type not recognized!")

              buffer['data'] = temp_array
      y = y + 1

  global_model_all_path_json = output_path + 'global_model_all.json'
  os.environ['global_model_all_path_json'] = global_model_all_path_json
  with open(global_model_all_path_json, 'w') as f:
      json.dump(model_json, f)

  os.system("flatc -b --strict-json --defaults-json -o ${output_path} ${schema} ${global_model_all_path_json}")

if nodes > 1:

    global_model_path_tflite = output_path + 'global_model_all.tflite'

    #Predict
    interpreter = tf.lite.Interpreter(
                model_path = global_model_path_tflite,
                experimental_preserve_all_tensors = True
            )
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Run inference
    interpreter.allocate_tensors()

    all_output_data = []
    for y in range(len(xs_test)):
      interpreter.set_tensor(input_details[0]['index'], [xs_test[y]])
      interpreter.invoke()
      output_data = interpreter.get_tensor(output_details[0]['index'])
      all_output_data.append(output_data)

    y_prediction = np.argmax(all_output_data, axis = 2)
    #y_test=np.argmax(ys_test, axis=1)
    #Create confusion matrix and normalizes it over predicted (columns)
    show_confusion_matrix(ys_test, y_prediction, all_output_data, WORDS, -2, "tflite", "normal")

if nodes > 1 and valid_nodes_tflite_count > 0:
  #Choose if you want to limit print configurations or not (optional)
  #np.set_printoptions(threshold=1000)
  #np.set_printoptions(threshold=np.inf)
  
  first_iteration = True

  for x in range(nodes):
    if (valid_nodes_tflite[x] > (nodes / 2)):
      # If schema version is smaller than v3, flatc needs to use --raw-binary
      output_path = '/content/drive/MyDrive/runs/' + RUN_NAME + '/models/tflite/normal/'
      os.environ['output_path'] = output_path
      tflite_filename = output_path + 'model_node_' + str(x) + '.tflite'
      os.environ['tflite_filename'] = tflite_filename
      os.system("flatc -t -o ${output_path} --strict-json --defaults-json ${schema} -- ${tflite_filename} # type: ignore")

      # Get and open json file
      json_filename = output_path + 'model_node_' + str(x) + '.json'
      with open(json_filename) as f:
        model_json = json.load(f)

      temp_numpy_2d_arrays = []

      # If first iteration, only open data and fill the first array
      if first_iteration:
          first_iteration = False
          numpy_2d_arrays_final = []
          for buffer in model_json['buffers']:
              if "data" in buffer:
                  numpy_2d_arrays_final.append(buffer['data'])
              else:
                  numpy_2d_arrays_final.append(np.array([]))

          y = 0
          temp_weight_final = []
          for buffer in model_json['buffers']:
              temp_weight_final_row = []
              if "data" in buffer:
                  numpy_2d_arrays_final[y] = np.array(numpy_2d_arrays_final[y])
                  temp_numpy_2d_arrays = list(partition(4,numpy_2d_arrays_final[y]))

                  for temp_list in range(len(temp_numpy_2d_arrays)):
                      temp_numpy_2d_arrays[temp_list] = list(temp_numpy_2d_arrays[temp_list])
                      temp_numpy_2d_arrays[temp_list] = bytearray(temp_numpy_2d_arrays[temp_list])

                      if y < 26 and model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT32':
                          temp_weight = struct.unpack('<l', temp_numpy_2d_arrays[temp_list])
                          temp_weight_final_row.append(temp_weight[0])
                      elif y < 26 and model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'FLOAT32':
                          temp_weight = struct.unpack('<f', temp_numpy_2d_arrays[temp_list])
                          temp_weight_final_row.append(temp_weight[0])
                      elif y < 26 and model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'COMPLEX64':
                          input("Complex Number, needs to be implemented!")
                      else:
                          temp_weight_final_row.append(0)


              temp_weight_final.append(temp_weight_final_row)
              y = y + 1

      # If not first iteration, open data and sum information to already existing one
      # Only do this if arrays are different from the already exists
      else:
          numpy_2d_arrays = []
          for buffer in model_json['buffers']:
              if "data" in buffer:
                  numpy_2d_arrays.append(buffer['data'])
              else:
                  numpy_2d_arrays.append(np.array([]))

          y = 0
          for buffer in model_json['buffers']:
              if "data" in buffer:
                  numpy_2d_arrays[y] = np.array(numpy_2d_arrays[y])
                  if not np.array_equal(numpy_2d_arrays[y],  numpy_2d_arrays_final[y]):
                      print("CHANGEABLE BUFFER! NODE: " + str(x) + ", BUFFER NUMBER: " + str(y))
                      temp_numpy_2d_arrays = list(partition(4,numpy_2d_arrays[y]))

                      for temp_list in range(len(temp_numpy_2d_arrays)):

                          temp_numpy_2d_arrays[temp_list] = list(temp_numpy_2d_arrays[temp_list])
                          temp_numpy_2d_arrays[temp_list] = bytearray(temp_numpy_2d_arrays[temp_list])

                          if model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT32':
                              temp_weight = struct.unpack('<l', temp_numpy_2d_arrays[temp_list])
                          elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'FLOAT32':
                              temp_weight = struct.unpack('<f', temp_numpy_2d_arrays[temp_list])
                          elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'COMPLEX64':
                              input("Complex Number, needs to be implemented!")
                          else:
                              input("Subgraph type not recognized!")

                          temp_weight_final[y][temp_list] = temp_weight_final[y][temp_list] + temp_weight[0]
              y = y + 1

  # Divide each
  y = 0
  j = 0
  for buffer in model_json['buffers']:
      if "data" in buffer:
          print("Y: " + str(y))
          temp_array = []
          if not np.array_equal(numpy_2d_arrays[y],  numpy_2d_arrays_final[y]):
              print("BUFFER NUMBER: " + str(y))
              for j in range(len(temp_weight_final[y])):
                  temp_weight_final[y][j] = temp_weight_final[y][j] / valid_nodes_tflite_count

                  if model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT32':
                      temp_weight_final[y][j] = np.int32(temp_weight_final[y][j])
                      temp_bytearray = bytearray(struct.pack("<l", temp_weight_final[y][j]))
                      for i in temp_bytearray:
                          temp_array.append(i)
                  elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'FLOAT32':
                      temp_weight_final[y][j] = np.float32(temp_weight_final[y][j])
                      temp_bytearray = bytearray(struct.pack("<f", temp_weight_final[y][j]))
                      for i in temp_bytearray:
                          temp_array.append(i)
                  elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'COMPLEX64':
                      input("Complex Number, needs to be implemented!")
                  else:
                      input("Subgraph type not recognized!")

              buffer['data'] = temp_array
      y = y + 1

  global_model_path_json = output_path + 'global_model.json'
  os.environ['global_model_path_json'] = global_model_path_json
  with open(global_model_path_json, 'w') as f:
      json.dump(model_json, f)

  os.system("flatc -b --strict-json --defaults-json -o ${output_path} ${schema} ${global_model_path_json}")

if nodes > 1 and valid_nodes_tflite_count > 0:

    global_model_path_tflite = output_path + 'global_model.tflite'

    #Predict
    interpreter = tf.lite.Interpreter(
                model_path = global_model_path_tflite,
                experimental_preserve_all_tensors = True
            )
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Run inference
    interpreter.allocate_tensors()

    all_output_data = []
    for y in range(len(xs_test)):
      interpreter.set_tensor(input_details[0]['index'], [xs_test[y]])
      interpreter.invoke()
      output_data = interpreter.get_tensor(output_details[0]['index'])
      all_output_data.append(output_data)

    y_prediction = np.argmax(all_output_data, axis = 2)
    #y_test=np.argmax(ys_test, axis=1)
    #Create confusion matrix and normalizes it over predicted (columns)
    show_confusion_matrix(ys_test, y_prediction, all_output_data, WORDS, -1, "tflite", "normal")

"""
if use_prunning:
    # Convert the models to TFLite.

    # We need to combine the preprocessing models and the newly trained 3-class model
    # so that the resultant models will be able to preform STFT and spectrogram
    # calculation on mobile devices (i.e., without web browser's WebAudio).

    combined_model_prunning = []
    for x in range(nodes):
        #combined_model_prunning.append(tf.keras.Sequential(name='CombinedModelPrunning' + str(x)))
        #combined_model_prunning[x].add(preproc_model)
        #combined_model_prunning[x].add(model_for_pruning[x])
        #combined_model_prunning[x].build([None, EXPECTED_WAVEFORM_LEN])
        #combined_model_prunning[x].compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["acc"])
        #combined_model_prunning[x].summary()

        output_path = '/content/drive/MyDrive/runs/' + RUN_NAME + '/models/tflite/prunning'
        os.makedirs(output_path, exist_ok=True)
        tflite_output_path = output_path + '/combined_model_' + str(x) + '.tflite'
        converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning[x])
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.experimental_enable_resource_variables = True
        with open(tflite_output_path, 'wb') as f:
            f.write(converter.convert())
        print("Saved tflite file at: %s" % tflite_output_path)
"""

DATA_ROOT_TEST = os.path.join("/content/dataset-test")

input_wav_paths_and_labels_test = []

for i, word in enumerate(WORDS):
  wav_paths_test = glob.glob(os.path.join(DATA_ROOT_TEST, word, "*.wav"))
  print("Test: Found %d examples for class %s" % (len(wav_paths_test), word))
  labels_test = [i] * len(wav_paths_test)
  input_wav_paths_and_labels_test.extend(zip(wav_paths_test, labels_test))

random.shuffle(input_wav_paths_and_labels_test)

input_wav_paths_test, labels_test = ([t[0] for t in input_wav_paths_and_labels_test],
                           [t[1] for t in input_wav_paths_and_labels_test])

dataset_test = get_dataset(input_wav_paths_test, labels_test)

xs_and_ys_test = list(dataset_test)
xs_test = np.stack([item[0] for item in xs_and_ys_test])
ys_test = np.stack([item[1] for item in xs_and_ys_test])

#audio_data, _ = librosa.load(xs_test, sr=TARGET_SAMPLE_RATE)
#simple_prediction = model.predict(xs_test)
#print(simple_prediction)

for x in range(nodes):
    #Predict
    interpreter = tf.lite.Interpreter(
                model_path='/content/drive/MyDrive/runs/' + RUN_NAME + '/models/tflite/quantization/model_node_' + str(x) + '.tflite',
                experimental_preserve_all_tensors=True
            )
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Run inference
    interpreter.allocate_tensors()

    all_output_data = []
    for y in range(len(xs_test)):
      interpreter.set_tensor(input_details[0]['index'], [xs_test[y]])
      interpreter.invoke()
      output_data = interpreter.get_tensor(output_details[0]['index'])
      all_output_data.append(output_data)

    y_prediction = np.argmax(all_output_data, axis = 2)
    #y_test=np.argmax(ys_test, axis=1)
    #Create confusion matrix and normalizes it over predicted (columns)
    show_confusion_matrix(ys_test, y_prediction, all_output_data, WORDS, x, "tflite", "quantization")

# Commented out IPython magic to ensure Python compatibility.
# %%capture cap_tflite_quantization --no-stderr

valid_nodes_tflite_quantization_count = 0

if (nodes > 1):
    tf_results_tflite_folder_quantization = '/content/drive/MyDrive/runs/' + RUN_NAME + '/results/tflite/quantization/global_model'
    cmd = "mkdir -p " + tf_results_tflite_folder_quantization
    os.system(cmd)
    file_name = tf_results_tflite_folder_quantization + "/evaluation_results.txt"

    valid_nodes_tflite_quantization = eval_all_nodes_models_tflite('quantization', file_name)

    f = open(file_name, "a")
    print('RESULTS:')
    f.write('RESULTS:\n')
    f.close()
    
    for x in range(nodes):
      if (valid_nodes_tflite_quantization[x] > (nodes / 2)):
        f = open(file_name, "a")
        print('Model ' + str(x) + ' passed with ' + str(valid_nodes_tflite_quantization[x]) + ' votes')
        f.write('Model ' + str(x) + ' passed with ' + str(valid_nodes_tflite_quantization[x]) + ' votes\n')
        f.close()
        valid_nodes_tflite_quantization_count = valid_nodes_tflite_quantization_count + 1
      else:
        f = open(file_name, "a")
        print('Model ' + str(x) + ' failed because only ontained ' + str(valid_nodes_tflite_quantization[x]) + ' votes')
        f.write('Model ' + str(x) + ' failed because only ontained ' + str(valid_nodes_tflite_quantization[x]) + ' votes\n')
        f.close()

if use_quantization and nodes > 1:
  tf_results_tflite_folder_quantization_all = '/content/drive/MyDrive/runs/' + RUN_NAME + '/results/tflite/quantization/global_model_all'
  cmd = "mkdir -p " + tf_results_tflite_folder_quantization_all
  os.system(cmd)

  #Choose if you want to limit print configurations or not (optional)
  #np.set_printoptions(threshold=1000)
  #np.set_printoptions(threshold=np.inf)

  for x in range(nodes):
      # If schema version is smaller than v3, flatc needs to use --raw-binary
      output_path = '/content/drive/MyDrive/runs/' + RUN_NAME + '/models/tflite/quantization/'
      os.environ['output_path'] = output_path
      tflite_filename = output_path + 'model_node_' + str(x) + '.tflite'
      os.environ['tflite_filename'] = tflite_filename
      os.system("flatc -t -o ${output_path} --strict-json --defaults-json ${schema} -- ${tflite_filename} # type: ignore")

      # Get and open json file
      json_filename = output_path + 'model_node_' + str(x) + '.json'
      with open(json_filename) as f:
        model_json = json.load(f)

      temp_numpy_2d_arrays = []

      # If first iteration, only open data and fill the first array
      if x == 0:
          numpy_2d_arrays_final = []
          for buffer in model_json['buffers']:
              if "data" in buffer:
                  numpy_2d_arrays_final.append(buffer['data'])
              else:
                  numpy_2d_arrays_final.append(np.array([]))

          y = 0
          temp_weight_final = []
          for buffer in model_json['buffers']:
              temp_weight_final_row = []
              if "data" in buffer:
                  numpy_2d_arrays_final[y] = np.array(numpy_2d_arrays_final[y])
                  if y < 26 and (model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT8'):
                      temp_int = 1
                  else:
                      temp_int = 4
                  temp_numpy_2d_arrays = list(partition(temp_int,numpy_2d_arrays_final[y]))

                  for temp_list in range(len(temp_numpy_2d_arrays)):
                      temp_numpy_2d_arrays[temp_list] = list(temp_numpy_2d_arrays[temp_list])
                      temp_numpy_2d_arrays[temp_list] = bytearray(temp_numpy_2d_arrays[temp_list])

                      if y < 26 and model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT8':
                          temp_weight = struct.unpack('<b', temp_numpy_2d_arrays[temp_list])
                          temp_weight_final_row.append(temp_weight[0])
                      elif y < 26 and model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT32':
                          temp_weight = struct.unpack('<l', temp_numpy_2d_arrays[temp_list])
                          temp_weight_final_row.append(temp_weight[0])
                      elif y < 26 and model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'FLOAT32':
                          temp_weight = struct.unpack('<f', temp_numpy_2d_arrays[temp_list])
                          temp_weight_final_row.append(temp_weight[0])
                      elif y < 26 and model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'COMPLEX64':
                          input("Complex Number, needs to be implemented!")
                      else:
                          temp_weight_final_row.append(0)


              temp_weight_final.append(temp_weight_final_row)
              y = y + 1

      # If not first iteration, open data and sum information to already existing one
      # Only do this if arrays are different from the already exists
      else:
          numpy_2d_arrays = []
          for buffer in model_json['buffers']:
              if "data" in buffer:
                  numpy_2d_arrays.append(buffer['data'])
              else:
                  numpy_2d_arrays.append(np.array([]))

          y = 0
          for buffer in model_json['buffers']:
              if "data" in buffer:
                  numpy_2d_arrays[y] = np.array(numpy_2d_arrays[y])
                  if not np.array_equal(numpy_2d_arrays[y],  numpy_2d_arrays_final[y]):
                      print("CHANGEABLE BUFFER! NODE: " + str(x) + ", BUFFER NUMBER: " + str(y))
                      if model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT8':
                          temp_int = 1
                      else:
                          temp_int = 4
                      temp_numpy_2d_arrays = list(partition(temp_int,numpy_2d_arrays[y]))

                      for temp_list in range(len(temp_numpy_2d_arrays)):

                          temp_numpy_2d_arrays[temp_list] = list(temp_numpy_2d_arrays[temp_list])
                          temp_numpy_2d_arrays[temp_list] = bytearray(temp_numpy_2d_arrays[temp_list])

                          if model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT8':
                              temp_weight = struct.unpack('<b', temp_numpy_2d_arrays[temp_list])
                          elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT32':
                              temp_weight = struct.unpack('<l', temp_numpy_2d_arrays[temp_list])
                          elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'FLOAT32':
                              temp_weight = struct.unpack('<f', temp_numpy_2d_arrays[temp_list])
                          elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'COMPLEX64':
                              input("Complex Number, needs to be implemented!")
                          else:
                              input("Subgraph type not recognized! Type: " + model_json['subgraphs'][0]['tensors'][y-1]['type'])

                          temp_weight_final[y][temp_list] = temp_weight_final[y][temp_list] + temp_weight[0]
              y = y + 1

  # Divide each
  y = 0
  j = 0
  for buffer in model_json['buffers']:
      if "data" in buffer:
          print("Y: " + str(y))
          temp_array = []
          if not np.array_equal(numpy_2d_arrays[y],  numpy_2d_arrays_final[y]):
              print("BUFFER NUMBER: " + str(y))
              for j in range(len(temp_weight_final[y])):
                  temp_weight_final[y][j] = temp_weight_final[y][j] / nodes

                  if model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT8':
                      temp_weight_final[y][j] = np.int8(temp_weight_final[y][j])
                      temp_bytearray = bytearray(struct.pack("<b", temp_weight_final[y][j]))
                      for i in temp_bytearray:
                          temp_array.append(i)
                  elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT32':
                      temp_weight_final[y][j] = np.int32(temp_weight_final[y][j])
                      temp_bytearray = bytearray(struct.pack("<l", temp_weight_final[y][j]))
                      for i in temp_bytearray:
                          temp_array.append(i)
                  elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'FLOAT32':
                      temp_weight_final[y][j] = np.float32(temp_weight_final[y][j])
                      temp_bytearray = bytearray(struct.pack("<f", temp_weight_final[y][j]))
                      for i in temp_bytearray:
                          temp_array.append(i)
                  elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'COMPLEX64':
                      input("Complex Number, needs to be implemented!")
                  else:
                      input("Subgraph type not recognized! Type: " + model_json['subgraphs'][0]['tensors'][y-1]['type'])

              buffer['data'] = temp_array
      y = y + 1

  global_model_all_path_json = output_path + 'global_model_all.json'
  os.environ['global_model_all_path_json'] = global_model_all_path_json
  with open(global_model_all_path_json, 'w') as f:
      json.dump(model_json, f)

  os.system("flatc -b --strict-json --defaults-json -o ${output_path} ${schema} ${global_model_all_path_json}")

if use_quantization and nodes > 1:

    global_model_path_tflite = output_path + 'global_model_all.tflite'

    #Predict
    interpreter = tf.lite.Interpreter(
                model_path = global_model_path_tflite,
                experimental_preserve_all_tensors = True
            )
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Run inference
    interpreter.allocate_tensors()

    all_output_data = []
    for y in range(len(xs_test)):
      interpreter.set_tensor(input_details[0]['index'], [xs_test[y]])
      interpreter.invoke()
      output_data = interpreter.get_tensor(output_details[0]['index'])
      all_output_data.append(output_data)

    y_prediction = np.argmax(all_output_data, axis = 2)
    #y_test=np.argmax(ys_test, axis=1)
    #Create confusion matrix and normalizes it over predicted (columns)
    show_confusion_matrix(ys_test, y_prediction, all_output_data, WORDS, -2, "tflite", "quantization")

if use_quantization and nodes > 1 and valid_nodes_tflite_quantization_count > 0:
  #Choose if you want to limit print configurations or not (optional)
  #np.set_printoptions(threshold=1000)
  #np.set_printoptions(threshold=np.inf)
  
  first_iteration = True

  for x in range(nodes):
    if (valid_nodes_tflite_quantization[x] > (nodes / 2)):
      # If schema version is smaller than v3, flatc needs to use --raw-binary
      output_path = '/content/drive/MyDrive/runs/' + RUN_NAME + '/models/tflite/quantization/'
      os.environ['output_path'] = output_path
      tflite_filename = output_path + 'model_node_' + str(x) + '.tflite'
      os.environ['tflite_filename'] = tflite_filename
      os.system("flatc -t -o ${output_path} --strict-json --defaults-json ${schema} -- ${tflite_filename} # type: ignore")

      # Get and open json file
      json_filename = output_path + 'model_node_' + str(x) + '.json'
      with open(json_filename) as f:
        model_json = json.load(f)

      temp_numpy_2d_arrays = []

      # If first iteration, only open data and fill the first array
      if first_iteration:
          first_iteration = False
          numpy_2d_arrays_final = []
          for buffer in model_json['buffers']:
              if "data" in buffer:
                  numpy_2d_arrays_final.append(buffer['data'])
              else:
                  numpy_2d_arrays_final.append(np.array([]))

          y = 0
          temp_weight_final = []
          for buffer in model_json['buffers']:
              temp_weight_final_row = []
              if "data" in buffer:
                  numpy_2d_arrays_final[y] = np.array(numpy_2d_arrays_final[y])
                  if y < 26 and (model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT8'):
                      temp_int = 1
                  else:
                      temp_int = 4
                  temp_numpy_2d_arrays = list(partition(temp_int,numpy_2d_arrays_final[y]))

                  for temp_list in range(len(temp_numpy_2d_arrays)):
                      temp_numpy_2d_arrays[temp_list] = list(temp_numpy_2d_arrays[temp_list])
                      temp_numpy_2d_arrays[temp_list] = bytearray(temp_numpy_2d_arrays[temp_list])

                      if y < 26 and model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT8':
                          temp_weight = struct.unpack('<b', temp_numpy_2d_arrays[temp_list])
                          temp_weight_final_row.append(temp_weight[0])
                      elif y < 26 and model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT32':
                          temp_weight = struct.unpack('<l', temp_numpy_2d_arrays[temp_list])
                          temp_weight_final_row.append(temp_weight[0])
                      elif y < 26 and model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'FLOAT32':
                          temp_weight = struct.unpack('<f', temp_numpy_2d_arrays[temp_list])
                          temp_weight_final_row.append(temp_weight[0])
                      elif y < 26 and model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'COMPLEX64':
                          input("Complex Number, needs to be implemented!")
                      else:
                          temp_weight_final_row.append(0)


              temp_weight_final.append(temp_weight_final_row)
              y = y + 1

      # If not first iteration, open data and sum information to already existing one
      # Only do this if arrays are different from the already exists
      else:
          numpy_2d_arrays = []
          for buffer in model_json['buffers']:
              if "data" in buffer:
                  numpy_2d_arrays.append(buffer['data'])
              else:
                  numpy_2d_arrays.append(np.array([]))

          y = 0
          for buffer in model_json['buffers']:
              if "data" in buffer:
                  numpy_2d_arrays[y] = np.array(numpy_2d_arrays[y])
                  if not np.array_equal(numpy_2d_arrays[y],  numpy_2d_arrays_final[y]):
                      print("CHANGEABLE BUFFER! NODE: " + str(x) + ", BUFFER NUMBER: " + str(y))
                      if model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT8':
                          temp_int = 1
                      else:
                          temp_int = 4
                      temp_numpy_2d_arrays = list(partition(temp_int,numpy_2d_arrays[y]))

                      for temp_list in range(len(temp_numpy_2d_arrays)):

                          temp_numpy_2d_arrays[temp_list] = list(temp_numpy_2d_arrays[temp_list])
                          temp_numpy_2d_arrays[temp_list] = bytearray(temp_numpy_2d_arrays[temp_list])

                          if model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT8':
                              temp_weight = struct.unpack('<b', temp_numpy_2d_arrays[temp_list])
                          elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT32':
                              temp_weight = struct.unpack('<l', temp_numpy_2d_arrays[temp_list])
                          elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'FLOAT32':
                              temp_weight = struct.unpack('<f', temp_numpy_2d_arrays[temp_list])
                          elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'COMPLEX64':
                              input("Complex Number, needs to be implemented!")
                          else:
                              input("Subgraph type not recognized! Type: " + model_json['subgraphs'][0]['tensors'][y-1]['type'])

                          temp_weight_final[y][temp_list] = temp_weight_final[y][temp_list] + temp_weight[0]
              y = y + 1

  # Divide each
  y = 0
  j = 0
  for buffer in model_json['buffers']:
      if "data" in buffer:
          print("Y: " + str(y))
          temp_array = []
          if not np.array_equal(numpy_2d_arrays[y],  numpy_2d_arrays_final[y]):
              print("BUFFER NUMBER: " + str(y))
              for j in range(len(temp_weight_final[y])):
                  temp_weight_final[y][j] = temp_weight_final[y][j] / valid_nodes_tflite_quantization_count

                  if model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT8':
                      temp_weight_final[y][j] = np.int8(temp_weight_final[y][j])
                      temp_bytearray = bytearray(struct.pack("<b", temp_weight_final[y][j]))
                      for i in temp_bytearray:
                          temp_array.append(i)
                  elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'INT32':
                      temp_weight_final[y][j] = np.int32(temp_weight_final[y][j])
                      temp_bytearray = bytearray(struct.pack("<l", temp_weight_final[y][j]))
                      for i in temp_bytearray:
                          temp_array.append(i)
                  elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'FLOAT32':
                      temp_weight_final[y][j] = np.float32(temp_weight_final[y][j])
                      temp_bytearray = bytearray(struct.pack("<f", temp_weight_final[y][j]))
                      for i in temp_bytearray:
                          temp_array.append(i)
                  elif model_json['subgraphs'][0]['tensors'][y-1]['type'] == 'COMPLEX64':
                      input("Complex Number, needs to be implemented!")
                  else:
                      input("Subgraph type not recognized! Type: " + model_json['subgraphs'][0]['tensors'][y-1]['type'])

              buffer['data'] = temp_array
      y = y + 1

  global_model_path_json = output_path + 'global_model.json'
  os.environ['global_model_path_json'] = global_model_path_json
  with open(global_model_path_json, 'w') as f:
      json.dump(model_json, f)

  os.system("flatc -b --strict-json --defaults-json -o ${output_path} ${schema} ${global_model_path_json}")

if use_quantization and nodes > 1 and valid_nodes_tflite_quantization_count > 0:

    global_model_path_tflite = output_path + 'global_model.tflite'

    #Predict
    interpreter = tf.lite.Interpreter(
                model_path = global_model_path_tflite,
                experimental_preserve_all_tensors = True
            )
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Run inference
    interpreter.allocate_tensors()

    all_output_data = []
    for y in range(len(xs_test)):
      interpreter.set_tensor(input_details[0]['index'], [xs_test[y]])
      interpreter.invoke()
      output_data = interpreter.get_tensor(output_details[0]['index'])
      all_output_data.append(output_data)

    y_prediction = np.argmax(all_output_data, axis = 2)
    #y_test=np.argmax(ys_test, axis=1)
    #Create confusion matrix and normalizes it over predicted (columns)
    show_confusion_matrix(ys_test, y_prediction, all_output_data, WORDS, -1, "tflite", "quantization")
