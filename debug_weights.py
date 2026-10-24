import h5py
import numpy as np

f = h5py.File('B7/efficientnetb7_baseline_fold2.weights.h5', 'r')
layers = f['layers']
layer_names = list(layers.keys())
print(f'Total layers in weights file: {len(layer_names)}')
print()

# Show the last 10 layers (top classifier layers)
for name in layer_names[-10:]:
    grp = layers[name]
    if 'vars' in grp:
        vars_keys = list(grp['vars'].keys())
        shapes = []
        for vk in vars_keys:
            s = grp['vars'][vk].shape
            shapes.append(f'{vk}: {s}')
        print(f'{name} -> vars: {shapes}')
    else:
        print(f'{name} -> {list(grp.keys())}')

print()

# Check target layer names
for target in ['dense', 'dense_1', 'batch_normalization']:
    if target in layer_names:
        grp = layers[target]
        if 'vars' in grp:
            vars_keys = list(grp['vars'].keys())
            shapes = [str(grp['vars'][vk].shape) for vk in vars_keys]
            print(f'FOUND: {target} -> vars: {shapes}')
        else:
            print(f'FOUND: {target} -> no vars')
    else:
        matches = [n for n in layer_names if target in n.lower()]
        print(f'NOT FOUND: {target}, similar: {matches[-5:] if matches else "none"}')

print()

# Now check what the Keras 2 model layer names look like
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

base_model = tf.keras.applications.EfficientNetB7(weights='imagenet', include_top=False, input_shape=(600, 600, 3))
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(256, activation='relu', name='dense')(x)
x = tf.keras.layers.BatchNormalization(name='batch_normalization')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(2, activation='softmax', dtype='float32', name='dense_1')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

print('Model top layers (last 6):')
for layer in model.layers[-6:]:
    w_shapes = [str(w.shape) for w in layer.get_weights()] if layer.get_weights() else 'no weights'
    print(f'  {layer.name}: {w_shapes}')

print()
print('Attempting direct load_weights...')
try:
    model.load_weights('B7/efficientnetb7_baseline_fold2.weights.h5')
    print('  SUCCESS: Direct load worked!')
except Exception as e:
    print(f'  FAILED: {str(e)[:200]}')

print()
print('Attempting by_name load_weights...')
try:
    model.load_weights('B7/efficientnetb7_baseline_fold2.weights.h5', by_name=True, skip_mismatch=True)
    print('  SUCCESS: by_name load completed (but may have skipped layers!)')
except Exception as e:
    print(f'  FAILED: {str(e)[:200]}')

# Check if dense weights changed after by_name load
dense_w = model.get_layer('dense').get_weights()
print(f'  dense weights[0] mean: {np.mean(dense_w[0]):.6f}, std: {np.std(dense_w[0]):.6f}')
print(f'  dense weights[0] first 5 values: {dense_w[0].flatten()[:5]}')

# Now try h5py manual loading
print()
print('Attempting h5py manual loading of top layers...')
h5 = f['layers']

if 'dense' in h5 and 'vars' in h5['dense']:
    vg = h5['dense']['vars']
    keys = sorted(vg.keys(), key=lambda x: int(x))
    w_h5 = [np.array(vg[k]) for k in keys]
    print(f'  H5 dense weights: {[w.shape for w in w_h5]}')
    print(f'  H5 dense weights[0] mean: {np.mean(w_h5[0]):.6f}, std: {np.std(w_h5[0]):.6f}')
    print(f'  H5 dense weights[0] first 5: {w_h5[0].flatten()[:5]}')
    
    # Compare with current model weights
    curr_w = model.get_layer('dense').get_weights()
    is_same = np.allclose(curr_w[0], w_h5[0])
    print(f'  Dense weights match after by_name load: {is_same}')

if 'dense_1' in h5 and 'vars' in h5['dense_1']:
    vg = h5['dense_1']['vars']
    keys = sorted(vg.keys(), key=lambda x: int(x))
    w_h5 = [np.array(vg[k]) for k in keys]
    print(f'  H5 dense_1 weights: {[w.shape for w in w_h5]}')

f.close()
