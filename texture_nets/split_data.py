import numpy as np
import pandas as pd
import sys
data_file = sys.argv[1]
seeds_count = int(sys.argv[2])
data = pd.read_csv(data_file)
np.random.seed(0)
image_names = np.unique(data['seed_img_name'])
selected_images = np.random.choice(image_names, seeds_count, False)
selected_data = data[data['seed_img_name'].isin(selected_images)]
selected_data.to_csv(data_file[:-4] + '_'  + str(seeds_count) + '.txt',  index = False)
