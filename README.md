# English_OCR

## Overview
This project focuses on the development of a model for English letter classification, providing a comprehensive approach to model training and data preprocessing. The initial dataset includes pictures of letters (PNG format) set against a white background.



## Data Preprocessing Steps
1. **Image Resizing**: Each image was resized to a 28x28 format.
2. **Conversion to Numpy Array**: The resized images were then converted into numpy arrays.
3. **Background Inversion and Normalization**: The arrays were processed using `np.where(data > 123, 0, 1)` to convert the black image on a white background into a white image on a black background. This step also normalizes the images, so the array now only consists of 1s and 0s.
4. **DataFrame Creation**: A DataFrame was created where each row represents an image, and one column represents one pixel. A labels column was added to include the image labels.
5. **Data Trimming**: Due to unequal distribution across classes, the data was trimmed to ensure a balanced dataset.
6. **Format Transformation**: The trimmed data was then transformed into a suitable format for model training.
## Viewing Images from the Dataset

To visualize an actual image from the dataset, you need to perform the following steps:

1. **Select a row** representing the image from your dataset. Each row contains pixel values for one image.

2. **Reshape the row** into a 28x28 format. This corresponds to the original dimensions of the image.

3. **Scale the pixel values** by multiplying by 255. Since we normalized our images to 1s and 0s, we need to scale them back to the original 0-255 range for proper visualization.

4. **Convert to uint8 format** to match the expected input format for image visualization libraries.

5. **Display the image** using the Python Imaging Library (Pillow).

Here's an example Python code snippet demonstrating these steps:

```python
import numpy as np
from PIL import Image

# Assuming 'row' is a selected row from the DataFrame
row = np.array(row, dtype=np.float32)  # Convert row to a numpy array if it's not already
image_array = row.reshape(28, 28) * 255  # Reshape and scale
image_array = image_array.astype(np.uint8)  # Convert to uint8

# Create and display the image
image = Image.fromarray(image_array)
image.show()
```

## Model Training
The model training process, along with further preprocessing steps, is detailed within the accompanying Jupyter notebook.

## Notebook
The notebook included in this repository (`English_OCR.ipynb`) walks through the step-by-step process of model training and data preprocessing in detail.

## How to Use
1. Clone this repository to your local machine.
2. Ensure you have Jupyter Notebook installed, or use [Google Colab](https://colab.research.google.com/) for an online environment.
3. Open `English_OCR.ipynb` and follow the steps outlined to train the model.

## Requirements
- Python 3.x
- Jupyter Notebook
- Numpy
- Pandas
- tensorflow
- Pillow (optional)


## Contributing
Contributions to the English_OCR project are welcome. Please refer to the contributing guidelines for more information.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

For more information and any queries, please open an issue in this repository.
