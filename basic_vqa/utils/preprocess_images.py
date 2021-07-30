import os
import argparse
import h5py
import numpy as np
from PIL import Image

def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def get_split_name( dirname ):
    '''
    Get split from dirname
    '''
    if 'train' in dirname:
        return 'train'
    if 'val' in dirname:
        return 'val'
    if 'test' in dirname:
        return 'test'
    return Exception( f'Unrecognized split: {dirname}' )

def resize_images(input_dir, out_file, size, fd):
    """
    Resize the images in 'input_dir' and save into 'output_dir'
    as hdf5 file.
    """
    for idir in os.scandir(input_dir):
        if not idir.is_dir():
            continue
        # create new group for split
        split = get_split_name( idir.name )
        group = fd.create_group( split )
        images = os.listdir(idir.path)
        n_images = len(images)
        # create an image and id dataset for each split
        image_set = group.create_dataset( 'images',
            (n_images, size, size, 3), dtype='uint8' )
        coco_id_set = group.create_dataset( 'coco_ids',
            (n_images,), dtype='int32' )
        # loop through images, resize and save to h5 file
        for iimage, image in enumerate(images):
            coco_id = int( image.split('_')[-1].split('.')[0] )
            try:
                with open(os.path.join(idir.path, image), 'r+b') as f:
                    with Image.open(f) as img:
                        img = img.convert('RGB')
                        img = np.array( resize_image(img, (size, size)) )
                        assert img.shape == ( size, size, 3 )
                        assert np.max( img ) <= 255
                        image_set[iimage] = img
                        coco_id_set[iimage] = coco_id
            except(IOError, SyntaxError) as e:
                # skip over error
                # import pdb; pdb.set_trace()
                print( f'Error on i: {iimage} image: {image} error: {e}' )
            
            if (iimage+1) % 100 == 0:
                print("[{}/{}] Resized the images and saved into '{}'."
                      .format(iimage+1, n_images, out_file))

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    image_size = args.image_size
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join( output_dir, f'images.h5' )
    print( f'Resizing images to ({image_size}, {image_size})' )

    with h5py.File( out_file, 'w', libver='latest' ) as fd:
        resize_images(input_dir, out_file, image_size, fd)
    
    print( 'Done!' )    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='../../../data/vqa/Images',
                        help='directory for input images (unresized images)')

    parser.add_argument('--output_dir', type=str, default='../../../data/vqa/hdf5',
                        help='directory for output images (resized images)')

    parser.add_argument('--image_size', type=int, default=224,
                        help='size of images after resizing')

    args = parser.parse_args()

    main(args)
