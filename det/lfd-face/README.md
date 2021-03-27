# LFD Face Detection

## Requirements
Follow the [README](../../README.md)
Special dependency:
- albumentations(pip install albumentations)

## Run
```python
python main.py
```

If you encounter some problems in program execution, please try the following solutions first.
> * turbo err:
  Build libjpeg-turbo
>   1. download the [source code v2.0.5](https://sourceforge.net/projects/libjpeg-turbo/files/)
>   2. decompress and compile:
     `cd [source code]`  
      `mkdir build`  
      `cd build`  
      `cmake ..`  
      `make`
>      
>      make sure that `cmake` configuration properly
>   3. copy `build/libturbojpeg.so.x.x.x` to `lfd/data_pipeline/dataset/utils/libs`

> * nms err:
  In the lfd-face root, run the code below:
  `python setup.py build_ext`
>
>   Once successful, you will see: `----> build and copy successfully!`