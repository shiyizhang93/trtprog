"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import argparse
import os.path

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# For our custom calibrator
from calibrator import YOLOEntropyCalibrator


class ModelData(object):
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        msg (str): Human readable string describing the exception.
        code (:obj:`int`, optional): Error code.

    Attributes:
        msg (str): Human readable string describing the exception.
        code (int): Exception error code.

    """
    INPUT_NAME = "images"
    OUTPUT_NAME = "output"
    OUTPUT_LANC_NAME = "output_l"
    OUTPUT_MANC_NAME = "output_m"
    OUTPUT_SANC_NAME = "output_s"
    # The original model is a float32 one.
    DTYPE = trt.float32


def build_plan(input_name="images",
               onnx="../onnx/yolov5n_384x640_op12_dyn_sim.onnx",
               input_channel=3,
               img_size=(384,640),
               workspace=1,
               dynamic=False,
               min_batch_size=1,
               opt_batch_size=4,
               max_batch_size=16,
               fp32=False,
               fp16=False,
               int8=False,
               dla=-1,
               calib_path="../dataPTQ/",
               cache="./plan.cache",
               verbose=False):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    # Parse ONNX model
    parser = trt.OnnxParser(network, logger)
    runtime = trt.Runtime(logger)
    print('Loading ONNX file from path {}'.format(onnx))

    if not os.path.isfile(onnx):
        raise ValueError('Please type the valid onnx file path!')

    print('Beginning ONNX file parsing')
    success = parser.parse_from_file(onnx)
    for idx in range(parser.num_errors):
        print(parser.get_errors(idx))
    if not success:
        raise TypeError("ONNX Parser parse failed.")
    print('Completed ONNX parser')
    config = builder.create_builder_config()
    # Set workspace for builder when building an optimized engine
    config.max_workspace_size = workspace << 30
    # Create optimization profile
    profile = builder.create_optimization_profile()
    if dynamic:
        # Set Dynamic Batch config
        profile.set_shape(input_name,
                          (min_batch_size, input_channel, img_size[0], img_size[1]),
                          (opt_batch_size, input_channel, img_size[0], img_size[1]),
                          (max_batch_size, input_channel, img_size[0], img_size[1]))
        config.add_optimization_profile(profile)
    else:
        # Set inference batch size if not dynamic batch
        builder.max_batch_size = min_batch_size
        profile.set_shape(input_name,
                          (min_batch_size, input_channel, img_size[0], img_size[1]),
                          (min_batch_size, input_channel, img_size[0], img_size[1]),
                          (min_batch_size, input_channel, img_size[0], img_size[1]))
        config.add_optimization_profile(profile)


    if int8:
        if not os.path.isdir(calib_path):
            raise ValueError("The calibrator images directory is invalid!")
        calib = YOLOEntropyCalibrator(calibration_images=calib_path,
                                      cache=cache,
                                      batch_size=opt_batch_size,
                                      input_channel=input_channel,
                                      img_size=img_size,
                                      dtype=ModelData.DTYPE)
        # Set INT8 config
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib
        if dynamic:
            config.set_calibration_profile(profile)
            ret = config.get_calibration_profile()
    elif fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Set DLA Core config
    if dla >= 0:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = dla
        # config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        print('Using DLA core %d.' % dla)

    # Build engine
    print('Building a plan from {}; \n'
          'this may take a while ...'.format(onnx))
    plan = builder.build_engine(network, config)
    print('Completed plan building.')
    print("Engine dimension: ", plan.get_binding_shape(0))

    return plan


def serialize_plan(trt_plan, plan_path):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    s = os.path.split(plan_path)
    if not os.path.isdir(s[0]):
        os.makedirs(s[0])
    with open(plan_path, 'wb') as f:
        f.write(trt_plan.serialize())
    print('Serialized the TensorRT engine to file: %s' % plan_path)


def run(input_name="images",
        onnx="../onnx/yolov5n_384x640_op12_dyn_sim.onnx",
        inputch=3,
        inputsz=(384,640),
        plan="../plan/yolov5n_384x640.plan",
        workspace=1,
        dynamic=False,
        min_batch_size=1,
        opt_batch_size=4,
        max_batch_size=16,
        fp32=False,
        fp16=False,
        int8=False,
        calimg="../dataPTQ/",
        cache="./plan.cache",
        dla=-1,
        verbose=False):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    trt_plan = build_plan(input_name=input_name,
                          onnx=onnx,
                          input_channel=inputch,
                          img_size=inputsz,
                          workspace=workspace,
                          dynamic=dynamic,
                          min_batch_size=min_batch_size,
                          opt_batch_size=opt_batch_size,
                          max_batch_size=max_batch_size,
                          fp32=fp32,
                          fp16=fp16,
                          int8=int8,
                          dla=dla,
                          calib_path=calimg,
                          cache=cache,
                          verbose=verbose)
    # serialize the built plan to the path with customized filname
    serialize_plan(trt_plan, plan_path=plan)


def parse_opt():
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default="images", help='fill the model input')
    parser.add_argument('--onnx', type=str, default="../onnx/yolov5n_384x640_op12_dyn_sim.onnx", help='path to onnx file')
    parser.add_argument('--inputch', type=int, default=3, help='input c')
    parser.add_argument('--inputsz', type=int, default=[384, 640], help='input (h, w)')
    parser.add_argument('--plan', type=str, default="../plan/yolov5n_384x640.plan", help='path to save plan file')
    parser.add_argument('--workspace', type=int, default=1, help='workspace for engine building in GB unit')
    parser.add_argument('--dynamic', action='store_true', help='dynamic axes')
    parser.add_argument('--min_batch_size', type=int, default=1, help='minimum batch size for dynamic shapes')
    parser.add_argument('--opt_batch_size', type=int, default=4, help='optimization batch size for dynamic shapes')
    parser.add_argument('--max_batch_size', type=int, default=16, help='maximum batch size for dynamic shapes')
    parser.add_argument('--fp32', action='store_true', help='FP32 weights')
    parser.add_argument('--fp16', action='store_true', help='FP16 quantization')
    parser.add_argument('--int8', action='store_true', help='INT8 quantization')
    parser.add_argument('--calimg', type=str, default="../dataPTQ", help='directory contained calibrator images')
    parser.add_argument('--cache', type=str, default="./plan.cache", help='quantization calibrator cache filename')
    parser.add_argument('--dla', type=int, default=-1, help='dla 0 or 1 enabling')
    parser.add_argument('--verbose', action='store_true', help='building plan information')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
