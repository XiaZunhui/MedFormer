import logging
import os

def logging_save(save_path, type):
    # 配置日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建文件处理程序，输出到文件
    if type == 'train':
        file_handler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
    elif type == 'test':
        file_handler = logging.FileHandler(os.path.join(save_path, 'log_result.txt'))
    else:
        raise ValueError("Invalid value for 'type'. 'type' must be either 'train' or 'test'.")
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理程序，输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理程序添加到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
