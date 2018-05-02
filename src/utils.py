import torch
import os
import codecs
import datetime
import logging
import pickle


"""
@author: Michael Guarino
"""


class prjPaths:
    def __init__(self, getDataset=True):
        self.SRC_DIR = os.path.abspath(os.path.curdir)
        self.ROOT_MOD_DIR = "/".join(self.SRC_DIR.split("/")[:-1])
        self.LIB_DIR = os.path.join(self.ROOT_MOD_DIR, "lib")
        self.CHECKPOINT_DIR = os.path.join(self.LIB_DIR, "chkpts")
        self.PERSIST_DIR = os.path.join(self.LIB_DIR, "persist")
        self.LOGS_DIR = os.path.join(self.LIB_DIR, "logs")

        pth_exists_else_mk = lambda path: os.mkdir(path) if not os.path.exists(path) else None

        pth_exists_else_mk(self.LIB_DIR)
        pth_exists_else_mk(self.CHECKPOINT_DIR)
        pth_exists_else_mk(self.PERSIST_DIR)
        pth_exists_else_mk(self.LOGS_DIR)
    # end
# end

def generate_data(gpu_available, batch_size, n):
    if gpu_available:
        return [torch.randint(0, 2, (batch_size, n), dtype=torch.float).cuda()*2-1,
                torch.randint(0, 2, (batch_size, n), dtype=torch.float).cuda()*2-1]
    else:
        return [torch.randint(0, 2, (batch_size, n), dtype=torch.float)*2-1,
                torch.randint(0, 2, (batch_size, n), dtype=torch.float)*2-1]
# end

def UTF_8_to_binary(p_utf_8):

    # utf-8 -> binary
    p_bs = " ".join(format(ord(x), "08b") for x in p_utf_8).split(" ")
    return p_bs
# end

def binary_to_UTF_8(p_bs):

    # binary string -> ord
    p_ords = [int(p_b, 2) for p_b in p_bs]

    # ord -> hex "0x68"[2:] must slice to be valid hex
    p_hexs = [hex(p_ord)[2:] for p_ord in p_ords]

    # hex -> utf-8
    decoded = "".join([codecs.decode(p_hex.strip(), "hex").decode("utf-8") for p_hex in p_hexs])
    return decoded
# end

def get_logger(log_dir, run_type):

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(os.path.join(log_dir, run_type)):
        os.mkdir(os.path.join(log_dir, run_type))

    current_Time = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
    fileHandler = logging.FileHandler(os.path.join(log_dir, run_type, "%s_%s.log"%(run_type,current_Time)))
    fileHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)

    return logger
# end

def persist_object(full_path, x):
    with open(full_path, "wb") as file:
        pickle.dump(x, file)
# end

def restore_persist_object(full_path):
    with open(full_path, "rb") as file:
        x = pickle.load(file)
    return x
# end