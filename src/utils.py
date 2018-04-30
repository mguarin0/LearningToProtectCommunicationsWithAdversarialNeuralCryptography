import torch
import os
import codecs

"""
@author: Michael Guarino
"""


class prjPaths:
    def __init__(self, getDataset=True):
        self.SRC_DIR = os.path.abspath(os.path.curdir)
        self.ROOT_MOD_DIR = "/".join(self.SRC_DIR.split("/")[:-1])
        self.LIB_DIR = os.path.join(self.ROOT_MOD_DIR, "lib")
        self.CHECKPOINT_DIR = os.path.join(self.LIB_DIR, "chkpts")
        self.LOGS_DIR = os.path.join(self.LIB_DIR, "logs")

        pth_exists_else_mk = lambda path: os.mkdir(path) if not os.path.exists(path) else None

        pth_exists_else_mk(self.LIB_DIR)
        pth_exists_else_mk(self.CHECKPOINT_DIR)
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
    # TODO if binary has leading 0 then it will be omitted...this must be fixed

    # utf-8 -> binary
    p_bs = " ".join(format(ord(x), 'b') for x in p_utf_8).split(" ")
    return p_bs
# end

def binary_to_UTF_8(p_bs):

    # binary string -> ord
    p_ords = [int(p_b, 2) for p_b in p_bs]
    #print("p_ords: {}".format(p_ords))

    # ord -> hex "0x68"[2:] must slice to be valid hex
    p_hexs = [hex(p_ord)[2:] for p_ord in p_ords]
    #print("p_hexs: {}".format(p_hexs))

    # hex -> utf-8
    decoded = [codecs.decode(p_hex, "hex").decode("utf-8") for p_hex in p_hexs]
    return decoded
# end
