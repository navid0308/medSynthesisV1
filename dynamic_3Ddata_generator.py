import SimpleITK as sitk
from multiprocessing import Pool
import os, argparse
import h5py
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.init
from torch.autograd import Variable
import ast
import random
import copy

# input patch size
d1 = 64
d2 = 64
d3 = 64
# output patch size
dFA = [d1, d2, d3]  # size of patches of input data
dSeg = [64, 64, 64]  # size of patches of label data
# stride for extracting patches along the volume
step1 = 16
step2 = 16
step3 = 16
step = [step1, step2, step3]

def Generator_3D_slicesV2(t1_path, t2_path, batchsize,inputKey='T1',outputKey='T2'):
    t1_scan = ScanFile(t1_path, postfix='.nii.gz') # the specify item for your files, change to your own style
    t2_scan = ScanFile(t2_path, postfix='.nii.gz') # the specify item for your files, change to your own style
    t1_filenames = t1_scan.scan_files()
    t2_filenames = t2_scan.scan_files()

    while True:
        temp = list(zip(t1_filenames, t2_filenames))
        random.shuffle(temp)
        t1_filenames, t2_filenames = zip(*temp)
        for t1_filename, t2_filename in zip(t1_filenames, t2_filenames):
            source_fn = t1_filename
            target_fn = t2_filename

            imgOrg = sitk.ReadImage(source_fn)
            sourcenp = sitk.GetArrayFromImage(imgOrg)

            imgOrg1 = sitk.ReadImage(target_fn)
            targetnp = sitk.GetArrayFromImage(imgOrg1)
            
            maskimg = sourcenp

            stdPercentSource = np.std(sourcenp)
            meanPercentSource = np.mean(sourcenp)
            
            maxV, minV = np.percentile(sourcenp, [100, 0])
            meanPercentSource = minV
            stdPercentSource = maxV - minV
            
            matSource = (sourcenp - meanPercentSource) / stdPercentSource
            matTarget = (targetnp - meanPercentSource) / stdPercentSource

            sdir = t1_filename.split('/')
            #print 'sdir is, ', sdir, 'and s6 is, ', sdir[len(sdir)-1]
            lpet_fn = sdir[len(sdir)-1]
            words = lpet_fn.split('_')

            fileID = words[0]
            rate = 1
            matFA = matSource
            matSeg = matTarget
            matMask = maskimg
            
            [row, col, leng] = matFA.shape
            #estNum = 6000
            testFA = [] #np.zeros([estNum, 1, dFA[0], dFA[1], dFA[2]])
            testSeg = [] #np.zeros([estNum, 1, dSeg[0], dSeg[1], dSeg[2]])
            # to padding for input
            margin1 = (dFA[0] - dSeg[0]) // 2
            margin2 = (dFA[1] - dSeg[1]) // 2
            margin3 = (dFA[2] - dSeg[2]) // 2
            cubicCnt = 0
            marginD = [margin1, margin2, margin3]
            #print('matFA shape is ', matFA.shape)
            matFAOut = np.zeros([row + 2 * marginD[0], col + 2 * marginD[1], leng + 2 * marginD[2]])
            #print('matFAOut shape is ', matFAOut.shape)
            matFAOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matFA

            matSegOut = np.zeros([row + 2 * marginD[0], col + 2 * marginD[1], leng + 2 * marginD[2]])
            matSegOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matSeg

            matMaskOut = np.zeros([row + 2 * marginD[0], col + 2 * marginD[1], leng + 2 * marginD[2]])
            matMaskOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matMask

            # for mageFA, enlarge it by padding
            if margin1 != 0:
                matFAOut[0:marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matFA[marginD[0] - 1::-1, :,:]  # reverse 0:marginD[0]
                matFAOut[row + marginD[0]:matFAOut.shape[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matFA[matFA.shape[0] - 1:row - marginD[0] - 1:-1,:,:]  # we'd better flip it along the 1st dimension
            if margin2 != 0:
                matFAOut[marginD[0]:row + marginD[0], 0:marginD[1], marginD[2]:leng + marginD[2]] = matFA[:, marginD[1] - 1::-1,:]  # we'd flip it along the 2nd dimension
                matFAOut[marginD[0]:row + marginD[0], col + marginD[1]:matFAOut.shape[1], marginD[2]:leng + marginD[2]] = matFA[:,matFA.shape[1] - 1:col -marginD[1] - 1:-1,:]  # we'd flip it along the 2nd dimension
            if margin3 != 0:
                matFAOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], 0:marginD[2]] = matFA[:, :, marginD[2] - 1::-1]  # we'd better flip it along the 3rd dimension
                matFAOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], marginD[2] + leng:matFAOut.shape[2]] = matFA[:, :,matFA.shape[2] - 1:leng -marginD[2] - 1:-1]
                # for matseg, enlarge it by padding
            if margin1 != 0:
                matSegOut[0:marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matSeg[marginD[0] - 1::-1, :,:]  # reverse 0:marginD[0]
                matSegOut[row + marginD[0]:matSegOut.shape[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matSeg[matSeg.shape[0] - 1:row - marginD[0] - 1:-1, :,:] # we'd better flip it along the 1st dimension
            if margin2 != 0:
                matSegOut[marginD[0]:row + marginD[0], 0:marginD[1], marginD[2]:leng + marginD[2]] = matSeg[:,marginD[1] - 1::-1,:]  # we'd flip it along the 2nd dimension
                matSegOut[marginD[0]:row + marginD[0], col + marginD[1]:matSegOut.shape[1],marginD[2]:leng + marginD[2]] = matSeg[:, matSeg.shape[1] - 1:col - marginD[1] - 1:-1,
                                                :]  # we'd flip it along the 2nd dimension
            if margin3 != 0:
                matSegOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], 0:marginD[2]] = matSeg[:, :, marginD[2] - 1::-1]  # we'd better flip it along the 3rd dimension
                matSegOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1],
                marginD[2] + leng:matSegOut.shape[2]] = matSeg[:, :, matSeg.shape[2] - 1:leng - marginD[2] - 1:-1]

            # for matseg, enlarge it by padding
            if margin1 != 0:
                matMaskOut[0:marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matMask[marginD[0] - 1::-1, :,:]  # reverse 0:marginD[0]
                matMaskOut[row + marginD[0]:matMaskOut.shape[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matMask[matMask.shape[0] - 1:row - marginD[0] - 1:-1, :,:]  # we'd better flip it along the 1st dimension
            if margin2 != 0:
                matMaskOut[marginD[0]:row + marginD[0], 0:marginD[1], marginD[2]:leng + marginD[2]] = matMask[:,marginD[1] - 1::-1,:]  # we'd flip it along the 2nd dimension
                matMaskOut[marginD[0]:row + marginD[0], col + marginD[1]:matMaskOut.shape[1],marginD[2]:leng + marginD[2]] = matMask[:, matMask.shape[1] - 1:col - marginD[1] - 1:-1,
                                                :]  # we'd flip it along the 2nd dimension
            if margin3 != 0:
                matMaskOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], 0:marginD[2]] = matMask[:, :, marginD[2] - 1::-1]  # we'd better flip it along the 3rd dimension
                matMaskOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1],marginD[2] + leng:matMaskOut.shape[2]] = matMask[:, :, matMask.shape[2] - 1:leng - marginD[2] - 1:-1]

            eps = np.mean(matMaskOut) * 3
            for i in range(0, row - dSeg[0], step[0]):
                for j in range(0, col - dSeg[1], step[1]):
                    for k in range(0, leng - dSeg[2], step[2]):
                        volMask = matMaskOut[i:i + dSeg[0], j:j + dSeg[1], k:k + dSeg[2]]
                        if np.mean(volMask) < eps:
                            continue
                        cubicCnt = cubicCnt + 1
                        # index at scale 1
                        volSeg = matSegOut[i:i + dSeg[0], j:j + dSeg[1], k:k + dSeg[2]]
                        volFA = matFAOut[i:i + dFA[0], j:j + dFA[1], k:k + dFA[2]]

                        testFA.append(volFA)
                        testSeg.append(volSeg)
                        #testFA[cubicCnt, 0, :, :, :] = volFA  # 32*32*32
                        #testSeg[cubicCnt, 0, :, :, :] = volSeg  # 24*24*24

            testFA = np.array(testFA)
            testSeg = np.array(testSeg)

            dataMR=np.squeeze(testFA)
            dataCT=np.squeeze(testSeg)

            shapedata=dataMR.shape
            #Shuffle data
            idx_rnd=np.random.choice(shapedata[0], shapedata[0], replace=False)
            dataMR=dataMR[idx_rnd,...]
            dataCT=dataCT[idx_rnd,...]
            modulo=np.mod(shapedata[0],batchsize)
        ################## always the number of samples will be a multiple of batchsz##########################3            
            if modulo!=0:
                to_add=batchsize-modulo
                inds_toadd=np.random.randint(0,dataMR.shape[0],to_add)
                X=np.zeros((dataMR.shape[0]+to_add,dataMR.shape[1],dataMR.shape[2],dataMR.shape[3]))#dataMR
                #X=np.zeros((dataMR.shape[0]+to_add,dataMR.shape[1],dataMR.shape[2]))#dataMR
                X[:dataMR.shape[0],...]=dataMR
                X[dataMR.shape[0]:,...]=dataMR[inds_toadd]                
                
                #y=np.zeros((dataCT.shape[0]+to_add,dataCT.shape[1],dataCT.shape[2]))#dataCT
                y=np.zeros((dataCT.shape[0]+to_add,dataCT.shape[1],dataCT.shape[2],dataCT.shape[3]))#dataCT
                y[:dataCT.shape[0],...]=dataCT
                y[dataCT.shape[0]:,...]=dataCT[inds_toadd]
                
            else:
                X=np.copy(dataMR)                
                y=np.copy(dataCT)

            #X = np.expand_dims(X, axis=3)
            X=X.astype(np.float32)
            #y=np.expand_dims(y, axis=1)#B,H,W,C
            y=y.astype(np.float32)
            #y[np.where(y==5)]=0

            #shuffle the data, by dong
            inds = np.arange(X.shape[0])
            np.random.shuffle(inds)
            X=X[inds,...]
            y=y[inds,...]

            print('x shape ', X.shape)
            print('y shape ', y.shape)                   
            for i_batch in range(int(X.shape[0]/batchsize)):
                yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...],  y[i_batch*batchsize:(i_batch+1)*batchsize,...])

class ScanFile(object):
    def __init__(self, directory, prefix=None, postfix=None):
        self.directory = directory
        self.prefix = prefix
        self.postfix = postfix

    def scan_files(self):
        files_list = []

        for dirpath, dirnames, filenames in os.walk(self.directory):
            for special_file in filenames:
                if self.postfix:
                    if special_file.endswith(self.postfix):
                        files_list.append(os.path.join(dirpath, special_file))
                elif self.prefix:
                    if special_file.startswith(self.prefix):
                        files_list.append(os.path.join(dirpath, special_file))
                else:
                    files_list.append(os.path.join(dirpath, special_file))

        return files_list

    def scan_subdir(self):
        subdir_list = []
        for dirpath, dirnames, files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list